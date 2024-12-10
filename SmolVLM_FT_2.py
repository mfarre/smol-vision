import os
import json
import random
# import cv2
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import List, Dict, Any
import time
from pathlib import Path
from queue import Queue
from threading import Event
import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset
import argparse

from transformers import (
    AutoProcessor, 
    BitsAndBytesConfig, 
    Idefics3ForConditionalGeneration,
    TrainingArguments, 
    Trainer
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import wandb 
import decord
from decord import VideoReader

# GPU config
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoFrameExtractor:
    def __init__(self, max_frames: int = 50):
        self.max_frames = max_frames
        
    def resize_and_center_crop(self, image: Image.Image, target_size: int) -> Image.Image:
        """Resize the image preserving aspect ratio and then center crop."""
        width, height = image.size
        
        if width < height:
            new_width = target_size
            new_height = int(height * (target_size / width))
        else:
            new_height = target_size
            new_width = int(width * (target_size / height))
            
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        left = (new_width - target_size) // 2
        top = (new_height - target_size) // 2
        right = left + target_size
        bottom = top + target_size
        
        return image.crop((left, top, right, bottom))

    def extract_frames_from_video(self, video_path: str) -> List[Image.Image]:
        """Extract frames from video file using decord"""
        decord.bridge.set_bridge('torch')  # Using torch bridge for better GPU support
        
        try:
            # Load video with decord
            vr = VideoReader(video_path)
            total_frames = len(vr)
            
            # Calculate frame indices for sampling
            if total_frames <= self.max_frames:
                frame_indices = list(range(total_frames))
            else:
                # Sample frames evenly
                frame_indices = np.linspace(0, total_frames - 1, self.max_frames, dtype=int).tolist()
            
            # Read frames
            frames = vr.get_batch(frame_indices)
            
            # Convert to PIL and process
            processed_frames = []
            for frame in frames:
                # Convert from torch tensor to PIL
                frame = frame.numpy()
                pil_image = Image.fromarray(frame)
                pil_image = self.resize_and_center_crop(pil_image, 384)
                processed_frames.append(pil_image)
                
            return processed_frames
            
        except Exception as e:
            print(f"Error processing video {video_path}: {str(e)}")
            raise

    def extract_frames_from_gif(self, gif_path: str) -> List[Image.Image]:
        """Extract frames from GIF file"""
        gif = Image.open(gif_path)
        frames = []
        try:
            while True:
                frames.append(gif.copy())
                gif.seek(gif.tell() + 1)
        except EOFError:
            pass

        if len(frames) > self.max_frames:
            indices = np.linspace(0, len(frames) - 1, self.max_frames, dtype=int)
            frames = [frames[i] for i in indices]

        return [self.resize_and_center_crop(frame.convert('RGB'), 384) for frame in frames]

    def extract_frames_from_folder(self, folder_path: str) -> List[Image.Image]:
        """Extract frames from folder containing numbered image files"""
        image_extensions = ('.jpg', '.jpeg', '.png')
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]
        
        # Extract base filename and number using regex
        import re
        pattern = re.compile(r'(.+?)[-_]?(\d+)\..*$')
        numbered_files = []
        
        for file in files:
            match = pattern.match(file)
            if match:
                base, num = match.groups()
                numbered_files.append((file, base, int(num)))
        
        if not numbered_files:
            raise ValueError(f"No valid numbered image files found in {folder_path}")
            
        # Sort by base filename and number
        numbered_files.sort(key=lambda x: (x[1], x[2]))
        sorted_files = [f[0] for f in numbered_files]
        # Sample frames evenly if needed
        if len(sorted_files) > self.max_frames:
            indices = np.linspace(0, len(sorted_files) - 1, self.max_frames, dtype=int)
            sorted_files = [sorted_files[i] for i in indices]
        
        frames = []
        for file in sorted_files:
            image_path = os.path.join(folder_path, file)
            image = Image.open(image_path)
            image = self.resize_and_center_crop(image.convert('RGB'), 384)
            frames.append(image)
            
        return frames

    def extract_frames(self, path: str) -> List[Image.Image]:
        """Extract frames from video file, GIF, or folder of images"""
        if os.path.isdir(path):
            return self.extract_frames_from_folder(path)
        elif path.lower().endswith('.gif'):
            return self.extract_frames_from_gif(path)
        else:
            return self.extract_frames_from_video(path)

class VideoQADataset(Dataset):
    def __init__(self, processor, max_frames: int = 50, split="train"):
        self.processor = processor
        self.frame_extractor = VideoFrameExtractor(max_frames)
        
        # Load HF dataset
        self.data = load_dataset("HuggingFaceFV/longvumix", split=split)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path = os.path.join('/fsx/miquel/longvu-dataset/composition', self.data[idx]['video'])
        rank = torch.distributed.get_rank()
        logger.info(f"[Rank {rank}] Loading video: {video_path}")

        frames = self.frame_extractor.extract_frames(video_path)
        if len(frames) == 0:
            logger.info(f"Zero frames for {video_path}")
        conversations = self.data[idx]['conversations']
        question = next(conv for conv in conversations if conv['from'] == 'human')['value'].split('<image>\n')[0]
        answer = next(conv for conv in conversations if conv['from'] == 'gpt')['value']
            
        return {
            'frames': frames,
            'question': question,
            'answer': answer
        }
    
def is_main_process():
    return os.environ.get('LOCAL_RANK', '0') == '0'

def is_main_process_multi_node():
    # Check if this is the main process across all nodes
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    global_rank = int(os.environ.get('RANK', '0'))
    return local_rank == 0 and global_rank == 0


def find_global_img_patterns(tokens):
    # Find positions where "< global - img >" pattern appears
    mask_positions = []
    for i in range(len(tokens) - 4):
        if (tokens[i] == '<' and 
            tokens[i+1] == 'global' and 
            tokens[i+2] == '-' and 
            tokens[i+3] == 'img' and 
            tokens[i+4] == '>'):
            mask_positions.extend([i, i+1, i+2, i+3, i+4])
    return mask_positions

def video_collate_fn(examples, processor, max_frames, use_temporal_tokens=True):
    texts = []
    images_list = []
    
    for example in examples:
        frames = example['frames']
        question = example['question']
        answer = example['answer']
        
        # Add temporal tokens between frames
        image_tokens = []
        for i in range(len(frames)):
            image_tokens.append({"type": "image"})
            if i < len(frames) - 1 and use_temporal_tokens:
                # Add temporal token between frames
                image_tokens.append({"type": "text", "text": f"<frame_{i}>"})
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Answer briefly."},
                    *image_tokens,
                    {"type": "text", "text": question}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer}
                ]
            }
        ]

        text = processor.apply_chat_template(messages, add_generation_prompt=False)
        example_images = [img for img in frames]
        
        texts.append(text.strip())
        images_list.append(example_images)  # Keep images grouped by example
    
    batch = processor(text=texts, images=images_list, return_tensors="pt", padding=True)



    # Handle labels
    image_token_id = processor.tokenizer.additional_special_tokens_ids[
        processor.tokenizer.additional_special_tokens.index("<image>")]
    fake_token_around_image_id = processor.tokenizer.additional_special_tokens_ids[
        processor.tokenizer.additional_special_tokens.index("<fake_token_around_image>")]
    end_of_utterance_id = processor.tokenizer.additional_special_tokens_ids[
        processor.tokenizer.additional_special_tokens.index("<end_of_utterance>")]

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == image_token_id] = -100
    labels[labels == fake_token_around_image_id] = -100
    labels[labels == end_of_utterance_id] = -100

    # Mask out temporal tokens in labels
    if use_temporal_tokens:
        temporal_token_ids = [processor.tokenizer.convert_tokens_to_ids(f"<frame_{i}>") 
                            for i in range(max_frames)]
        for token_id in temporal_token_ids:
            labels[labels == token_id] = -100

    # Mask global-img patterns for each sequence in the batch
    for i in range(len(batch["input_ids"])):
        tokens = processor.tokenizer.convert_ids_to_tokens(batch["input_ids"][i])
        positions_to_mask = find_global_img_patterns(tokens)
        
        # Mask the identified positions
        for pos in positions_to_mask:
            token = tokens[pos]
            token_id = processor.tokenizer.convert_tokens_to_ids(token)
            labels[i, pos] = -100

    batch["labels"] = labels
    
    return batch

def main():
    parser = argparse.ArgumentParser(description='Video-LLM Training')
    parser.add_argument('--max_frames', type=int, default=50, help='Maximum number of frames per video')
    parser.add_argument('--temporal_tokens', action='store_true', help='Enable temporal tokens between frames')
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--use_lora', action='store_true', default = False, help='Enable LoRA training')
    parser.add_argument('--use_qlora', action='store_true', default = False, help='Enable QLoRA training')
    args = parser.parse_args()

    # Initialize distributed environment
    torch.distributed.init_process_group(backend='nccl')
    
    # Configuration
    model_id = "HuggingFaceTB/SmolVLM_converted_4"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Generate output directory
    temporal_str = "with_temp" if args.temporal_tokens else "no_temp"
    output_dir = f"./smolvlm_frames{args.max_frames}_{temporal_str}_lr_1e-5"
    
    # Print configuration if main process
    if is_main_process_multi_node():
        print("\n=== Training Configuration ===")
        print(f"Max frames per video: {args.max_frames}")
        print(f"Temporal tokens: {'enabled' if args.temporal_tokens else 'disabled'}")
        print(f"Wandb: {'enabled' if args.wandb else 'disabled'}")
        print(f"Output directory: {output_dir}")
        print("===========================\n")

    # Initialize wandb
    if args.wandb and is_main_process_multi_node():
        wandb.init(
            project="smolvlm-longvumix",
            name=os.path.basename(output_dir),
            config={
                "model_id": model_id,
                "max_frames": args.max_frames,
                "temporal_tokens": args.temporal_tokens,
                "use_lora": args.use_lora,
                "use_qlora": args.use_qlora,
                "device": DEVICE,
            }
        )

    # Initialize processor and model
    processor = AutoProcessor.from_pretrained(model_id)

    # Special config for video
    processor.image_processor.size = (384, 384)
    processor.image_processor.do_resize = False
    processor.image_processor.do_image_splitting = False

    # Adding frame special tokens
    existing_tokens = processor.tokenizer.additional_special_tokens

    # Add temporal tokens if enabled
    if args.temporal_tokens:
        new_temporal_tokens = [f"<frame_{i}>" for i in range(args.max_frames)]
        all_special_tokens = existing_tokens + new_temporal_tokens
        processor.tokenizer.add_special_tokens({"additional_special_tokens": all_special_tokens})

    if args.use_qlora or args.use_lora:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules=['down_proj','o_proj','k_proj','q_proj','gate_proj','up_proj','v_proj'],
            use_dora=False if args.use_qlora else True,
            init_lora_weights="gaussian"
        )
        lora_config.inference_mode = False
        
        if args.use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            
        model = Idefics3ForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config if args.use_qlora else None,
            _attn_implementation="flash_attention_2",
            use_memory_efficient_attention=True,
            device_map="auto"
        )
        model.add_adapter(lora_config)
        model.enable_adapters()
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        print(model.get_nb_trainable_parameters())
    else:
        model = Idefics3ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            _attn_implementation="flash_attention_2",
        ).to(DEVICE)

    # Resize token embeddings to account for new tokens
    model.resize_token_embeddings(len(processor.tokenizer))

    # Enable gradient checkpointing to reduce memory usage
    model.gradient_checkpointing_enable()

    # Initialize dataset
    train_dataset = VideoQADataset(processor, args.max_frames, split="train")
    train_sampler = DistributedSampler(train_dataset)
    


    # Training arguments
    num_nodes = int(16)
    training_args = TrainingArguments(
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16//num_nodes,
        warmup_ratio = 0.15,
        max_grad_norm = 2.0,
        learning_rate=1e-5 * num_nodes, #Original was 5e-6, 5e-5 bounces back. Trying 5e-7
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        logging_steps=20,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=30,
        optim="adamw_torch" if not (args.use_lora or args.use_qlora) else "paged_adamw_8bit",
        bf16=True,
        output_dir=output_dir,
        remove_unused_columns=False,
        report_to="wandb" if args.wandb else "none",
        logging_dir="./logs",
        logging_first_step=True,
        dataloader_drop_last=True,
    )

    # Save added new tokens
    processor.tokenizer.save_pretrained(training_args.output_dir)

    # Pass temporal tokens flag to collate_fn through a lambda
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=lambda examples: video_collate_fn(examples, processor, args.max_frames, use_temporal_tokens=args.temporal_tokens),
    )

    # Train
    trainer.train()
    
    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()