import os
import json
import random
import cv2
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

from transformers import (
    AutoProcessor, 
    BitsAndBytesConfig, 
    Idefics3ForConditionalGeneration,
    TrainingArguments, 
    Trainer
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import wandb 

# GPU config
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"  # Use GPUs 0-3
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
        """Extract frames from video file (mp4/mov)"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_indices = list(range(0, total_frames, fps))
        
        if len(frame_indices) > self.max_frames:
            indices = np.linspace(0, len(frame_indices) - 1, self.max_frames, dtype=int)
            frame_indices = [frame_indices[i] for i in indices]
        
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame)
                pil_image = self.resize_and_center_crop(pil_image, 384)
                frames.append(pil_image)
        
        cap.release()
        return frames

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


def video_collate_fn(examples, processor):
    texts = []
    images_list = []
    
    for example in examples:
        frames = example['frames']
        question = example['question']
        answer = example['answer']
        
        # Create exactly as many image tokens as we have frames
        image_tokens = [{"type": "image"} for _ in range(len(frames))]
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
        # logger.info(f"Len image tokens {len(image_tokens)} messages\n\n {messages} text\n\n {text}")
        # n_images_in_text = text.count("<image>")
        # n_images_in_frames = len(frames)
        
        # logger.info(f"Number of frames: {n_images_in_frames}, Number of image tokens: {n_images_in_text}")
        
        # Each example's frames get converted to list of single-image lists
        example_images = [img for img in frames]
        
        texts.append(text.strip())
        images_list.append(example_images)  # Keep images grouped by example

    # print(f"Num examples: {len(texts)}, Images per example: {[len(x) for x in images_list]}")
    
    batch = processor(text=texts, images=images_list, return_tensors="pt", padding=True)

    # # Debug print for the first sequence in batch
    # print("\nFirst sequence tokens:")
    # input_ids = batch["input_ids"][0]
    # tokens = processor.tokenizer.convert_ids_to_tokens(input_ids)
    # print("\nToken ID -> Token mapping:")
    # for idx, (token_id, token) in enumerate(zip(input_ids.tolist(), tokens)):
    #     print(f"Position {idx}: ID {token_id} -> '{token}'")

    # # Print what we're masking
    # print("\nCurrently masking:")
    # print(f"Pad token ID: {processor.tokenizer.pad_token_id}")
    # print(f"Image token ID: {processor.tokenizer.additional_special_tokens_ids[processor.tokenizer.additional_special_tokens.index('<image>')]}")
    
    # print("\nAll special token IDs:")
    # special_tokens = {
    #     'pad_token': processor.tokenizer.pad_token_id,
    #     'eos_token': processor.tokenizer.eos_token_id,
    #     'bos_token': processor.tokenizer.bos_token_id,
    #     'unk_token': processor.tokenizer.unk_token_id,
    #     'additional_special_tokens': processor.tokenizer.additional_special_tokens_ids
    # }
    # print(special_tokens)



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

    batch["labels"] = labels
    
    # # Print what we're actually predicting
    # print("\nPredicting these tokens (non -100):")
    # non_masked = (labels[0] != -100).nonzero().squeeze()
    # predict_tokens = processor.tokenizer.convert_ids_to_tokens(batch["input_ids"][0][non_masked])
    # predict_ids = batch["input_ids"][0][non_masked].tolist()
    # print("\nToken ID -> Token mapping for predictions:")
    # for idx, (token_id, token) in enumerate(zip(predict_ids, predict_tokens)):
    #     print(f"Position {idx}: ID {token_id} -> '{token}'")

    return batch

def main():
    # Initialize distributed environment
    torch.distributed.init_process_group(backend='nccl')
    
    # Configuration
    USE_LORA = False
    USE_QLORA = False
    USE_WANDB = True
    model_id = "HuggingFaceTB/SmolVLM_converted_4"
    # data_path = "/fsx/miquel/LongVU/trainings/prep_material/train_video_data_simplevideo.json"
    max_frames = 50
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize wandb
    if USE_WANDB and is_main_process_multi_node():
        wandb.init(
            project="smolvlm-longvumix",
            name="v1-filter-lowlrhighwarmup_4",
            config={
                "model_id": model_id,
                "use_lora": USE_LORA,
                "use_qlora": USE_QLORA,
                "max_frames": max_frames,
                "device": DEVICE,
            }
        )


    # Initialize processor and model
    processor = AutoProcessor.from_pretrained(model_id)

    # Special config for video
    processor.image_processor.size = (384, 384)
    processor.image_processor.do_resize = False
    processor.image_processor.do_image_splitting = False

    if USE_QLORA or USE_LORA:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules=['down_proj','o_proj','k_proj','q_proj','gate_proj','up_proj','v_proj'],
            use_dora=False if USE_QLORA else True,
            init_lora_weights="gaussian"
        )
        lora_config.inference_mode = False
        
        if USE_QLORA:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            
        model = Idefics3ForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config if USE_QLORA else None,
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
    
    # Enable gradient checkpointing to reduce memory usage
    model.gradient_checkpointing_enable()

    # Initialize dataset
    dataset = VideoQADataset(processor, max_frames)
    
    # Split dataset
    train_dataset = VideoQADataset(processor, max_frames, split="train")
    train_sampler = DistributedSampler(train_dataset)
    
    # Training arguments
    num_nodes = int(16)
    training_args = TrainingArguments(
        num_train_epochs=1,
        per_device_train_batch_size=1,  # Reduced due to multiple frames
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16//num_nodes,  # Increased to compensate
        warmup_steps=200 * num_nodes, # increased from 30 to 200
        learning_rate=5e-7 * num_nodes, # reduced from 1e-4
        weight_decay=0.01,
        logging_steps=20,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=30,
        optim="adamw_torch" if not (USE_LORA or USE_QLORA) else "paged_adamw_8bit",
        bf16=True,
        output_dir="./smolvlm-longvumix-filter1-lowlrhighwarm_4",
        remove_unused_columns=False,
        report_to="wandb" if USE_WANDB else "none",
        logging_dir="./logs",
        logging_first_step=True,
        dataloader_drop_last=True,
    )

    # training_args = TrainingArguments(
    #     num_train_epochs=4,
    #     per_device_train_batch_size=1,  # Reduced due to multiple frames
    #     per_device_eval_batch_size=1,
    #     gradient_accumulation_steps=16//num_nodes,  # Increased to compensate
    #     warmup_steps=50 * num_nodes,
    #     learning_rate=1e-4 * num_nodes,
    #     weight_decay=0.01,
    #     logging_steps=20,
    #     save_strategy="steps",
    #     save_steps=1000,
    #     save_total_limit=30,
    #     optim="adamw_torch" if not (USE_LORA or USE_QLORA) else "paged_adamw_8bit",
    #     bf16=True,
    #     output_dir="./smolvlm-longvumix-filter1-long-round2",
    #     remove_unused_columns=False,
    #     report_to="wandb" if USE_WANDB else "none",
    #     logging_dir="./logs",
    #     logging_first_step=True,
    #     dataloader_drop_last=True,
    # )

    
    # Initialize trainer with custom collate function
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=lambda examples: video_collate_fn(examples, processor),
    )
    
    # Train and evaluate
    trainer.train()
    # trainer.evaluate()
    
    # Clean up
    dataset.executor.shutdown()
    if USE_WANDB:
        wandb.finish()

if __name__ == "__main__":
    main()