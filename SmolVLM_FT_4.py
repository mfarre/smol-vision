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
import torch.nn as nn


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
    def __init__(self, max_frames: int = 100, fps: float = 2.0):
        self.max_frames = max_frames
        self.fps = fps

    def resize_and_center_crop(self, image: Image.Image, target_size: int) -> Image.Image:
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

    def extract_frames_from_video(self, video_path: str) -> tuple[List[Image.Image], List[str]]:
        """Extract frames from video file using decord"""
        decord.bridge.set_bridge('torch')
        
        try:
            vr = VideoReader(video_path)
            total_frames = len(vr)
            fps_original = vr.get_avg_fps()
            duration = total_frames / fps_original
            
            # Calculate frame indices based on desired fps
            if self.fps > 0:
                # For video/gif, use fps-based sampling
                step = fps_original / self.fps  # frames to skip
                frame_indices = []
                timestamps = []
                current_frame = 0
                
                while current_frame < total_frames:
                    frame_indices.append(int(current_frame))
                    time_in_seconds = current_frame / fps_original
                    minutes = int(time_in_seconds // 60)
                    seconds = int(time_in_seconds % 60)
                    timestamps.append(f"{minutes:02d}:{seconds:02d}")
                    current_frame += step
                    
                    if len(frame_indices) >= self.max_frames:
                        print(f"MAXIMUM FRAMES {self.max_frames} reached for {video_path}")
                        break
            else:
                print(f"Failed to get FPS reading {video_path}")
            # Read frames
            frames = vr.get_batch(frame_indices)
            
            # Convert to PIL and process
            processed_frames = []
            for frame in frames:
                frame = frame.numpy()
                pil_image = Image.fromarray(frame)
                pil_image = self.resize_and_center_crop(pil_image, 384)
                processed_frames.append(pil_image)
                
            return processed_frames, timestamps
            
        except Exception as e:
            print(f"Error processing video {video_path}: {str(e)}")
            raise

    def extract_frames_from_gif(self, gif_path: str) -> tuple[List[Image.Image], List[str]]:
        """Extract frames from GIF file"""
        gif = Image.open(gif_path)
        frames = []
        durations = []
        total_duration = 0
        
        try:
            while True:
                frames.append(gif.copy())
                # Get frame duration in milliseconds (default to 100ms if not specified)
                duration = gif.info.get('duration', 100)
                durations.append(duration)
                total_duration += duration
                gif.seek(gif.tell() + 1)
        except EOFError:
            pass

        # Convert total_duration from ms to seconds
        total_duration_sec = total_duration / 1000
        
        if self.fps > 0:
            # Calculate how many frames we want based on fps
            target_frames = min(int(total_duration_sec * self.fps), self.max_frames)
            indices = np.linspace(0, len(frames) - 1, target_frames, dtype=int)
            
            # Calculate timestamps
            timestamps = []
            cumulative_time = 0
            for idx in indices:
                cumulative_time = sum(durations[:idx]) / 1000  # Convert to seconds
                minutes = int(cumulative_time // 60)
                seconds = int(cumulative_time % 60)
                timestamps.append(f"{minutes:02d}:{seconds:02d}")
            
            frames = [frames[i] for i in indices]
        else:
            #In that case we process it as a sequence
            print(f"Did not manage to get FPS for {gif_path}")            
            timestamps = [None] * len(frames)  

        return [self.resize_and_center_crop(frame.convert('RGB'), 384) for frame in frames], timestamps

    def extract_jpeg(self, jpeg_path: str) -> List[Image.Image]:
        """Load and return a single image"""
        jpeg = Image.open(jpeg_path)
        frames = []
        frames.append(jpeg.copy().convert('RGB'))
        # return [self.resize_and_center_crop(frame.convert('RGB'), 384) for frame in frames]
        return frames
    
    def extract_frames_from_folder(self, folder_path: str, content_type: str) -> List[Image.Image]:
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
            if content_type == 'video':
                image = self.resize_and_center_crop(image.convert('RGB'), 384)
            else:
                image = image.convert('RGB')
            frames.append(image)
            
        return frames

    def extract_frames(self, path: str, content_type: str) -> List[Image.Image]:
        """Extract frames from video file, GIF, or folder of images"""
        if os.path.isdir(path):
            frames = self.extract_frames_from_folder(path, content_type)
            return frames,[None] * len(frames)
        elif path.lower().endswith('.gif'):
            return self.extract_frames_from_gif(path)
        elif path.lower().endswith('.jpeg'):
            return self.extract_jpeg(path), [None]
        else:
            return self.extract_frames_from_video(path)

class VideoQADataset(Dataset):
    def __init__(self, processor, max_frames: int = 100, fps: float = 2.0, split="train"):
        self.processor = processor
        self.frame_extractor = VideoFrameExtractor(max_frames, fps)
        
        # Load HF dataset
        self.data = load_dataset("HuggingFaceFV/longvucauldron", split=split)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        content_type = self.data[idx].get('type', 'video')
        rank = torch.distributed.get_rank()

        if content_type != 'only_text':
            video_path = os.path.join('/fsx/miquel/longvu-dataset/composition', self.data[idx]['video'])
            logger.info(f"[Rank {rank}] Loading video: {video_path}")
            frames, timestamps = self.frame_extractor.extract_frames(video_path,content_type)
            if len(frames) == 0:
                logger.info(f"Zero frames for {video_path}")
        else:
            logger.info(f"[Rank {rank}] Loading text Q&A")
            frames = []
            timestamps = []

        conversations = self.data[idx]['conversations']
        question = next(conv for conv in conversations if conv['from'] == 'human')['value'].split('<image>\n')[0]
        answer = next(conv for conv in conversations if conv['from'] == 'gpt')['value']
            
        return {
            'frames': frames,
            'timestamps': timestamps,
            'question': question,
            'answer': answer,
            'type': content_type
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

def find_row_col_patterns(tokens):
    # Find positions where "<row_X_col_Y>" pattern appears, where X,Y are 1-9
    mask_positions = []
    for i in range(len(tokens) - 8):  # Need 9 tokens for the pattern
        if (tokens[i] == '<' and 
            tokens[i+1] == 'row' and
            tokens[i+2] == '_' and
            tokens[i+3].isdigit() and int(tokens[i+3]) in range(1, 10) and
            tokens[i+4] == '_' and
            tokens[i+5] == 'col' and
            tokens[i+6] == '_' and
            tokens[i+7].isdigit() and int(tokens[i+7]) in range(1, 10) and
            tokens[i+8] == '>'):
            # print(f"\nFound row-col pattern at position {i}:")
            # print(f"Complete sequence: {' '.join(tokens[i:i+9])}")
            mask_positions.extend([i, i+1, i+2, i+3, i+4, i+5, i+6, i+7, i+8])
    return mask_positions


def video_collate_fn(examples, processor, max_frames, use_temporal_tokens=True):
    processed_batches = []
    
    # Process each example individually
    for example in examples:
        frames = example['frames']
        timestamps = example.get('timestamps', [None] * len(frames))  # Get timestamps if available
        question = example['question']
        answer = example['answer']
        content_type = example.get('type', 'video')
        
        # Apply appropriate processor settings based on content type
        if content_type == 'video':
            processor.image_processor.size = (384, 384)
            processor.image_processor.do_resize = False
            processor.image_processor.do_image_splitting = False
        else:
            processor.image_processor.size = {'longest_edge': 1920}
            processor.image_processor.do_resize = True
            processor.image_processor.do_image_splitting = True
        
        # Add temporal tokens between frames
        image_tokens = []
        for i in range(len(frames)):
            if timestamps[i] is not None:
                next_timestamp = timestamps[i + 1] if i < len(frames) - 1 else f"{timestamps[i]}+1s" #HACK
                image_tokens.append({"type": "text", "text": f"clip from {timestamps[i]}-{next_timestamp}:"})            
            image_tokens.append({"type": "image"})

        
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

        # # Debug print the tokenized output
        # print(f"\n=== Example  ===")
        # print(f"Content type: {content_type}")
        # print(f"Number of frames: {len(frames)}")
        # if len(timestamps) > 0:
        #     print(f"Timestamps available: {timestamps[0] is not None}")
        # print("\nTokenized text:")
        # tokens = processor.tokenizer.tokenize(text)
        # for idx, token in enumerate(tokens):
        #     print(f"{idx}: {token}")

        # Process single example
        single_batch = processor(
            text=text.strip(),
            images=[img for img in frames] if frames else None,
            return_tensors="pt",
            padding=False  # Don't pad individual examples yet
        )
        
        # Handle labels for this example
        image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<image>")]
        fake_token_around_image_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<fake_token_around_image>")]
        end_of_utterance_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<end_of_utterance>")]

        labels = single_batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        labels[labels == image_token_id] = -100
        labels[labels == fake_token_around_image_id] = -100
        labels[labels == end_of_utterance_id] = -100

        if use_temporal_tokens:
            temporal_token_ids = [processor.tokenizer.convert_tokens_to_ids(f"<frame_{i}>") 
                                for i in range(max_frames)]
            for token_id in temporal_token_ids:
                labels[labels == token_id] = -100

        tokens = processor.tokenizer.convert_ids_to_tokens(single_batch["input_ids"][0])
        positions_to_mask = find_global_img_patterns(tokens) + find_row_col_patterns(tokens)
        for pos in positions_to_mask:
            labels[0, pos] = -100

        single_batch["labels"] = labels
        processed_batches.append(single_batch)
    
    # Find maximum length
    max_length = max(batch["input_ids"].size(1) for batch in processed_batches)
    
    # Pad each batch to max_length
    for i in range(len(processed_batches)):
        cur_len = processed_batches[i]["input_ids"].size(1)
        if cur_len < max_length:
            # Pad input_ids
            padding = torch.full(
                (1, max_length - cur_len),
                processor.tokenizer.pad_token_id,
                device=processed_batches[i]["input_ids"].device,
                dtype=processed_batches[i]["input_ids"].dtype
            )
            processed_batches[i]["input_ids"] = torch.cat([processed_batches[i]["input_ids"], padding], dim=1)
            
            # Pad attention_mask
            attention_padding = torch.zeros(
                (1, max_length - cur_len),
                device=processed_batches[i]["attention_mask"].device,
                dtype=processed_batches[i]["attention_mask"].dtype
            )
            processed_batches[i]["attention_mask"] = torch.cat([processed_batches[i]["attention_mask"], attention_padding], dim=1)
            
            # Pad labels
            labels_padding = torch.full(
                (1, max_length - cur_len),
                -100,
                device=processed_batches[i]["labels"].device,
                dtype=processed_batches[i]["labels"].dtype
            )
            processed_batches[i]["labels"] = torch.cat([processed_batches[i]["labels"], labels_padding], dim=1)
    
    # Combine all processed batches
    combined_batch = {
        "input_ids": torch.cat([batch["input_ids"] for batch in processed_batches], dim=0),
        "attention_mask": torch.cat([batch["attention_mask"] for batch in processed_batches], dim=0),
        "labels": torch.cat([batch["labels"] for batch in processed_batches], dim=0),
    }
    
    # Add pixel_values if any example has images
    if any("pixel_values" in batch for batch in processed_batches):
        # Get the shape from the first batch with pixel_values
        first_image_batch = next(batch for batch in processed_batches if "pixel_values" in batch)
        pixel_shape = first_image_batch["pixel_values"].shape[2:]  # Shape after batch and frame dimensions
        
        # Find max number of frames across all batches
        max_frames_in_batch = max(
            batch["pixel_values"].size(1) if "pixel_values" in batch else 0 
            for batch in processed_batches
        )
        
        # Create padded pixel values tensor
        pixel_values = []
        for batch in processed_batches:
            if "pixel_values" in batch:
                current_frames = batch["pixel_values"].size(1)
                if current_frames < max_frames_in_batch:
                    # Pad with zeros up to max_frames
                    padding = torch.zeros(
                        (1, max_frames_in_batch - current_frames, *pixel_shape),
                        device=batch["pixel_values"].device,
                        dtype=batch["pixel_values"].dtype
                    )
                    pixel_values.append(torch.cat([batch["pixel_values"], padding], dim=1))
                else:
                    pixel_values.append(batch["pixel_values"])
            else:
                # Add zero tensor with correct shape for text-only examples
                pixel_values.append(
                    torch.zeros((1, max_frames_in_batch, *pixel_shape), 
                              device=batch["input_ids"].device,
                              dtype=torch.float32)
                )
        
        combined_batch["pixel_values"] = torch.cat(pixel_values, dim=0)
    
    return combined_batch


class CustomTrainer(Trainer):
    def _wrap_model(self, model, training=True, dataloader=None):
        print(f"\n=== Wrapping Model ===")
        print(f"Training mode: {training}")
        print(f"Local rank: {self.args.local_rank}")
        print(f"CUDA Memory before DDP: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
        
        if not training:
            return model

        if self.args.local_rank != -1:
            try:
                print("Attempting DDP wrap...")
                kwargs = {
                    "device_ids": [self.args.local_rank] if self.args.local_rank != -1 else None,
                    "output_device": self.args.local_rank if self.args.local_rank != -1 else None,
                    "find_unused_parameters": True,
                    "static_graph": True,
                    "gradient_as_bucket_view": True
                }
                model = nn.parallel.DistributedDataParallel(model, **kwargs)
                print("DDP wrap successful")
                print(f"CUDA Memory after DDP: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
            except Exception as e:
                print(f"Error during DDP wrap: {str(e)}")
                raise

        return model

    def training_step(self, model, inputs, num_items_in_batch=None):
        print(f"\n=== Training Step ===")
        print(f"Batch size: {inputs['input_ids'].shape[0]}")
        print(f"CUDA Memory before forward: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
        
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        try:
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
            print(f"Forward pass successful, loss: {loss.item()}")
            print(f"CUDA Memory after forward: {torch.cuda.memory_allocated()/1024**2:.2f}MB")

            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps

            print("Starting backward pass...")
            try:
                # Remove the **kwargs here
                self.accelerator.backward(loss)  
                print("Backward pass successful")
            except Exception as e:
                print(f"Error during backward pass: {str(e)}")
                print(f"Loss device: {loss.device}")
                print(f"Model parameters devices: {[p.device for p in model.parameters()][:5]}")
                raise
            print(f"CUDA Memory after backward: {torch.cuda.memory_allocated()/1024**2:.2f}MB")

        except Exception as e:
            print(f"Error during training step: {str(e)}")
            shapes = {k: v.shape if hasattr(v, 'shape') else type(v) for k, v in inputs.items()}
            print(f"Input shapes: {shapes}")
            raise

        return loss.detach() / self.args.gradient_accumulation_steps

def main():
    parser = argparse.ArgumentParser(description='Video-LLM Training')
    parser.add_argument('--max_frames', type=int, default=100, help='Maximum number of frames per video')
    parser.add_argument('--fps', type=float, default=2.0, help='Frames per second to extract from videos/GIFs')
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
    # output_dir = f"./smolvlm_frames{args.max_frames}_{temporal_str}_lr_1e-5"
    output_dir = f"./smolvlm_longvucauldron_FPS_fps{int(args.fps)}_frames{args.max_frames}_{temporal_str}_lr_3e-7"
    
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

    if args.temporal_tokens:
        # Resize token embeddings to account for new tokens
        model.resize_token_embeddings(len(processor.tokenizer))

    # Enable gradient checkpointing to reduce memory usage
    model.config.use_cache = False
    model.config.use_reentrant_checkpointing = False
    # model.gradient_checkpointing_enable()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # Initialize dataset
    train_dataset = VideoQADataset(processor, args.max_frames, args.fps, split="train")
    train_sampler = DistributedSampler(train_dataset)
    


    # Training arguments
    num_nodes = int(16)
    training_args = TrainingArguments(
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=32//num_nodes,
        warmup_ratio = 0.15,
        max_grad_norm = 2.0,
        learning_rate=3e-7 * num_nodes, #prev: 5e-6
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
        ddp_find_unused_parameters=True
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