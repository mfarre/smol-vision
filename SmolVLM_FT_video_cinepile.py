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
        
    def extract_frames(self, video_path: str) -> List[Image.Image]:
        """Extract frames from video with error handling."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.warning(f"Could not open video: {video_path}")
                return []
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            if total_frames == 0 or fps == 0:
                logger.error(f"Invalid video properties for {video_path}: frames={total_frames}, fps={fps}")
                return []
            
            # Calculate frame indices to extract (1fps)
            frame_indices = list(range(0, total_frames, fps))
            
            # If we have more frames than max_frames, sample evenly
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
            
            if not frames:
                logger.error(f"No frames extracted from {video_path}")
                
            return frames
            
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {str(e)}")
            return []

class VideoQADataset(Dataset):
    def __init__(self, data_path: str, processor, max_frames: int = 50):
        self.processor = processor
        self.frame_extractor = VideoFrameExtractor(max_frames)
        
        logger.info(f"Loading data from {data_path}")
        with open(data_path, 'r') as f:
            raw_data = json.load(f)
            
        # Process the raw data into the required format
        self.data = []
        for item in raw_data:
            processed_item = self._process_conversation(item)
            if processed_item:
                self.data.append(processed_item)
                
        random.seed(42)
        random.shuffle(self.data)
        logger.info(f"Loaded {len(self.data)} examples")

    def _process_conversation(self, item):
        """Process a single conversation item from the new format to the required format."""
        try:
            messages = item['messages']
            
            # Extract system message
            system_message = next((msg for msg in messages if msg['role'] == 'system'), None)
            
            # Find the user message 
            user_message = next((msg for msg in messages if msg['role'] == 'user'), None)
            if not user_message:
                return None
                
            # Get content from user message
            user_content = user_message['content']
            
            # Extract video path from video content
            video_info = next((content for content in user_content if content['type'] == 'video'), None)
            if not video_info:
                return None
                
            # Extract question from text content
            text_info = next((content for content in user_content if content['type'] == 'text'), None)
            if not text_info:
                return None
                
            # Extract answer from assistant message
            assistant_message = next((msg for msg in messages if msg['role'] == 'assistant'), None)
            if not assistant_message or not assistant_message['content']:
                return None
                
            return {
                'video': video_info['video'],
                'question': text_info['text'],
                'answer': assistant_message['content'][0]['text']
            }
        except Exception as e:
            logger.error(f"Error processing item: {str(e)}")
            return None
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        frames = self.frame_extractor.extract_frames(item['video'])
        
        # Skip examples with no frames
        if not frames:
            logger.warning(f"Skipping example {idx} due to no frames")
            # Return the next valid example
            return self.__getitem__((idx + 1) % len(self))
            
        return {
            'frames': frames,
            'question': item['question'],
            'answer': item['answer']
        }

def video_collate_fn(examples, processor):
    texts = []
    images_list = []
    
    for example in examples:
        frames = example['frames']
        
        # Skip if no frames (shouldn't happen due to __getitem__ handling)
        if not frames:
            continue
            
        question = example['question']
        answer = example['answer']
        
        # Create image tokens for each frame
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
        # logger.info(messages)
        example_images = [img for img in frames]
        
        texts.append(text.strip())
        images_list.append(example_images)
        
    # Make sure we have at least one example
    if not texts or not images_list:
        logger.error("No valid examples in batch")
        raise ValueError("No valid examples in batch")

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

    batch["labels"] = labels
    return batch

def is_main_process():
    return os.environ.get('LOCAL_RANK', '0') == '0'

def is_main_process_multi_node():
    # Check if this is the main process across all nodes
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    global_rank = int(os.environ.get('RANK', '0'))
    return local_rank == 0 and global_rank == 0


def main():
    # Initialize distributed environment
    torch.distributed.init_process_group(backend='nccl')
    
    # Configuration
    USE_LORA = False
    USE_QLORA = False
    USE_WANDB = True
    model_id = "HuggingFaceTB/SmolVLM_converted_4"
    data_path = "/fsx/miquel/simplevideo_trl/simplevideo_trl/lastExplorationCinePile/temp.json"
    max_frames = 50
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize wandb
    if USE_WANDB and is_main_process_multi_node():
        wandb.init(
            project="smolvlm-video-qa",
            name="smolvideolm-50frames-cinepile",
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
    dataset = VideoQADataset(data_path, processor, max_frames)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [train_size, len(dataset) - train_size]
    )

    train_sampler = DistributedSampler(train_dataset)
    eval_sampler = DistributedSampler(eval_dataset)
    
    # Training arguments
    num_nodes = int(16)
    training_args = TrainingArguments(
        num_train_epochs=1,
        per_device_train_batch_size=1,  # Reduced due to multiple frames
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16//num_nodes,  # Increased to compensate
        warmup_steps=50 * num_nodes,
        learning_rate=1e-4 * num_nodes,
        weight_decay=0.01,
        logging_steps=1,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=10,
        optim="adamw_torch" if not (USE_LORA or USE_QLORA) else "paged_adamw_8bit",
        bf16=True,
        output_dir="./smolvlm-video-qa",
        remove_unused_columns=False,
        report_to="wandb" if USE_WANDB else "none",
        logging_dir="./logs",
        logging_first_step=True,
        dataloader_drop_last=True,
    )

    
    # Initialize trainer with custom collate function
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=lambda examples: video_collate_fn(examples, processor),
    )
    
    # Train and evaluate
    trainer.train()
    trainer.evaluate()
    
    # Clean up
    dataset.executor.shutdown()
    if USE_WANDB:
        wandb.finish()

if __name__ == "__main__":
    main()