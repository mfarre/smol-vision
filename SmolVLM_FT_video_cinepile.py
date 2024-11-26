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
from tqdm.auto import tqdm
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

# GPU config remains the same
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


VID_DIR = "/fsx/miquel/hf-cinepile-collab2/hf-cinepile-collab/fix_cinepile/yt_videos/"

class VideoFrameExtractor:
    def __init__(self, max_frames: int = 50):
        self.max_frames = max_frames
        
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
        
    def extract_frames(self, video_path: str, fps: float = 1.0) -> List[Image.Image]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        frame_step = int(video_fps / fps)
        frame_indices = list(range(0, total_frames, frame_step))
        
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

class CinepileDataset(Dataset):
    def __init__(self, processor, max_frames=50, split="train", max_length=512):
        self.dataset = load_dataset("tomg-group-umd/cinepile", split=split)
        self.processor = processor
        self.frame_extractor = VideoFrameExtractor(max_frames=max_frames)
        self.max_length = max_length
        
        # Pre-filter valid indices during initialization
        self.valid_indices = []
        print("Filtering valid videos...")
        for idx in tqdm(range(len(self.dataset))):
            example = self.dataset[idx]
            yt_link = example['yt_clip_link']
            vid_file_name = f"{example['movie_name']}_{yt_link.split('/')[-1]}"
            video_path = f"{VID_DIR}/{vid_file_name}.mp4"
            
            if os.path.exists(video_path):
                self.valid_indices.append(idx)
        
        print(f"Found {len(self.valid_indices)} valid videos out of {len(self.dataset)} total")
        if len(self.valid_indices) == 0:
            raise RuntimeError("No valid videos found in dataset")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        try:
            # Get the actual dataset index from our filtered list
            dataset_idx = self.valid_indices[idx]
            example = self.dataset[dataset_idx]
            
            yt_link = example['yt_clip_link']
            vid_file_name = f"{example['movie_name']}_{yt_link.split('/')[-1]}"
            video_path = f"{VID_DIR}/{vid_file_name}.mp4"
            
            # Extract frames
            frames = self.frame_extractor.extract_frames(video_path)
            
            # Format question and choices
            choice = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}[example['answer_key_position']]
            question = f"{example['question']}\n"
            for i, opt in enumerate(example['choices']):
                question += f"- {chr(65+i)}) {opt}\n"

            # Create messages format
            image_tokens = [{"type": "image"} for _ in range(len(frames))]
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Answer briefly with just the letter of your choice."},
                        *image_tokens,
                        {"type": "text", "text": f"Given these subtitles:\n{example['subtitles']}\n\n{question}"}
                    ]
                }
            ]

            # Process inputs
            model_inputs = self.processor(
                text=self.processor.apply_chat_template(messages, add_generation_prompt=True),
                images=frames,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            )

            # Create labels by encoding the choice letter
            labels = self.processor(
                text=choice,
                return_tensors="pt",
                padding=True,
                max_length=2,  # Just enough for the single letter answer
                truncation=True,
            ).input_ids

            # Remove batch dimension
            for k in model_inputs:
                if isinstance(model_inputs[k], torch.Tensor):
                    model_inputs[k] = model_inputs[k].squeeze(0)

            model_inputs["labels"] = labels.squeeze(0)
            
            return model_inputs
            
        except Exception as e:
            print(f"Error loading item at index {idx} (dataset index {dataset_idx}): {str(e)}")
            # If we hit an error, try the next valid index
            if idx + 1 < len(self.valid_indices):
                return self.__getitem__(idx + 1)
            else:
                return self.__getitem__(0)  # Wrap around to start if at end

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
        subtitles = example['subtitles']
        question = example['question']
        answer = example['answer']
        
        image_tokens = [{"type": "image"} for _ in range(len(frames))]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Given these video frames and subtitles:\n" + subtitles},
                    *image_tokens,
                    {"type": "text", "text": "\n" + question}
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
        images_list.append(example_images)

    batch = processor(text=texts, images=images_list, return_tensors="pt", padding=True)

    labels = batch["input_ids"].clone()
    image_token_id = processor.tokenizer.additional_special_tokens_ids[
        processor.tokenizer.additional_special_tokens.index("<image>")]
    fake_token_around_image_id = processor.tokenizer.additional_special_tokens_ids[
        processor.tokenizer.additional_special_tokens.index("<fake_token_around_image>")]
    end_of_utterance_id = processor.tokenizer.additional_special_tokens_ids[
        processor.tokenizer.additional_special_tokens.index("<end_of_utterance>")]

    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == image_token_id] = -100
    labels[labels == fake_token_around_image_id] = -100
    labels[labels == end_of_utterance_id] = -100

    batch["labels"] = labels
    return batch

# Main function remains largely the same, with dataset changes
def main():
    torch.distributed.init_process_group(backend='nccl')
    
    USE_LORA = False
    USE_QLORA = False
    USE_WANDB = True
    model_id = "HuggingFaceTB/SmolVLM_converted_4"
    max_frames = 50
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    if USE_WANDB and is_main_process_multi_node():
        wandb.init(
            project="smolvlm-video-qa",
            name="smolvideolm-cinepile-50frames",
            config={
                "model_id": model_id,
                "use_lora": USE_LORA,
                "use_qlora": USE_QLORA,
                "max_frames": max_frames,
                "device": DEVICE,
            }
        )

    processor = AutoProcessor.from_pretrained(model_id)
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
    
    model.gradient_checkpointing_enable()

    # Initialize dataset with new CinepileDataset
    dataset = CinepileDataset(processor, max_frames)
    
    train_dataset = CinepileDataset(
        processor=processor,
        max_frames=50,
        split="train"
    )

    # For validation data
    eval_dataset = CinepileDataset(
        processor=processor,
        max_frames=50,
        split="test"
    )
    train_sampler = DistributedSampler(train_dataset)
    eval_sampler = DistributedSampler(eval_dataset)
    
    # Training arguments remain the same
    num_nodes = int(16)
    training_args = TrainingArguments(
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16//num_nodes,
        warmup_steps=50 * num_nodes,
        learning_rate=1e-4 * num_nodes,
        weight_decay=0.01,
        logging_steps=1,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit = 10,
        optim="adamw_torch" if not (USE_LORA or USE_QLORA) else "paged_adamw_8bit",
        bf16=True,
        output_dir="./smolvlm-video-qa-cinepile",
        remove_unused_columns=False,
        report_to="wandb" if USE_WANDB else "none",
        logging_dir="./logs",
        logging_first_step=True,
        dataloader_drop_last=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=lambda examples: video_collate_fn(examples, processor),
    )
    
    trainer.train()
    trainer.evaluate()
    dataset.executor.shutdown()    
    if USE_WANDB:
        wandb.finish()

if __name__ == "__main__":
    main()