#!/usr/bin/env python

import os
import logging
from typing import List
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import argparse
from datasets import load_dataset
from transformers import AutoProcessor, Idefics3ForConditionalGeneration
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin
import deepspeed
from torch.utils.data.distributed import DistributedSampler
import wandb
import decord
from decord import VideoReader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

    def extract_frames_from_video(self, video_path: str) -> List[Image.Image]:
        decord.bridge.set_bridge('torch')
        try:
            vr = VideoReader(video_path)
            total_frames = len(vr)
            if total_frames <= self.max_frames:
                frame_indices = list(range(total_frames))
            else:
                frame_indices = np.linspace(0, total_frames - 1, self.max_frames, dtype=int).tolist()
            frames = vr.get_batch(frame_indices)
            
            processed_frames = []
            for frame in frames:
                frame = frame.numpy()
                pil_image = Image.fromarray(frame)
                pil_image = self.resize_and_center_crop(pil_image, 364)
                processed_frames.append(pil_image)
                
            return processed_frames
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {str(e)}")
            raise

    def extract_frames_from_gif(self, gif_path: str) -> List[Image.Image]:
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
        return [self.resize_and_center_crop(frame.convert('RGB'), 364) for frame in frames]

    def extract_frames_from_folder(self, folder_path: str) -> List[Image.Image]:
        image_extensions = ('.jpg', '.jpeg', '.png')
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]
        
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
            
        numbered_files.sort(key=lambda x: (x[1], x[2]))
        sorted_files = [f[0] for f in numbered_files]
        if len(sorted_files) > self.max_frames:
            indices = np.linspace(0, len(sorted_files) - 1, self.max_frames, dtype=int)
            sorted_files = [sorted_files[i] for i in indices]
        
        frames = []
        for file in sorted_files:
            image_path = os.path.join(folder_path, file)
            image = Image.open(image_path)
            image = self.resize_and_center_crop(image.convert('RGB'), 364)
            frames.append(image)
        return frames

    def extract_frames(self, path: str) -> List[Image.Image]:
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
        self.data = load_dataset("HuggingFaceFV/longvumix", split=split)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path = os.path.join('/fsx/miquel/longvu-dataset/composition', self.data[idx]['video'])
        frames = self.frame_extractor.extract_frames(video_path)
        conversations = self.data[idx]['conversations']
        question = next(conv for conv in conversations if conv['from'] == 'human')['value'].split('<image>\n')[0]
        answer = next(conv for conv in conversations if conv['from'] == 'gpt')['value']
        return {
            'frames': frames,
            'question': question,
            'answer': answer
        }

def find_global_img_patterns(tokens):
    mask_positions = []
    for i in range(len(tokens) - 4):
        if (tokens[i] == '<' and tokens[i+1] == 'global' and 
            tokens[i+2] == '-' and tokens[i+3] == 'img' and 
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
        
        image_tokens = []
        for i in range(len(frames)):
            image_tokens.append({"type": "image"})
            if i < len(frames) - 1 and use_temporal_tokens:
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
        texts.append(text.strip())
        images_list.append(frames)

    batch = processor(text=texts, images=images_list, return_tensors="pt", padding=True)

    # Handle labels
    special_tokens = {
        'image': processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<image>")],
        'fake': processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<fake_token_around_image>")],
        'eou': processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<end_of_utterance>")]
    }

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    for token_id in special_tokens.values():
        labels[labels == token_id] = -100

    # Also mask out any temporal tokens, if used
    if use_temporal_tokens:
        temporal_token_ids = [
            processor.tokenizer.convert_tokens_to_ids(f"<frame_{i}>") for i in range(max_frames)
        ]
        for token_id in temporal_token_ids:
            labels[labels == token_id] = -100

    # Mask out <global-img> patterns in labels
    for i in range(len(batch["input_ids"])):
        tokens = processor.tokenizer.convert_ids_to_tokens(batch["input_ids"][i])
        positions_to_mask = find_global_img_patterns(tokens)
        for pos in positions_to_mask:
            labels[i, pos] = -100

    batch["labels"] = labels
    return batch


def main():
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    node_rank = int(os.environ.get('SLURM_NODEID', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    
    if local_rank == 0:
        print(f"Initializing process group with: MASTER_ADDR={os.environ['MASTER_ADDR']}, MASTER_PORT={os.environ['MASTER_PORT']}")
        print(f"Starting initialization: node_rank={node_rank}, world_size={world_size}")

    GRAD_ACCUMULATION_STEPS = 16
    TRAIN_MICRO_BATCH_SIZE = 1
    GRADIENT_CLIPPING = 2.0

    parser = argparse.ArgumentParser(description='Video-LLM Training with DeepSpeed')
    parser.add_argument('--max_frames', type=int, default=8)
    parser.add_argument('--temporal_tokens', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--local_rank', type=int, default=-1)
    args = parser.parse_args()

    torch.cuda.set_device(local_rank)

    # Get the number of processes from the environment
    num_processes = int(os.environ.get('GPUS_PER_NODE', '1'))
    print(f"Num processes {num_processes}")
    print(f"Train batch size {TRAIN_MICRO_BATCH_SIZE}x{GRAD_ACCUMULATION_STEPS}x{num_processes}")
    # Create DeepSpeed config
    ds_config = {
        # "train_batch_size": TRAIN_MICRO_BATCH_SIZE * GRAD_ACCUMULATION_STEPS * num_processes,
        "train_micro_batch_size_per_gpu": TRAIN_MICRO_BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUMULATION_STEPS,
        "gradient_clipping": GRADIENT_CLIPPING,
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu"
            },
            "offload_param": {
                "device": "cpu"
            },
            "stage3_gather_fp16_weights_on_model_save": True
        },
        "fp16": {
            "enabled": True
        }
    }

    # Create DeepSpeed plugin
    ds_plugin = DeepSpeedPlugin(
        hf_ds_config=ds_config,
        gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
        gradient_clipping=GRADIENT_CLIPPING,
        zero_stage=3,
        offload_optimizer_device="cpu",
        offload_param_device="cpu",
        zero3_init_flag=True,
        zero3_save_16bit_model=True,
    )

    # Initialize Accelerator once with all configurations
    accelerator = Accelerator(
        gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
        mixed_precision="fp16",
        deepspeed_plugin=ds_plugin
    )

    model_id = "HuggingFaceM4/Idefics3-8B-Llama3"
    temporal_str = "with_temp" if args.temporal_tokens else "no_temp"
    output_dir = f"./idefics3_frames{args.max_frames}_{temporal_str}_lr_5e-7"

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        if args.wandb:
            wandb.init(
                project="smolvlm-longvumix",
                name=os.path.basename(output_dir),
                config=vars(args)
            )

    # Initialize processor
    processor = AutoProcessor.from_pretrained(model_id)
    processor.image_processor.size = (364, 364)
    processor.image_processor.do_resize = False
    processor.image_processor.do_image_splitting = False

    # Add temporal tokens if requested
    if args.temporal_tokens:
        new_tokens = [f"<frame_{i}>" for i in range(args.max_frames)]
        all_tokens = processor.tokenizer.additional_special_tokens + new_tokens
        processor.tokenizer.add_special_tokens({"additional_special_tokens": all_tokens})

    if local_rank == 0:
        print(f"[Node {node_rank}] Created Accelerator")
        print(f"[Node {node_rank}] Starting model creation")

    # Log before and after model creation
    if local_rank == 0:
        print(f"[Node {node_rank}] About to create model")
    

    # -------------------------------------
    # Option 1: Use deepspeed.zero.Init()
    # -------------------------------------
    # This avoids the "meta tensor" conflict by sharding weights at creation time.
    with deepspeed.zero.Init(config_dict_or_path=ds_plugin.deepspeed_config):
        # If your model's from_pretrained doesn't accept torch_dtype, remove it here:
        model = Idefics3ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,  # If supported, else remove
            device_map=None,            # Let DeepSpeed ZeRO handle param partitioning
            trust_remote_code=True
        )

    # If we have newly added tokens, do this AFTER from_pretrained but BEFORE accelerator.prepare
    if args.temporal_tokens:
        model.resize_token_embeddings(len(processor.tokenizer))

    if local_rank == 0:
        print(f"[Node {node_rank}] Model created")
        

    # Prepare dataset & dataloader
    train_dataset = VideoQADataset(processor, args.max_frames, split="train")
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=True
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TRAIN_MICRO_BATCH_SIZE,
        sampler=train_sampler,
        collate_fn=lambda examples: video_collate_fn(
            examples, processor, args.max_frames, use_temporal_tokens=args.temporal_tokens
        ),
        num_workers=1,
        pin_memory=True
    )

    # Create optimizer & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-7,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    num_training_steps = len(train_dataloader) * 1  # single epoch
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_training_steps,
        eta_min=1e-8
    )

    # Prepare with accelerator
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # Training loop
    model.train()
    completed_steps = 0
    for epoch in range(1):
        train_sampler.set_epoch(epoch)
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            if (step + 1) % GRAD_ACCUMULATION_STEPS == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            completed_steps += 1
            if accelerator.is_main_process and step % 20 == 0:
                logger.info(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item():.4f}")
                if args.wandb:
                    wandb.log({
                        "train_loss": loss.item(),
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "epoch": epoch,
                        "step": completed_steps,
                    })

            if completed_steps % 250 == 0:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                accelerator.save(
                    {
                        "model": unwrapped_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "step": completed_steps,
                    },
                    f"{output_dir}/checkpoint-{completed_steps}"
                )

    # Final save
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        accelerator.save(
            {
                "model": unwrapped_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "step": completed_steps,
            },
            f"{output_dir}/final-checkpoint"
        )
        if args.wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
