import torch
from transformers import AutoProcessor, Idefics3ForConditionalGeneration
from PIL import Image
import cv2
import numpy as np
from typing import List
import logging
from transformers import AutoTokenizer
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: str, base_model_id: str = "HuggingFaceTB/SmolVLM-Instruct", device: str = "cuda"):
    # Load processor from original model
    processor = AutoProcessor.from_pretrained(base_model_id)
    if checkpoint_path:
        # Load fine-tuned model from checkpoint
        model = Idefics3ForConditionalGeneration.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
    else:
        model = Idefics3ForConditionalGeneration.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map=device
        )    

    # Configure processor for video frames
    processor.image_processor.size = (384, 384)
    processor.image_processor.do_resize = False
    processor.image_processor.do_image_splitting = False
    
    return model, processor

class VideoFrameExtractor:
    def __init__(self, max_frames: int = 100, fps: float = 2.0):
        self.max_frames = max_frames
        self.fps = fps

        
    def resize_and_center_crop(self, image: Image.Image, target_size: int) -> Image.Image:
        # Get current dimensions
        width, height = image.size
        
        # Calculate new dimensions keeping aspect ratio
        if width < height:
            new_width = target_size
            new_height = int(height * (target_size / width))
        else:
            new_height = target_size
            new_width = int(width * (target_size / height))
            
        # Resize
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Center crop
        left = (new_width - target_size) // 2
        top = (new_height - target_size) // 2
        right = left + target_size
        bottom = top + target_size
        
        return image.crop((left, top, right, bottom))
        
                
    def extract_frames(self, video_path: str) -> tuple[List[Image.Image], List[str]]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_original = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Calculate frame indices based on desired fps
        if self.fps > 0:
            # For video, use fps-based sampling
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
                    print(f"MAXIMUM FRAMES {self.max_frames} reached")
                    break
        
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
        return frames, timestamps

def generate_response(model, processor, video_path: str, question: str, max_frames: int = 100, fps: float = 2.0):
    # Extract frames
    frame_extractor = VideoFrameExtractor(max_frames, fps)
    frames, timestamps = frame_extractor.extract_frames(video_path)
    logger.info(f"Extracted {len(frames)} frames from video at {fps} FPS")
    
    # Create prompt with frames and timestamps
    image_tokens = []
    for i in range(len(frames)):
        if timestamps[i] is not None:
            next_timestamp = timestamps[i + 1] if i < len(frames) - 1 else f"{timestamps[i]}+1s"
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
        }
    ]

    # Process inputs
    inputs = processor(
        text=processor.apply_chat_template(messages, add_generation_prompt=True),
        images=[img for img in frames],
        return_tensors="pt"
    ).to(model.device)

    # Debug: print context length
    seq_length = inputs["input_ids"].size(1)
    print(f"\n=== Context Length Usage ===")
    print(f"Sequence length: {seq_length}")
    print(f"Model max length: {processor.tokenizer.model_max_length}")
    print(f"Context utilization: {(seq_length / processor.tokenizer.model_max_length) * 100:.2f}%")

    # Generate response
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        num_beams=5,
        temperature=0.7,
        do_sample=True,
        use_cache=True
    )
    
    # Decode response
    response = processor.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    # Configuration
    checkpoint_path = "/fsx/miquel/smol-vision/smolvlm_longvucauldron_FPS_fps2_frames100_no_temp_lr_5e-7/checkpoint-1000"
    base_model_id = "HuggingFaceTB/SmolVLM-Instruct"  
    video_path = "/fsx/miquel/fineVideo2Idefics/a/videos/--Dq6kFSRDE_scene_1.mp4"
    question = "can you explain step by step what is happening in this video?"
    max_frames = 150
    fps = 2.0  # Added FPS parameter

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    logger.info("Loading model...")
    model, processor = load_model(checkpoint_path, base_model_id, device)

    # Generate response
    logger.info("Generating response...")
    response = generate_response(model, processor, video_path, question, max_frames=max_frames, fps=fps)
    
    # Print results
    print("Question:", question)
    print("Response:", response)

if __name__ == "__main__":
    main()
