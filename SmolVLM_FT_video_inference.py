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


# Alternative video reader
# import decord
# from decord import VideoReader
# class VideoFrameExtractor:
#     def __init__(self, max_frames: int = 50):
#         self.max_frames = max_frames
        
#     def resize_and_center_crop(self, image: Image.Image, target_size: int) -> Image.Image:
#         """Resize the image preserving aspect ratio and then center crop."""
#         width, height = image.size
        
#         if width < height:
#             new_width = target_size
#             new_height = int(height * (target_size / width))
#         else:
#             new_height = target_size
#             new_width = int(width * (target_size / height))
            
#         image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
#         left = (new_width - target_size) // 2
#         top = (new_height - target_size) // 2
#         right = left + target_size
#         bottom = top + target_size
        
#         return image.crop((left, top, right, bottom))

#     def extract_frames_from_video(self, video_path: str) -> List[Image.Image]:
#         """Extract frames from video file using decord"""
#         decord.bridge.set_bridge('torch')  # Using torch bridge for better GPU support
        
#         try:
#             # Load video with decord
#             vr = VideoReader(video_path)
#             total_frames = len(vr)
            
#             # Calculate frame indices for sampling
#             if total_frames <= self.max_frames:
#                 frame_indices = list(range(total_frames))
#             else:
#                 # Sample frames evenly
#                 frame_indices = np.linspace(0, total_frames - 1, self.max_frames, dtype=int).tolist()
            
#             # Read frames
#             frames = vr.get_batch(frame_indices)
            
#             # Convert to PIL and process
#             processed_frames = []
#             for frame in frames:
#                 # Convert from torch tensor to PIL
#                 frame = frame.numpy()
#                 pil_image = Image.fromarray(frame)
#                 pil_image = self.resize_and_center_crop(pil_image, 384)
#                 processed_frames.append(pil_image)
                
#             return processed_frames
            
#         except Exception as e:
#             print(f"Error processing video {video_path}: {str(e)}")
#             raise

#     def extract_frames_from_gif(self, gif_path: str) -> List[Image.Image]:
#         """Extract frames from GIF file"""
#         gif = Image.open(gif_path)
#         frames = []
#         try:
#             while True:
#                 frames.append(gif.copy())
#                 gif.seek(gif.tell() + 1)
#         except EOFError:
#             pass

#         if len(frames) > self.max_frames:
#             indices = np.linspace(0, len(frames) - 1, self.max_frames, dtype=int)
#             frames = [frames[i] for i in indices]

#         return [self.resize_and_center_crop(frame.convert('RGB'), 384) for frame in frames]

#     def extract_frames_from_folder(self, folder_path: str) -> List[Image.Image]:
#         """Extract frames from folder containing numbered image files"""
#         image_extensions = ('.jpg', '.jpeg', '.png')
#         files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]
        
#         # Extract base filename and number using regex
#         import re
#         pattern = re.compile(r'(.+?)[-_]?(\d+)\..*$')
#         numbered_files = []
        
#         for file in files:
#             match = pattern.match(file)
#             if match:
#                 base, num = match.groups()
#                 numbered_files.append((file, base, int(num)))
        
#         if not numbered_files:
#             raise ValueError(f"No valid numbered image files found in {folder_path}")
            
#         # Sort by base filename and number
#         numbered_files.sort(key=lambda x: (x[1], x[2]))
#         sorted_files = [f[0] for f in numbered_files]
#         # Sample frames evenly if needed
#         if len(sorted_files) > self.max_frames:
#             indices = np.linspace(0, len(sorted_files) - 1, self.max_frames, dtype=int)
#             sorted_files = [sorted_files[i] for i in indices]
        
#         frames = []
#         for file in sorted_files:
#             image_path = os.path.join(folder_path, file)
#             image = Image.open(image_path)
#             image = self.resize_and_center_crop(image.convert('RGB'), 384)
#             frames.append(image)
            
#         return frames

#     def extract_frames(self, path: str) -> List[Image.Image]:
#         """Extract frames from video file, GIF, or folder of images"""
#         if os.path.isdir(path):
#             return self.extract_frames_from_folder(path)
#         elif path.lower().endswith('.gif'):
#             return self.extract_frames_from_gif(path)
#         else:
#             return self.extract_frames_from_video(path)




class VideoFrameExtractor:
    def __init__(self, max_frames: int = 50):
        self.max_frames = max_frames
        
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
        
    def extract_frames(self, video_path: str) -> List[Image.Image]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
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
        return frames

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

def generate_response(model, processor, video_path: str, question: str, max_frames: int = 50, temp_tokens: bool= False):
    # Extract frames
    frame_extractor = VideoFrameExtractor(max_frames)
    frames = frame_extractor.extract_frames(video_path)
    logger.info(f"Extracted {len(frames)} frames from video")
    
    # Create prompt with frames
    image_tokens = []
    for i in range(len(frames)):
        image_tokens.append({"type": "image"})
        if temp_tokens:
            if i < len(frames) -1:
                image_tokens.append({"type": "text", "text": f"<frame_{i}>"})

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
    temp_tokens = False
    # Configuration
    #checkpoint_path = "/fsx/miquel/smol-vision/smolvlm_frames8_with_temp_lr_1e-5/checkpoint-624"
    # checkpoint_path = "/fsx/miquel/smol-vision/smolvlm-longvumix-filter1-lowlrhighwarm_4_v2/checkpoint-4000"
    checkpoint_path = None
    base_model_id = "HuggingFaceTB/SmolVLM-Instruct"  
    # base_model_id = "HuggingFaceTB/SmolVLM_converted_4"
    video_path = "/fsx/miquel/fineVideo2Idefics/a/videos/--Dq6kFSRDE_scene_1.mp4"
    # video_path = "/fsx/miquel/cinepile/fulldatasetvideoscenes/00053/0005375.mp4"
    question = "can you explain step by step what is happening in this video?"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    logger.info("Loading model...")
    model, processor = load_model(checkpoint_path, base_model_id, device)

    if checkpoint_path is not None:
        if 'tokenizer.json' in os.listdir(os.path.dirname(checkpoint_path)):
            print("LOADING CUSTOM TOKENIZER")
            processor.tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(checkpoint_path))
            temp_tokens = True

    # Generate response
    logger.info("Generating response...")
    response = generate_response(model, processor, video_path, question, max_frames=100, temp_tokens=temp_tokens)
    
    # Print results
    print("Question:", question)
    print("Response:", response)

if __name__ == "__main__":
    main()