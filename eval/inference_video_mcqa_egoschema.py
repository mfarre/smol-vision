import os
import re
import json
import argparse
import warnings
import logging
import torch
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from typing import List
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, Idefics3ForConditionalGeneration, AutoTokenizer
import decord
from decord import VideoReader


warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_MODEL_ID = "HuggingFaceTB/SmolVLM-Instruct"

# class VideoFrameExtractor:
#     def __init__(self, max_frames: int = 50):
#         self.max_frames = max_frames
        
#     def resize_and_center_crop(self, image: Image.Image, target_size: int) -> Image.Image:
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
        
#     def extract_frames(self, video_path: str) -> list:
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             raise ValueError(f"Could not open video: {video_path}")
            
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         fps = int(cap.get(cv2.CAP_PROP_FPS))
        
#         frame_indices = list(range(0, total_frames, fps))
        
#         if len(frame_indices) > self.max_frames:
#             indices = np.linspace(0, len(frame_indices) - 1, self.max_frames, dtype=int)
#             frame_indices = [frame_indices[i] for i in indices]
        
#         frames = []
#         for frame_idx in frame_indices:
#             cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
#             ret, frame = cap.read()
#             if ret:
#                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 pil_image = Image.fromarray(frame)
#                 pil_image = self.resize_and_center_crop(pil_image, 384)
#                 frames.append(pil_image)
        
#         cap.release()
#         return frames

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

class EgoschemaDataset(Dataset):
    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    def __init__(self, data_folder, data_list, frame_extractor):
        self.data_folder = data_folder
        self.data_list = data_list
        self.frame_extractor = frame_extractor

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        line = self.data_list[idx]
        q_uid = line['q_uid']

        for fmt in self.video_formats:
            temp_path = os.path.join(self.data_folder, f"{q_uid}{fmt}")
            if os.path.exists(temp_path):
                video_path = temp_path
                break
        else:
            raise FileNotFoundError(f"No video file found for {q_uid}")

        frames = self.frame_extractor.extract_frames(video_path)

        question = line['question']
        options = [
            line['option 0'], line['option 1'], line['option 2'],
            line['option 3'], line['option 4']
        ]
        
        instruct = (f'Question: {question}\nOptions:\n' + 
                   '\n'.join(f'({chr(65+i)}) {opt}' for i, opt in enumerate(options)) +
                   '\nAnswer with the option\'s letter from the given choices directly and only give the best option.')

        return {
            'q_uid': q_uid,
            'frames': frames,
            'instruct': instruct,
        }

def extract_answer_letter(response):
    # matches = re.findall(r'\(?([A-E])\)?', response)
    # if matches:
    #     return matches[0]
    try:
        return response.split("Answer: ")[1][0]
    except:
        logger.info('Returning None: No valid answer letter found in response')
        return None

def collate_fn(batch):
    return {
        'q_uid': [item['q_uid'] for item in batch],
        'frames': [item['frames'] for item in batch],
        'instruct': [item['instruct'] for item in batch],
    }

def run_inference(args):
    # Load model and processor
    if args.checkpoint_path:
        logger.info(f"Loading model from checkpoint: {args.checkpoint_path}")
        model_path = args.checkpoint_path
    else:
        logger.info(f"Loading vanilla model from {BASE_MODEL_ID}")
        model_path = BASE_MODEL_ID

    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)  # Always load processor from base model
    processor.image_processor.size = (384, 384)
    processor.image_processor.do_resize = False
    processor.image_processor.do_image_splitting = False

    temp_tokens = False

    if args.checkpoint_path:
            if 'tokenizer.json' in os.listdir(os.path.dirname(args.checkpoint_path)):
                print("LOADING CUSTOM TOKENIZER")
                processor.tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)
                temp_tokens = True
                print("VALIDATING OWN TOKEN ADDITIONS")
                print(processor.tokenizer.special_tokens_map)
                print(processor.tokenizer.additional_special_tokens)
                print(processor.tokenizer.pretrained_vocab_files_map)


    model = Idefics3ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Create output directory if needed
    answer_file = os.path.expanduser(args.answer_file)
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    ans_file = open(answer_file, "w")

    # Load questions and create dataset
    questions = json.load(open(args.question_file, "r"))
    frame_extractor = VideoFrameExtractor(max_frames=args.max_frames)
    dataset = EgoschemaDataset(args.video_folder, questions, frame_extractor)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    logger.info(f"Starting inference on {len(dataset)} questions")
    for batch in tqdm(dataloader):
        q_uid = batch['q_uid'][0]
        frames = batch['frames'][0]
        instruct = batch['instruct'][0]

        try:


            # Create prompt with frames
            image_tokens = []
            for i in range(len(frames)):
                image_tokens.append({"type": "image"})
                if i < len(frames) -1 and temp_tokens:
                    image_tokens.append({"type": "text", "text": f"<frame_{i}>"})

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Answer briefly."},
                        *image_tokens,
                        {"type": "text", "text": instruct}
                    ]
                }
            ]
            # # Create image tokens and messages structure
            # image_tokens = [{"type": "image"} for _ in range(len(frames))]
            # messages = [
            #     {
            #         "role": "user",
            #         "content": [
            #             {"type": "text", "text": "Answer briefly."},
            #             *image_tokens,
            #             {"type": "text", "text": instruct}
            #         ]
            #     }
            # ]

            # Process inputs
            inputs = processor(
                text=processor.apply_chat_template(messages, add_generation_prompt=True),
                images=frames,
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

            # Decode and extract answer
            response = processor.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"\nFull model response for {q_uid}: {response}")
            
            answer_letter = extract_answer_letter(response)
            
            if answer_letter:
                pred_idx = ord(answer_letter) - ord('A')
                logger.info(f"PARSED Prediction {pred_idx}")
            else:
                logger.warning(f'No valid answer found in response for {q_uid}')
                pred_idx = -1

            ans_file.write(f'{q_uid}, {pred_idx}\n')

        except Exception as e:
            logger.error(f"Error processing q_uid {q_uid}: {str(e)}")
            ans_file.write(f'{q_uid}, -1\n')

    ans_file.close()
    logger.info(f"Inference complete. Results written to {answer_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Video QA Evaluation Script for SmolVLM')
    parser.add_argument('--video-folder', help='Directory containing video files', required=True)
    parser.add_argument('--question-file', help='Path to the questions file', required=True)
    parser.add_argument('--answer-file', help='Path to save the answers', required=True)
    parser.add_argument('--max-frames', type=int, default=50, help='Maximum number of frames to extract per video')
    parser.add_argument('--checkpoint-path', help='Path to a fine-tuned checkpoint (optional)', default=None)

    args = parser.parse_args()
    run_inference(args)