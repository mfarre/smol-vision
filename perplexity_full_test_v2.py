import torch
from transformers import AutoProcessor, Idefics3ForConditionalGeneration
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import logging
from typing import List, Tuple
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageGenerator:
    def create_test_image(self, text: str, image_index: int) -> Image:
        """Creates a test image with embedded text."""
        size = (384, 384)  # Fixed size for the vision model
        img = Image.new('RGB', size, color='white')
        draw = ImageDraw.Draw(img)
        
        try:
            # Try to load Arial font, fallback to default if not available
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default(20)  # Set size for default font too
        
        # Add index marker (1-based indexing)
        draw.text((10, 10), f"Image #{image_index + 1}", fill="black", font=font)
        
        # Split text into lines for better visibility
        words = text.split()
        lines = []
        current_line = []
        for word in words:
            current_line.append(word)
            if len(" ".join(current_line)) > 30:  # characters per line
                lines.append(" ".join(current_line[:-1]))
                current_line = [word]
        if current_line:
            lines.append(" ".join(current_line))
            
        # Draw text lines with adjusted spacing
        y_position = 40
        for line in lines:
            draw.text((10, y_position), line, fill="black", font=font)
            y_position += 25
            
        return img

    def generate_context_dependent_content(self, num_images: int) -> List[str]:
        """Generate content where each image refers to previous ones."""
        contents = []
        for i in range(num_images):
            if i == 0:
                content = "This is the first image containing a secret number: 7425."
            else:
                references = []
                for j in range(max(0, i-3), i):
                    if j == 0:
                        references.append("the secret number from image #1")
                    else:
                        references.append(f"the calculation from image #{j+1}")
                
                if i == 1:
                    content = f"Using {references[0]}, multiply it by 2 to get: 14850."
                else:
                    prev_refs = ", ".join(references[:-1])
                    if prev_refs:
                        prev_refs += " and "
                    prev_refs += references[-1]
                    content = f"Take {prev_refs}, add them together and divide by {i} to get a new number."
            
            contents.append(content)
        return contents

    def generate_test_images(self, num_images: int, output_dir: str = "generated") -> None:
        """Generate test images with context-dependent content."""
        os.makedirs(output_dir, exist_ok=True)
        
        contents = self.generate_context_dependent_content(num_images)
        
        for i, content in enumerate(contents):
            image = self.create_test_image(text=content, image_index=i)
            output_path = os.path.join(output_dir, f"image_{i}.png")
            image.save(output_path)
            logger.info(f"Generated {output_path}")
            logger.debug(f"Content: {content}")


class SmolVLMContextTester:
    def __init__(self, model_id: str = "HuggingFaceTB/SmolVLM-Instruct", device: str = "cuda"):
        logger.info(f"Loading model {model_id}...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Idefics3ForConditionalGeneration.from_pretrained(
            model_id,
            # "/fsx/miquel/smol-vision/smolvlm_longvucauldron_frames50_no_temp_lr_5e-7/checkpoint-1736",
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        
        # Configure processor for correct image size
        self.processor.image_processor.size = (384, 384)
        self.processor.image_processor.do_resize = False
        self.processor.image_processor.do_image_splitting = False
        
    def load_images_from_folder(self, folder_path: str) -> List[Image.Image]:
        """Load images in order from folder"""
        image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])
        images = []
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            images.append(Image.open(img_path))
        return images
        
    def test_context_window(self, images_folder: str, step_size: int = 2) -> List[Tuple[int, float, str]]:
        """Test context window by incrementally adding more images."""
        results = []
        all_images = self.load_images_from_folder(images_folder)
        
        question = "What was the initial secret number, and what calculations were performed in each subsequent image? List them in order."
        
        for num_images in range(step_size, len(all_images) + 1, step_size):
            current_images = all_images[:num_images]
            logger.info(f"Testing with {num_images} images...")
            
            image_tokens = []
            for i in range(len(current_images)):
                image_tokens.append({"type": "image"})
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Answer step by step."},
                        *image_tokens,
                        {"type": "text", "text": question}
                    ]
                }
            ]
            
            inputs = self.processor(
                text=self.processor.apply_chat_template(messages, add_generation_prompt=True),
                images=current_images,
                return_tensors="pt"
            ).to(self.model.device)
            
            with torch.no_grad():
                # Get perplexity
                outputs = self.model(**inputs)
                loss = torch.nn.CrossEntropyLoss()(
                    outputs.logits[0, :-1].view(-1, outputs.logits.size(-1)),
                    inputs.input_ids[0, 1:].view(-1)
                )
                perplexity = torch.exp(loss).item()
                
                # Generate response
                response_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    num_beams=5,
                    temperature=0.7,
                    do_sample=True,
                    use_cache=True
                )
                response = self.processor.decode(response_ids[0], skip_special_tokens=True)
            
            results.append((num_images, perplexity, response))
            
            # Calculate percentage increase from last step if available
            if len(results) > 1:
                prev_num, prev_perplexity, _ = results[-2]
                increase_percent = ((perplexity - prev_perplexity) / prev_perplexity) * 100
                logger.info(f"Perplexity increase from {prev_num} to {num_images} images: {increase_percent:.2f}%")
            
            logger.info(f"Perplexity with {num_images} images: {perplexity}")
            logger.info(f"Response: {response}\n")
            
            # if len(results) > 1:
            #     _, prev_perplexity, _ = results[-2]
            #     if perplexity > prev_perplexity * 50.0:  # 5000% increase threshold
            #         logger.info(f"Significant perplexity increase detected at {num_images} images")
            #         break
        
        return results


def ensure_test_images(num_images: int, images_dir: str):
    """Ensure test images exist, generate if they don't."""
    if not os.path.exists(images_dir) or len([f for f in os.listdir(images_dir) if f.endswith('.png')]) < num_images:
        logger.info(f"Generating {num_images} test images...")
        generator = ImageGenerator()
        generator.generate_test_images(num_images, images_dir)
    else:
        logger.info(f"Using existing test images from {images_dir}")


def main():
    parser = argparse.ArgumentParser(description="Test SmolVLM context window with generated images")
    parser.add_argument("--images-dir", type=str, default="generated",
                      help="Directory containing test images (default: generated)")
    parser.add_argument("--step-size", type=int, default=2,
                      help="Number of images to add in each step (default: 2)")
    parser.add_argument("--device", type=str, default="cuda",
                      help="Device to run the model on (default: cuda)")
    parser.add_argument("--num-images", type=int, default=500,
                      help="Number of test images to pre-generate (default: 500)")
    
    args = parser.parse_args()
    
    # Ensure we have enough test images
    ensure_test_images(args.num_images, args.images_dir)
    
    # Initialize tester and run tests
    tester = SmolVLMContextTester(device=args.device)
    results = tester.test_context_window(
        images_folder=args.images_dir,
        step_size=args.step_size
    )
    
    # Print results
    print("\nResults:")
    print("Number of Images | Perplexity | Response")
    print("-" * 100)
    for num_images, perplexity, response in results:
        # Truncate response for display
        truncated_response = response[:50] + "..." if len(response) > 50 else response
        print(f"{num_images:^14} | {perplexity:.4f} | {truncated_response}")

    # Save detailed results to file
    # output_file = "context_test_results_sft.txt"
    output_file = "context_test_results.txt"
    with open(output_file, "w") as f:
        f.write("Number of Images | Perplexity | Response\n")
        f.write("-" * 100 + "\n")
        for num_images, perplexity, response in results:
            f.write(f"{num_images:^14} | {perplexity:.4f} | {response}\n\n")
    
    logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()