import torch
from PIL import Image
import numpy as np
from typing import List, Tuple

class VisionContextTester:
    def __init__(self, model, tokenizer, processor):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        
    def create_test_image(self, text: str, image_index: int) -> Image:
        """Creates a test image with embedded text.
        
        Args:
            text: Text to write on image
            image_index: Index of this image in the sequence (used for positioning)
        """
        size = (384, 384)  # Fixed size for the vision model
        """Creates a test image with embedded text.
        
        Args:
            size: Image dimensions (width, height)
            text: Text to write on image
            image_index: Index of this image in the sequence (used for positioning)
        """
        img = Image.new('RGB', size, color='white')
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        
        try:
            # Try to load Arial font, fallback to default if not available
            font = ImageFont.truetype("arial.ttf", 30)  # Larger font for 384x384 images
        except IOError:
            font = ImageFont.load_default(20)
        
        # Add index marker (1-based indexing)
        draw.text((10, 10), f"Image #{image_index + 1}", fill="black", font=font)
        
        # Split text into lines for better visibility
        words = text.split()
        lines = []
        current_line = []
        for word in words:
            current_line.append(word)
            if len(" ".join(current_line)) > 30:  # characters per line for larger image
                lines.append(" ".join(current_line[:-1]))
                current_line = [word]
        if current_line:
            lines.append(" ".join(current_line))
            
        # Draw text lines
        y_position = 40
        for line in lines:
            draw.text((10, y_position), line, fill="black", font=font)
            y_position += 25
            
        return img
        
    def calculate_perplexity(self, outputs, labels) -> float:
        """Calculate perplexity from model outputs."""
        loss = torch.nn.CrossEntropyLoss()(
            outputs.logits.view(-1, outputs.logits.size(-1)),
            labels.view(-1)
        )
        return torch.exp(loss).item()
        
    def generate_context_dependent_content(self, num_images: int) -> List[str]:
        """Generate content where each image refers to previous ones."""
        contents = []
        for i in range(num_images):
            if i == 0:
                content = "This is the first image containing a secret number: 7425."
            else:
                # Reference previous images to create dependencies
                references = []
                # Reference up to 3 previous images
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

    def test_context_window(self, 
                           max_length: int,
                           step_size: int = 1000,
                           num_trials: int = 5) -> List[Tuple[int, float]]:
        """
        Test the model's context window by incrementally increasing input length
        and measuring perplexity.
        
        Args:
            max_length: Maximum sequence length to test
            step_size: Increment size for sequence length
            num_trials: Number of trials per length to average
            
        Returns:
            List of tuples containing (sequence_length, avg_perplexity)
        """
        results = []
        
        for length in range(step_size, max_length + step_size, step_size):
            trial_perplexities = []
            
            for _ in range(num_trials):
                # Calculate number of images needed for this length
                num_images = length // step_size
                
                # Generate context-dependent content
                contents = self.generate_context_dependent_content(num_images)
                
                # Create test images
                test_images = [
                    self.create_test_image(
                        text=content,
                        image_index=i
                    )
                    for i, content in enumerate(contents)
                ]
                
                # Create input text of specific length
                input_text = " ".join(["test"] * (length // 4))  # Approximate tokens
                
                # Process inputs
                inputs = self.processor(
                    images=test_image,
                    text=input_text,
                    return_tensors="pt"
                )
                
                # Generate outputs
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Calculate perplexity
                perplexity = self.calculate_perplexity(outputs, inputs["labels"])
                trial_perplexities.append(perplexity)
            
            avg_perplexity = np.mean(trial_perplexities)
            results.append((length, avg_perplexity))
            
            # Check for significant perplexity increase
            if len(results) > 1:
                prev_perplexity = results[-2][1]
                if avg_perplexity > prev_perplexity * 1.5:  # 50% increase threshold
                    print(f"Significant perplexity increase at length {length}")
                    break
                    
        return results

    def plot_results(self, results: List[Tuple[int, float]]) -> None:
        """Plot the perplexity results."""
        import matplotlib.pyplot as plt
        
        lengths, perplexities = zip(*results)
        
        plt.figure(figsize=(10, 6))
        plt.plot(lengths, perplexities, marker='o')
        plt.xlabel('Sequence Length')
        plt.ylabel('Perplexity')
        plt.title('Vision LLM Context Window Analysis')
        plt.grid(True)
        plt.show()

# Example usage:
def generate_test_images(num_images: int, output_dir: str = "generated") -> None:
    """
    Generate test images with context-dependent content and save them to disk.
    
    Args:
        num_images: Number of images to generate
        output_dir: Directory to save the images (will be created if it doesn't exist)
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a tester instance without model (just for image generation)
    tester = VisionContextTester(model=None, tokenizer=None, processor=None)
    
    # Generate context-dependent content
    contents = tester.generate_context_dependent_content(num_images)
    
    # Generate and save images
    for i, content in enumerate(contents):
        image = tester.create_test_image(text=content, image_index=i)
        output_path = os.path.join(output_dir, f"image_{i}.png")
        image.save(output_path)
        print(f"Generated {output_path}")
        print(f"Content: {content}\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate test images for vision LLM context testing")
    parser.add_argument("num_images", type=int, help="Number of images to generate")
    parser.add_argument("--output-dir", type=str, default="generated", 
                      help="Output directory for generated images (default: generated)")
    
    args = parser.parse_args()
    
    generate_test_images(args.num_images, args.output_dir)