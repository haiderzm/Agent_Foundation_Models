from typing import List, Optional
from collections import Counter
from agentlego.types import ImageIO
from agentlego.tools import BaseTool
from agentlego.types import Annotated, Info
import os
import tempfile
from PIL import Image
import numpy as np
import cv2
from typing import List, Union

class Qwen2VLInferencer:
    def __init__(self, model='/home/Md.Zama@mbzuai.ac.ae/Agent_Foundation_Models/models/Qwen2.5-VL-7B-Instruct', device='cuda'):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info
        import torch

        self.device = device

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model,
            torch_dtype=torch.float16,
            device_map=self.device
        )

        self.processor = AutoProcessor.from_pretrained(model)
        self.process_vision_info = process_vision_info
    
    @staticmethod
    def _resize_pil(pil_image: Image.Image, max_side: int) -> Image.Image:
        w, h = pil_image.size
        if max(w, h) <= max_side:
            return pil_image
        scale = max_side / float(max(w, h))
        new_w, new_h = int(w * scale), int(h * scale)
        return pil_image.resize((new_w, new_h), resample=Image.BICUBIC)
    
    def __call__(self, image: ImageIO, text: str, max_side: Optional[int] = None):
        """Run inference on image with text prompt."""
        from PIL import Image
        import numpy as np
        import torch
        
        # Convert ImageIO to PIL Image
        img_array = image.to_array()
        pil_image = Image.fromarray(img_array)

        if max_side is not None:
            pil_image = self._resize_pil(pil_image, max_side=max_side)
        
        # Prepare messages in Qwen2-VL format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": text},
                ],
            }
        ]
        
        # Prepare inputs
        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = self.process_vision_info(messages)
        
        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        
        # Decode
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        
        return output_text[0]

class ImageDescription(BaseTool):
    default_desc = ('A useful tool that returns a detailed '
                    'description of the input image.')
    def __init__(self,
                model='/home/Md.Zama@mbzuai.ac.ae/Agent_Foundation_Models/models/Qwen2.5-VL-7B-Instruct',
                 device: str = 'cuda',
                 inferencer=None,
                 toolmeta=None):
        super().__init__(toolmeta=toolmeta)
        self.model = model
        self.device = device
        self._inferencer = inferencer
    
    def setup(self):
        """Initialize the Qwen2-VL model."""
        if self._inferencer is None:
            self._inferencer = Qwen2VLInferencer(
                model=self.model,
                device=self.device
            )
    
    def apply(self, image: ImageIO) -> str:
        if self._inferencer is None:
            self.setup()
        
        prompt = 'Provide a comprehensive and detailed description of this image, including objects, people, actions, colors, and context.'
        return self._inferencer(image, prompt, max_side=1280)
    
class CountGivenObject(BaseTool):
    default_desc = 'A tool that accurately counts the number of specific objects in an image.'
    def __init__(self,
                model='/home/Md.Zama@mbzuai.ac.ae/Agent_Foundation_Models/models/Qwen2.5-VL-7B-Instruct',
                 device: str = 'cuda',
                 inferencer=None,
                 toolmeta=None):
        super().__init__(toolmeta=toolmeta)
        self.model = model
        self.device = device
        self._inferencer = inferencer
    
    def setup(self):
        """Initialize the Qwen2-VL model."""
        if self._inferencer is None:
            self._inferencer = Qwen2VLInferencer(
                model=self.model,
                device=self.device
            )

    def apply(
        self,
        image: ImageIO,
        text: Annotated[str, Info('The object description in English.')],
        bbox: Annotated[Optional[str],
                        Info('The bbox coordinate in the format of `(x1, y1, x2, y2)`')] = None,
    ) -> int:
        import re
        
        if self._inferencer is None:
            self.setup()
        
        # Handle bbox cropping by creating new ImageIO
        input_image = image
        if bbox is not None:
            from agentlego.utils import parse_multi_float
            from PIL import Image
            
            x1, y1, x2, y2 = (int(item) for item in parse_multi_float(bbox))
            img_array = image.to_array()[y1:y2, x1:x2]
            
            # Create temporary ImageIO from cropped array
            pil_image = Image.fromarray(img_array)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            pil_image.save(temp_file.name)
            temp_file.close()
            input_image = ImageIO(temp_file.name)
        
        prompts_to_try = [
            f'Count the exact number of {text} in the image. Answer with only the number.',
            f'How many {text} are there in this image? Reply with only a single number.',
        ]
        
        print("Object : ", text)

        answers = []
        for prompt in prompts_to_try:
            try:
                # res = self._inferencer(input_image, prompt)
                res = self._inferencer(input_image, prompt, max_side=1280)

                numbers = re.findall(r'\b\d+\b', res)
                if numbers:
                    answers.append(int(numbers[0]))
            except Exception:
                continue
        
        # Clean up temp file if created
        if bbox is not None:
            try:
                os.unlink(temp_file.name)
            except:
                pass
        
        if answers:
            count = Counter(answers).most_common(1)[0][0]
            return count
        else:
            return 0

class RegionAttributeDescription(BaseTool):
    default_desc = 'Accurately describes specific attributes of a region in an image.'
    def __init__(self,
                model='/home/Md.Zama@mbzuai.ac.ae/Agent_Foundation_Models/models/Qwen2.5-VL-7B-Instruct',
                 device: str = 'cuda',
                 inferencer=None,
                 toolmeta=None):
        super().__init__(toolmeta=toolmeta)
        self.model = model
        self.device = device
        self._inferencer = inferencer

    def setup(self):
        """Initialize the Qwen2-VL model."""
        if self._inferencer is None:
            self._inferencer = Qwen2VLInferencer(
                model=self.model,
                device=self.device
            )

    def _construct_attribute_prompt(self, attribute: str) -> str:
        attribute_lower = attribute.lower()
        
        prompt_templates = {
            'color': 'What is the color of the object in this image? Describe the color precisely.',
            'texture': 'Describe the texture of the object in this image in detail.',
            'material': 'What material is the object in this image made of? Be specific.',
            'shape': 'Describe the shape of the object in this image.',
            'size': 'Describe the size of the object in this image.',
            'pattern': 'Describe any patterns visible on the object in this image.',
            'style': 'Describe the style of the object in this image.',
            'condition': 'Describe the condition or state of the object in this image.',
            'age': 'Estimate the age or how old the object in this image appears to be.',
            'brand': 'Can you identify the brand or manufacturer of the object in this image?',
        }
        
        for key, template in prompt_templates.items():
            if key in attribute_lower:
                return template
        
        return f'Describe the {attribute} of the object in this image in detail. Be specific and thorough.'

    def apply(
        self,
        image: ImageIO,
        bbox: Annotated[str,
                        Info('The bbox coordinate in the format of `(x1, y1, x2, y2)`')],
        attribute: Annotated[str, Info('The attribute to describe')],
    ) -> str:
        from agentlego.utils import parse_multi_float
        from PIL import Image
        
        if self._inferencer is None:
            self.setup()
        
        try:
            x1, y1, x2, y2 = (int(item) for item in parse_multi_float(bbox))
        except Exception as e:
            return f"Error parsing bbox coordinates: {str(e)}"
        
        img_array = image.to_array()
        h, w = img_array.shape[:2]
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return "Invalid bbox: region has no area."
        
        # Crop and create temporary ImageIO
        cropped_image = img_array[y1:y2, x1:x2]
        pil_image = Image.fromarray(cropped_image)
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        pil_image.save(temp_file.name)
        temp_file.close()
        
        cropped_imageio = ImageIO(temp_file.name)
        prompt = self._construct_attribute_prompt(attribute)
        
        try:
            response = self._inferencer(cropped_imageio, prompt)
            try:
                os.unlink(temp_file.name)
            except:
                pass
            return response.strip()
        except Exception as e:
            try:
                os.unlink(temp_file.name)
            except:
                pass
            return f"Error during inference: {str(e)}"