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
    """Qwen2-VL inferencer - properly supported by transformers."""
    
    # def __init__(self, model='/home/Md.Zama@mbzuai.ac.ae/Agent_Foundation_Models/models/Qwen2-VL-7B', device='cuda'):
    def __init__(self, model='/home/Md.Zama@mbzuai.ac.ae/Agent_Foundation_Models/models/Qwen2.5-VL-7B-Instruct', device='cuda'):
        # from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info
        import torch
        
        self.device = device
        
        # # Load model
        # self.model = Qwen2VLForConditionalGeneration.from_pretrained(
        #     model,
        #     torch_dtype=torch.float16,
        #     device_map="auto"
        # )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        # self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        #     model,
        #     torch_dtype=torch.float32,
        #     device_map=self.device
        # )
        
        # Load processor
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
                #  model='/home/Md.Zama@mbzuai.ac.ae/Agent_Foundation_Models/models/Qwen2-VL-7B',
                model='/home/Md.Zama@mbzuai.ac.ae/Agent_Foundation_Models/models/Qwen2.5-VL-7B-Instruct',
                 device: str = 'cuda',
                # device: str = 'cpu',
                 toolmeta=None):
        super().__init__(toolmeta=toolmeta)
        self.model = model
        self.device = device
        self._inferencer = None

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
        
        # prompt = 'Provide a comprehensive and detailed description of this image, including objects, people, actions, colors, and context. Limit your response to at most 300 words.'
        prompt = 'Provide a comprehensive and detailed description of this image, including objects, people, actions, colors, and context.'
        return self._inferencer(image, prompt, max_side=1280)
    


# class VideoDescription(BaseTool):
#     default_desc = "Returns a detailed description of the input video (multi-frame)."

#     def __init__(
#         self,
#         model='/share/users/md_zama/hf_cache/Qwen2VL7B',
#         device: str = 'cuda',
#         toolmeta=None,
#         num_frames: int = 12,
#         max_side: int = 1280,
#     ):
#         super().__init__(toolmeta=toolmeta)
#         self.model = model
#         self.device = device
#         self.num_frames = num_frames
#         self.max_side = max_side
#         self._inferencer = None

#     def setup(self):
#         if self._inferencer is None:
#             self._inferencer = Qwen2VLInferencer(model=self.model, device=self.device)

#     def _sample_frames(self, video_path: str, num_frames: int) -> List[np.ndarray]:
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             raise ValueError(f"Could not open video: {video_path}")

#         frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         if frame_count <= 0:
#             # fallback: read sequentially and take first num_frames
#             frames = []
#             while len(frames) < num_frames:
#                 ok, frame = cap.read()
#                 if not ok:
#                     break
#                 frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#             cap.release()
#             return frames

#         idxs = np.linspace(0, frame_count - 1, num_frames).astype(int)
#         frames = []
#         for i in idxs:
#             cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
#             ok, frame = cap.read()
#             if not ok:
#                 continue
#             frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#         cap.release()
#         return frames

#     def apply(self, video: Union[str, "VideoIO"]) -> str:
#         if self._inferencer is None:
#             self.setup()

#         # assume 'video' is a filepath; adapt if your VideoIO wraps a path/bytes
#         video_path = video if isinstance(video, str) else video.path

#         frames = self._sample_frames(video_path, self.num_frames)
#         if len(frames) == 0:
#             return "Could not decode frames from the video."

#         prompt = (
#             "You will be given multiple frames sampled across a video in time order.\n"
#             "1) Describe the overall scene and main entities.\n"
#             "2) Describe actions and how things change over time.\n"
#             "3) Note any key events, interactions, or state transitions.\n"
#             "Be specific and chronological when possible."
#         )

#         # IMPORTANT: this assumes your inferencer can take List[image]
#         # If it only supports a single image, use Option B below.
#         return self._inferencer(frames, prompt, max_side=self.max_side)


class CountGivenObject(BaseTool):
    default_desc = 'A tool that accurately counts the number of specific objects in an image.'

    def __init__(self,
                #  model='/home/Md.Zama@mbzuai.ac.ae/Agent_Foundation_Models/models/Qwen2-VL-7B',
                model='/home/Md.Zama@mbzuai.ac.ae/Agent_Foundation_Models/models/Qwen2.5-VL-7B-Instruct',
                 device: str = 'cuda',
                # device: str = 'cpu',
                 toolmeta=None):
        super().__init__(toolmeta=toolmeta)
        self.model = model
        self.device = device
        self._inferencer = None

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
                #  model='/home/Md.Zama@mbzuai.ac.ae/Agent_Foundation_Models/models/Qwen2-VL-7B',
                model='/home/Md.Zama@mbzuai.ac.ae/Agent_Foundation_Models/models/Qwen2.5-VL-7B-Instruct',
                 device: str = 'cuda',
                # device: str = 'cpu',
                 toolmeta=None):
        super().__init__(toolmeta=toolmeta)
        self.model = model
        self.device = device
        self._inferencer = None

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
    
# from typing import List, Optional
# from agentlego.types import ImageIO
# from agentlego.utils import load_or_build_object, require
# from agentlego.tools import BaseTool
# from agentlego.types import ImageIO, Annotated, Info


# class QwenVLInferencer:

#     def __init__(self,
#                  model='Qwen/Qwen-VL-Chat',
#                  revision='f57cfbd358cb56b710d963669ad1bcfb44cdcdd8',
#                  fp16=False,
#                  device=None):
#         from transformers import AutoModelForCausalLM, AutoTokenizer
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             model, trust_remote_code=True, revision=revision)

#         self.model = AutoModelForCausalLM.from_pretrained(
#             model,
#             device_map=device or 'auto',
#             trust_remote_code=True,
#             revision=revision,
#             fp16=fp16).eval()

#     def __call__(self, image: ImageIO, text: str):
#         query = self.tokenizer.from_list_format(
#             [dict(image=image.to_path()), dict(text=text)])
#         response, _ = self.model.chat(self.tokenizer, query=query, history=None)
#         return response

#     def fetch_all_box_with_ref(self, text) -> List[dict]:
#         items = self.tokenizer._fetch_all_box_with_ref(text)
#         return items



# # class ImageDescription(BaseTool):
# #     default_desc = ('A useful tool that returns a brief '
# #                     'description of the input image.')

# #     @require('mmpretrain')
# #     def __init__(self,
# #                 #  model: str = 'llava-7b-v1.5_vqa',
# #                 model: str = 'qwen2_vl_7b_instruct',
# #                  device: str = 'cuda',
# #                  toolmeta=None):
# #         super().__init__(toolmeta=toolmeta)
# #         self.model = model
# #         self.device = device

# #     def setup(self):
# #         from mmengine.registry import DefaultScope
# #         from mmpretrain.apis import VisualQuestionAnsweringInferencer
# #         with DefaultScope.overwrite_default_scope('mmpretrain'):
# #             self._inferencer = load_or_build_object(
# #                 VisualQuestionAnsweringInferencer,
# #                 model=self.model,
# #                 device=self.device,
# #             )

# #     def apply(self, image: ImageIO) -> str:
# #         image = image.to_array()[:, :, ::-1]
# #         return self._inferencer(image, 'Describe the image in detail')[0]['pred_answer']

# # class ImageDescription(BaseTool):
# #     default_desc = ('A useful tool that returns a detailed '
# #                     'description of the input image.')

# #     @require('mmpretrain')
# #     def __init__(self,
# #                  model: str = 'internvl2_8b',  # Great balance of quality/speed
# #                  device: str = 'cuda',
# #                  toolmeta=None):
# #         super().__init__(toolmeta=toolmeta)
# #         self.model = model
# #         self.device = device

# #     def setup(self):
# #         from mmengine.registry import DefaultScope
# #         from mmpretrain.apis import VisualQuestionAnsweringInferencer
# #         with DefaultScope.overwrite_default_scope('mmpretrain'):
# #             self._inferencer = load_or_build_object(
# #                 VisualQuestionAnsweringInferencer,
# #                 model=self.model,
# #                 device=self.device,
# #             )

# #     def apply(self, image: ImageIO) -> str:
# #         image = image.to_array()[:, :, ::-1]
# #         # You can also customize the prompt for better descriptions
# #         prompt = 'Provide a comprehensive and detailed description of this image, including objects, people, actions, colors, and context.'
# #         return self._inferencer(image, prompt)[0]['pred_answer']

# class ImageDescription(BaseTool):
#     default_desc = ('A useful tool that returns a detailed '
#                     'description of the input image.')

#     @require('mmpretrain')
#     def __init__(self,
#                  model: str = 'qwen2_vl_7b_instruct',  # Memory efficient and high quality
#                  device: str = 'cuda',
#                  toolmeta=None):
#         super().__init__(toolmeta=toolmeta)
#         self.model = model
#         self.device = device

#     def setup(self):
#         from mmengine.registry import DefaultScope
#         from mmpretrain.apis import VisualQuestionAnsweringInferencer
#         with DefaultScope.overwrite_default_scope('mmpretrain'):
#             self._inferencer = load_or_build_object(
#                 VisualQuestionAnsweringInferencer,
#                 model=self.model,
#                 device=self.device,
#             )

#     def apply(self, image: ImageIO) -> str:
#         image = image.to_array()[:, :, ::-1]
#         # Optimized prompt for Qwen2-VL
#         prompt = 'Provide a comprehensive and detailed description of this image, including objects, people, actions, colors, and context.'
#         return self._inferencer(image, prompt)[0]['pred_answer']


# # class CountGivenObject(BaseTool):
# #     default_desc = 'The tool can count the number of a certain object in the image.'

# #     @require('mmpretrain')
# #     def __init__(self,
# #                  model: str = 'llava-7b-v1.5_vqa',
# #                  device: str = 'cuda',
# #                  toolmeta=None):
# #         super().__init__(toolmeta=toolmeta)
# #         self.model = model
# #         self.device = device

# #     def setup(self):
# #         from mmengine.registry import DefaultScope
# #         from mmpretrain.apis import VisualQuestionAnsweringInferencer
# #         with DefaultScope.overwrite_default_scope('mmpretrain'):
# #             self._inferencer = load_or_build_object(
# #                 VisualQuestionAnsweringInferencer,
# #                 model=self.model,
# #                 device=self.device,
# #             )

# #     def apply(
# #         self,
# #         image: ImageIO,
# #         text: Annotated[str, Info('The object description in English.')],
# #         bbox: Annotated[Optional[str],
# #                         Info('The bbox coordinate in the format of `(x1, y1, x2, y2)`')] = None,
# #     ) -> int:
# #         import re
# #         if bbox is None:
# #             image = image.to_array()[:, :, ::-1]
# #         else:
# #             from agentlego.utils import parse_multi_float
# #             x1, y1, x2, y2 = (int(item) for item in parse_multi_float(bbox))
# #             image = image.to_array()[y1:y2, x1:x2, ::-1]
# #         res = self._inferencer(image, f'How many {text} are in the image? Reply a digit')[0]['pred_answer']
# #         res = re.findall(r'\d+', res)
# #         if len(res) > 0:
# #             return int(res[0])
# #         else:
# #             return 0

# class CountGivenObject(BaseTool):
#     default_desc = 'A tool that accurately counts the number of specific objects in an image.'

#     @require('mmpretrain')
#     def __init__(self,
#                  model: str = 'qwen2_vl_7b_instruct',  # Better at counting tasks
#                  device: str = 'cuda',
#                  toolmeta=None):
#         super().__init__(toolmeta=toolmeta)
#         self.model = model
#         self.device = device

#     def setup(self):
#         from mmengine.registry import DefaultScope
#         from mmpretrain.apis import VisualQuestionAnsweringInferencer
#         with DefaultScope.overwrite_default_scope('mmpretrain'):
#             self._inferencer = load_or_build_object(
#                 VisualQuestionAnsweringInferencer,
#                 model=self.model,
#                 device=self.device,
#             )

#     def apply(
#         self,
#         image: ImageIO,
#         text: Annotated[str, Info('The object description in English.')],
#         bbox: Annotated[Optional[str],
#                         Info('The bbox coordinate in the format of `(x1, y1, x2, y2)`')] = None,
#     ) -> int:
#         import re
        
#         # Handle bbox cropping if provided
#         if bbox is None:
#             image_array = image.to_array()[:, :, ::-1]
#         else:
#             from agentlego.utils import parse_multi_float
#             x1, y1, x2, y2 = (int(item) for item in parse_multi_float(bbox))
#             image_array = image.to_array()[y1:y2, x1:x2, ::-1]
        
#         # Enhanced prompt for better counting accuracy
#         prompts_to_try = [
#             f'Count the exact number of {text} in the image. Answer with only the number.',
#             f'How many {text} are there in this image? Reply with only a single number.',
#             f'Please count all {text} visible in the image and provide only the numerical count.'
#         ]
        
#         # Try multiple prompts and use the most common answer
#         answers = []
#         for prompt in prompts_to_try:
#             try:
#                 res = self._inferencer(image_array, prompt)[0]['pred_answer']
#                 # Extract numbers from response
#                 numbers = re.findall(r'\b\d+\b', res)
#                 if numbers:
#                     answers.append(int(numbers[0]))
#             except Exception as e:
#                 continue
        
#         # Return the most common count, or 0 if no valid answer
#         if answers:
#             from collections import Counter
#             # Get the most common answer
#             count = Counter(answers).most_common(1)[0][0]
#             return count
#         else:
#             return 0


# # class RegionAttributeDescription(BaseTool):
# #     default_desc = 'Describe the attribute of a region of the input image.'

# #     @require('mmpretrain')
# #     def __init__(self,
# #                  model: str = 'llava-7b-v1.5_vqa',
# #                  device: str = 'cuda',
# #                  toolmeta=None):
# #         super().__init__(toolmeta=toolmeta)
# #         self.model = model
# #         self.device = device


# #     def setup(self):
# #         from mmengine.registry import DefaultScope
# #         from mmpretrain.apis import VisualQuestionAnsweringInferencer
# #         with DefaultScope.overwrite_default_scope('mmpretrain'):
# #             self._inferencer = load_or_build_object(
# #                 VisualQuestionAnsweringInferencer,
# #                 model=self.model,
# #                 device=self.device,
# #             )

# #     def apply(
# #         self,
# #         image: ImageIO,
# #         bbox: Annotated[str,
# #                         Info('The bbox coordinate in the format of `(x1, y1, x2, y2)`')],
# #         attribute: Annotated[str, Info('The attribute to describe')],
# #     ) -> str:
# #         from agentlego.utils import parse_multi_float
# #         x1, y1, x2, y2 = (int(item) for item in parse_multi_float(bbox))
# #         cropped_image = image.to_array()[y1:y2, x1:x2, ::-1]
# #         return self._inferencer(cropped_image, f'Describe {attribute} on the image in detail')[0]['pred_answer']

# class RegionAttributeDescription(BaseTool):
#     default_desc = 'Accurately describes specific attributes of a region in an image.'

#     @require('mmpretrain')
#     def __init__(self,
#                  model: str = 'qwen2_vl_7b_instruct',  # Better at detailed region analysis
#                  device: str = 'cuda',
#                  toolmeta=None):
#         super().__init__(toolmeta=toolmeta)
#         self.model = model
#         self.device = device

#     def setup(self):
#         from mmengine.registry import DefaultScope
#         from mmpretrain.apis import VisualQuestionAnsweringInferencer
#         with DefaultScope.overwrite_default_scope('mmpretrain'):
#             self._inferencer = load_or_build_object(
#                 VisualQuestionAnsweringInferencer,
#                 model=self.model,
#                 device=self.device,
#             )

#     def _construct_attribute_prompt(self, attribute: str) -> str:
#         """Construct optimized prompts based on attribute type."""
#         attribute_lower = attribute.lower()
        
#         # Tailored prompts for common attributes
#         prompt_templates = {
#             'color': f'What is the color of the object in this image? Describe the color precisely.',
#             'texture': f'Describe the texture of the object in this image in detail.',
#             'material': f'What material is the object in this image made of? Be specific.',
#             'shape': f'Describe the shape of the object in this image.',
#             'size': f'Describe the size of the object in this image.',
#             'pattern': f'Describe any patterns visible on the object in this image.',
#             'style': f'Describe the style of the object in this image.',
#             'condition': f'Describe the condition or state of the object in this image.',
#             'age': f'Estimate the age or how old the object in this image appears to be.',
#             'brand': f'Can you identify the brand or manufacturer of the object in this image?',
#         }
        
#         # Check if attribute matches common types
#         for key, template in prompt_templates.items():
#             if key in attribute_lower:
#                 return template
        
#         # Generic prompt for other attributes
#         return f'Describe the {attribute} of the object in this image in detail. Be specific and thorough.'

#     def apply(
#         self,
#         image: ImageIO,
#         bbox: Annotated[str,
#                         Info('The bbox coordinate in the format of `(x1, y1, x2, y2)`')],
#         attribute: Annotated[str, Info('The attribute to describe (e.g., color, texture, material, shape)')],
#     ) -> str:
#         from agentlego.utils import parse_multi_float
        
#         # Parse and validate bbox coordinates
#         try:
#             x1, y1, x2, y2 = (int(item) for item in parse_multi_float(bbox))
#         except Exception as e:
#             return f"Error parsing bbox coordinates: {str(e)}"
        
#         # Validate bbox is within image bounds
#         img_array = image.to_array()
#         h, w = img_array.shape[:2]
        
#         x1, y1 = max(0, x1), max(0, y1)
#         x2, y2 = min(w, x2), min(h, y2)
        
#         if x2 <= x1 or y2 <= y1:
#             return "Invalid bbox: region has no area."
        
#         # Crop the region
#         cropped_image = img_array[y1:y2, x1:x2, ::-1]
        
#         # Construct optimized prompt
#         prompt = self._construct_attribute_prompt(attribute)
        
#         try:
#             response = self._inferencer(cropped_image, prompt)[0]['pred_answer']
#             return response.strip()
#         except Exception as e:
#             return f"Error during inference: {str(e)}"