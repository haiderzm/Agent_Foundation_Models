# from typing import Sequence, Tuple, Union

# from agentlego.types import Annotated, ImageIO, Info
# from agentlego.utils import load_or_build_object, require
# from ..base import BaseTool

# import os 

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"     # hide INFO & WARNING logs
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  

# class OCR(BaseTool):
#     """A tool to recognize the optical characters on an image using PaddleOCR.

#     Args:
#         lang (str): Language to recognize. Examples: 'en', 'ch', 'fr', 'de'.
#         use_angle_cls (bool): Whether to use angle classifier.
#         device (str | bool): 'cpu', 'gpu', or True for auto.
#         toolmeta: Tool metadata.
#         **ocr_args: Additional PaddleOCR arguments.
#     """

#     default_desc = 'This tool can recognize all text on the input image.'

#     @require('paddleocr')
#     def __init__(
#         self,
#         lang: str = 'en',
#         use_angle_cls: bool = True,
#         device: Union[str, bool] = True,
#         toolmeta=None,
#         **ocr_args,
#     ):
#         super().__init__(toolmeta=toolmeta)
#         self.lang = lang
#         self.use_angle_cls = use_angle_cls
#         self.device = device
#         self.ocr_args = ocr_args

#     def setup(self):
#         from paddleocr import PaddleOCR

#         use_gpu = False
#         # if self.device is True or self.device == 'gpu':
#         #     use_gpu = True

#         self._ocr = load_or_build_object(
#             PaddleOCR,
#             use_angle_cls=self.use_angle_cls,
#             lang=self.lang,
#             use_gpu=False,
#             **self.ocr_args,
#         )

#     def apply(
#         self,
#         image: ImageIO,
#     ) -> Annotated[
#         str,
#         Info(
#             'OCR results, include bbox in x1, y1, x2, y2 format '
#             'and the recognized text.'
#         ),
#     ]:
#         image = image.to_array()
#         results = self._ocr.ocr(image, cls=self.use_angle_cls)

#         outputs = []

#         if not results:
#             return ""

#         for line in results:
#             for item in line:
#                 box, (text, conf) = item
#                 x1, y1, x2, y2 = self.extract_bbox(box)
#                 outputs.append(f"({x1}, {y1}, {x2}, {y2}) {text}")

#         return "\n".join(outputs)

#     @staticmethod
#     def extract_bbox(box) -> Tuple[int, int, int, int]:
#         """
#         PaddleOCR box format:
#         [
#             [x1, y1],
#             [x2, y2],
#             [x3, y3],
#             [x4, y4]
#         ]
#         """
#         xs = [int(p[0]) for p in box]
#         ys = [int(p[1]) for p in box]
#         return min(xs), min(ys), max(xs), max(ys)


from typing import Sequence, Tuple, Union, List
import requests

from agentlego.types import Annotated, ImageIO, Info
from ..base import BaseTool


class OCR(BaseTool):
    """A tool to recognize the optical characters on an image.

    NOTE:
    This implementation delegates OCR to a local PaddleOCR HTTP service.
    """

    default_desc = 'This tool can recognize all text on the input image.'

    def __init__(self,
                 lang: Union[str, Sequence[str]] = 'en',
                 line_group_tolerance: int = -1,
                 device: Union[bool, str] = True,
                 toolmeta=None,
                 **kwargs):
        # lang / device kept ONLY for API compatibility
        super().__init__(toolmeta=toolmeta)
        self.line_group_tolerance = line_group_tolerance

        # PaddleOCR server endpoint
        # self.ocr_endpoint = "http://10.127.30.114:9104/ocr"
        self.ocr_endpoint = "http://127.0.0.1:9104/ocr"
        self.timeout = 30  # seconds

    def setup(self):
        # No heavy model loading here
        # PaddleOCR lives in a separate service
        pass

    def apply(
        self,
        image: ImageIO,
    ) -> Annotated[str,
                   Info('OCR results, include bbox in x1, y1, x2, y2 format '
                        'and the recognized text.')]:
        # We must send IMAGE PATH (not array) to PaddleOCR service
        # image_path = image
        # if image_path is None:
        #     raise ValueError("OCR requires image path, not in-memory image")

        if hasattr(image, 'to_path'):
            image_path = image.to_path()  # <-- this ensures we get a file path
        else:
            raise ValueError("OCR requires an ImageIO object that can provide a path")


        resp = requests.post(
            self.ocr_endpoint,
            json={"image_path": image_path},
            timeout=self.timeout,
        )

        if resp.status_code != 200:
            raise RuntimeError(
                f"PaddleOCR server error {resp.status_code}: {resp.text}"
            )

        data = resp.json()
        items = data.get("results", [])

        # Convert to [(bbox, text)] format
        results: List[Tuple[Tuple[int, int, int, int], str]] = [
            ((item["bbox"][0], item["bbox"][1],
              item["bbox"][2], item["bbox"][3]),
             item["text"])
            for item in items
        ]

        # --------------------------------------------------
        # Line grouping (same logic as original EasyOCR code)
        # --------------------------------------------------
        if self.line_group_tolerance >= 0:
            results.sort(key=lambda x: x[0][1])  # sort by y1

            groups = []
            group = []

            for item in results:
                if not group:
                    group.append(item)
                    continue

                if abs(item[0][1] - group[-1][0][1]) <= self.line_group_tolerance:
                    group.append(item)
                else:
                    groups.append(group)
                    group = [item]

            groups.append(group)

            merged = []
            for group in groups:
                line = sorted(group, key=lambda x: x[0][0])  # sort by x1
                bboxes = [item[0] for item in line]
                text = ' '.join(item[1] for item in line)
                merged.append((self.extract_bbox(bboxes), text))

            results = merged

        # --------------------------------------------------
        # Format output EXACTLY like original OCR tool
        # --------------------------------------------------
        outputs = [
            f"({x1}, {y1}, {x2}, {y2}) {text}"
            for (x1, y1, x2, y2), text in results
        ]

        return "\n".join(outputs)

    @staticmethod
    def extract_bbox(bboxes) -> Tuple[int, int, int, int]:
        xs = [b[0] for b in bboxes] + [b[2] for b in bboxes]
        ys = [b[1] for b in bboxes] + [b[3] for b in bboxes]
        return min(xs), min(ys), max(xs), max(ys)


# from typing import Sequence, Tuple, Union

# from agentlego.types import Annotated, ImageIO, Info
# from agentlego.utils import load_or_build_object, require
# from ..base import BaseTool


# class OCR(BaseTool):
#     """A tool to recognize the optical characters on an image.

#     Args:
#         lang (str | Sequence[str]): The language to be recognized.
#             Defaults to 'en'.
#         line_group_tolerance (int): The line group tolerance threshold.
#             Defaults to -1, which means to disable the line group method.
#         device (str | bool): The device to load the model. Defaults to True,
#             which means automatically select device.
#         **read_args: Other keyword arguments for read text. Please check the
#             `EasyOCR docs <https://www.jaided.ai/easyocr/documentation/>`_.
#         toolmeta (None | dict | ToolMeta): The additional info of the tool.
#             Defaults to None.
#     """

#     default_desc = 'This tool can recognize all text on the input image.'

#     @require('easyocr')
#     def __init__(self,
#                  lang: Union[str, Sequence[str]] = 'en',
#                  line_group_tolerance: int = -1,
#                  device: Union[bool, str] = True,
#                  toolmeta=None,
#                  **read_args):
#         super().__init__(toolmeta=toolmeta)
#         if isinstance(lang, str):
#             lang = [lang]
#         self.lang = list(lang)
#         self.read_args = read_args
#         self.device = device
#         self.line_group_tolerance = line_group_tolerance
#         read_args.setdefault('decoder', 'beamsearch')

#         if line_group_tolerance >= 0:
#             read_args.setdefault('paragraph', False)
#         else:
#             read_args.setdefault('paragraph', True)

#     def setup(self):
#         import easyocr
#         self._reader: easyocr.Reader = load_or_build_object(
#             easyocr.Reader, self.lang, gpu=self.device)

#     def apply(
#         self,
#         image: ImageIO,
#     ) -> Annotated[str,
#                    Info('OCR results, include bbox in x1, y1, x2, y2 format '
#                         'and the recognized text.')]:

#         image = image.to_array()
#         results = self._reader.readtext(image, detail=1, **self.read_args)
#         results = [(self.extract_bbox(item[0]), item[1]) for item in results]

#         if self.line_group_tolerance >= 0:
#             results.sort(key=lambda x: x[0][1])

#             groups = []
#             group = []

#             for item in results:
#                 if not group:
#                     group.append(item)
#                     continue

#                 if abs(item[0][1] - group[-1][0][1]) <= self.line_group_tolerance:
#                     group.append(item)
#                 else:
#                     groups.append(group)
#                     group = [item]

#             groups.append(group)

#             results = []
#             for group in groups:
#                 # For each line, sort the elements by their left x-coordinate and join their texts
#                 line = sorted(group, key=lambda x: x[0][0])
#                 bboxes = [item[0] for item in line]
#                 text = ' '.join(item[1] for item in line)
#                 results.append((self.extract_bbox(bboxes), text))

#         outputs = []
#         for item in results:
#             outputs.append('({}, {}, {}, {}) {}'.format(*item[0], item[1]))
#         outputs = '\n'.join(outputs)
#         return outputs

#     @staticmethod
#     def extract_bbox(char_boxes) -> Tuple[int, int, int, int]:
#         xs = [int(box[0]) for box in char_boxes]
#         ys = [int(box[1]) for box in char_boxes]
#         return min(xs), min(ys), max(xs), max(ys)
