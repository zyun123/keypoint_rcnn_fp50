from PIL.Image import Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from PIL import ImageColor
import numpy as np
import torch
from types import FunctionType
STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def draw_text(draw,
              box: list,
              cls: int,
              score: float,
              category_index: dict,
              color: str,
              font: str = 'arial.ttf',
              font_size: int = 24):
    """
    将目标边界框和类别信息绘制到图片上
    """
    try:
        font = ImageFont.truetype(font, font_size)
    except IOError:
        font = ImageFont.load_default()

    left, top, right, bottom = box
    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str = f"{category_index[str(cls)]}: {int(100 * score)}%"
    display_str_heights = [font.getsize(ds)[1] for ds in display_str]
    # Each display_str has a top and bottom margin of 0.05x.
    display_str_height = (1 + 2 * 0.05) * max(display_str_heights)

    if top > display_str_height:
        text_top = top - display_str_height
        text_bottom = top
    else:
        text_top = bottom
        text_bottom = bottom + display_str_height

    for ds in display_str:
        text_width, text_height = font.getsize(ds)
        margin = np.ceil(0.05 * text_width)
        draw.rectangle([(left, text_top),
                        (left + text_width + 2 * margin, text_bottom)], fill=color)
        draw.text((left + margin, text_top),
                  ds,
                  fill='black',
                  font=font)
        left += text_width


def draw_masks(image, masks, colors, thresh: float = 0.7, alpha: float = 0.5):
    np_image = np.array(image)
    masks = np.where(masks > thresh, True, False)

    # colors = np.array(colors)
    img_to_draw = np.copy(np_image)
    # TODO: There might be a way to vectorize this
    for mask, color in zip(masks, colors):
        img_to_draw[mask] = color

    out = np_image * (1 - alpha) + img_to_draw * alpha
    return fromarray(out.astype(np.uint8))


def draw_objs(image: Image,
              boxes: np.ndarray = None,
              classes: np.ndarray = None,
              scores: np.ndarray = None,
              masks: np.ndarray = None,
              category_index: dict = None,
              box_thresh: float = 0.1,
              mask_thresh: float = 0.5,
              line_thickness: int = 8,
              font: str = 'arial.ttf',
              font_size: int = 24,
              draw_boxes_on_image: bool = True,
              draw_masks_on_image: bool = True):
    """
    将目标边界框信息，类别信息，mask信息绘制在图片上
    Args:
        image: 需要绘制的图片
        boxes: 目标边界框信息
        classes: 目标类别信息
        scores: 目标概率信息
        masks: 目标mask信息
        category_index: 类别与名称字典
        box_thresh: 过滤的概率阈值
        mask_thresh:
        line_thickness: 边界框宽度
        font: 字体类型
        font_size: 字体大小
        draw_boxes_on_image:
        draw_masks_on_image:

    Returns:

    """

    # 过滤掉低概率的目标
    idxs = np.greater(scores, box_thresh)
    boxes = boxes[idxs]
    classes = classes[idxs]
    scores = scores[idxs]
    if masks is not None:
        masks = masks[idxs]
    if len(boxes) == 0:
        return image

    colors = [ImageColor.getrgb(STANDARD_COLORS[cls % len(STANDARD_COLORS)]) for cls in classes]

    if draw_boxes_on_image:
        # Draw all boxes onto image.
        draw = ImageDraw.Draw(image)
        for box, cls, score, color in zip(boxes, classes, scores, colors):
            left, top, right, bottom = box
            # 绘制目标边界框
            draw.line([(left, top), (left, bottom), (right, bottom),
                       (right, top), (left, top)], width=line_thickness, fill=color)
            # 绘制类别和概率信息
            draw_text(draw, box.tolist(), int(cls), float(score), category_index, color, font, font_size)

    if draw_masks_on_image and (masks is not None):
        # Draw all mask onto image.
        image = draw_masks(image, masks, colors, mask_thresh)

    return image


from typing import Any, BinaryIO, List, Optional, Tuple, Union
def draw_keypoints(
    image: torch.Tensor,
    keypoints: torch.Tensor,
    connectivity: Optional[List[Tuple[int, int]]] = None,
    colors: Optional[Union[str, Tuple[int, int, int]]] = None,
    radius: int = 2,
    width: int = 3,
) -> torch.Tensor:

    """
    Draws Keypoints on given RGB image.
    The values of the input image should be uint8 between 0 and 255.

    Args:
        image (Tensor): Tensor of shape (3, H, W) and dtype uint8.
        keypoints (Tensor): Tensor of shape (num_instances, K, 2) the K keypoints location for each of the N instances,
            in the format [x, y].
        connectivity (List[Tuple[int, int]]]): A List of tuple where,
            each tuple contains pair of keypoints to be connected.
        colors (str, Tuple): The color can be represented as
            PIL strings e.g. "red" or "#FF00FF", or as RGB tuples e.g. ``(240, 10, 157)``.
        radius (int): Integer denoting radius of keypoint.
        width (int): Integer denoting width of line connecting keypoints.

    Returns:
        img (Tensor[C, H, W]): Image Tensor of dtype uint8 with keypoints drawn.
    """

    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(draw_keypoints)
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"The image must be a tensor, got {type(image)}")
    elif image.dtype != torch.uint8:
        raise ValueError(f"The image dtype must be uint8, got {image.dtype}")
    elif image.dim() != 3:
        raise ValueError("Pass individual images, not batches")
    elif image.size()[0] != 3:
        raise ValueError("Pass an RGB image. Other Image formats are not supported")

    if keypoints.ndim != 3:
        raise ValueError("keypoints must be of shape (num_instances, K, 2)")

    ndarr = image.permute(1, 2, 0).cpu().numpy()
    img_to_draw = Image.fromarray(ndarr)
    draw = ImageDraw.Draw(img_to_draw)
    img_kpts = keypoints.to(torch.int64).tolist()

    for kpt_id, kpt_inst in enumerate(img_kpts):
        for inst_id, kpt in enumerate(kpt_inst):
            x1 = kpt[0] - radius
            x2 = kpt[0] + radius
            y1 = kpt[1] - radius
            y2 = kpt[1] + radius
            draw.ellipse([x1, y1, x2, y2], fill=colors, outline=None, width=0)

        if connectivity:
            for connection in connectivity:
                start_pt_x = kpt_inst[connection[0]][0]
                start_pt_y = kpt_inst[connection[0]][1]

                end_pt_x = kpt_inst[connection[1]][0]
                end_pt_y = kpt_inst[connection[1]][1]

                draw.line(
                    ((start_pt_x, start_pt_y), (end_pt_x, end_pt_y)),
                    width=width,
                )

    return torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1).to(dtype=torch.uint8)


def _log_api_usage_once(obj: Any) -> None:

    """
    Logs API usage(module and name) within an organization.
    In a large ecosystem, it's often useful to track the PyTorch and
    TorchVision APIs usage. This API provides the similar functionality to the
    logging module in the Python stdlib. It can be used for debugging purpose
    to log which methods are used and by default it is inactive, unless the user
    manually subscribes a logger via the `SetAPIUsageLogger method <https://github.com/pytorch/pytorch/blob/eb3b9fe719b21fae13c7a7cf3253f970290a573e/c10/util/Logging.cpp#L114>`_.
    Please note it is triggered only once for the same API call within a process.
    It does not collect any data from open-source users since it is no-op by default.
    For more information, please refer to
    * PyTorch note: https://pytorch.org/docs/stable/notes/large_scale_deployments.html#api-usage-logging;
    * Logging policy: https://github.com/pytorch/vision/issues/5052;

    Args:
        obj (class instance or method): an object to extract info from.
    """
    module = obj.__module__
    if not module.startswith("torchvision"):
        module = f"torchvision.internal.{module}"
    name = obj.__class__.__name__
    if isinstance(obj, FunctionType):
        name = obj.__name__
    torch._C._log_api_usage_once(f"{module}.{name}")