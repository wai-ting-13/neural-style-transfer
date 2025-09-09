from PIL import Image

from typing import Tuple, Union

import torch
import torch.nn as nn
import torchvision

from models.nst import NSTNetwork

def prepare_model(
    device,
    feature_extractor : nn.Module,
    style_layer_names: list[str],
    content_layer_names: list[str],
    use_avgpool : bool = False
) -> nn.Module:
    # Define Our Model
    net = NSTNetwork(
        feature_extractor=feature_extractor,
        style_layer_names=style_layer_names,
        content_layer_names=content_layer_names,
        use_avgpool=use_avgpool
    )

    # Disable Gradient and Turn Model to Evaluation Model
    net.requires_grad_(False)
    net.eval()
    net.to(device)

    return net

def images_to_tensor(
    content_image_filepath : str, 
    style_image_filepath : str,
    max_img_size : Tuple[int, int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    content_img = Image.open(content_image_filepath).convert('RGB')
    style_img = Image.open(style_image_filepath).convert('RGB')

    # Find desire size for images (scale-down)
    (max_width, max_height) = max_img_size
    (new_width, new_height) = content_img.size

    scaling_factor = 1

    if content_img.width > max_width and content_img.height <= max_height:
        scaling_factor = max_width / content_img.width
    elif content_img.height > max_height and content_img.width <= max_width:
        scaling_factor = max_height / content_img.height
    elif content_img.height > max_height and content_img.width > max_width:
        scaling_factor = min(
          max_width / content_img.width,
          max_height / content_img.height
        )
    
    # Scale-down the images
    (new_width, new_height) = (
        int(content_img.width * scaling_factor), 
        int(content_img.height * scaling_factor)
    )
    content_img = content_img.resize((new_width, new_height), Image.LANCZOS)
    style_img = style_img.resize((new_width, new_height), Image.LANCZOS)

    torch_transformation = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    content_img = torch_transformation(content_img).unsqueeze(0)
    style_img = torch_transformation(style_img).unsqueeze(0)

    return (content_img.to(torch.float), style_img.to(torch.float))