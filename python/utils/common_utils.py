from PIL import Image

import torch
import torch.nn as nn
import torchvision

from models.nst import NSTNetwork

from utils.const import SMALL_DIM

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

def image_to_tensor(image_filepath : str, image_dimension : int = SMALL_DIM) -> torch.Tensor:
    img = Image.open(image_filepath).convert('RGB')

    # Central-crop the image if it is not square
    if img.height != img.width:
        width, height = img.size
        min_dim = min(width, height)
        left = (width - min_dim) / 2
        top = (height - min_dim) / 2
        right = (width + min_dim) / 2
        bottom = (height + min_dim) / 2
        box = (left, top, right, bottom)
        img = img.crop(box)

    # Scale-up image if it is too small
    if img.height < image_dimension or img.width < image_dimension:
      scaling_factor = image_dimension / max(img.size)

      new_width = int(img.width * scaling_factor)
      new_height = int(img.height * scaling_factor)

      img = img.resize((new_width, new_height), Image.LANCZOS)

    torch_transformation = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_dimension),
        torchvision.transforms.ToTensor()
    ])

    img = torch_transformation(img).unsqueeze(0)

    return img.to(torch.float)