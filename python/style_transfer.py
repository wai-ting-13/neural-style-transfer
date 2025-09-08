import os 
import shutil
import argparse

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim

from tqdm import tqdm

from utils.math_utils import gram_matrix
from utils.common_utils import image_to_tensor, prepare_model
from utils.const import BIG_DIM, SMALL_DIM


def style_transfer(
    # neural network
    net : nn.Module,
    # Inputs
    input_image : torch.Tensor,
    content_image : torch.Tensor,
    style_image : torch.Tensor,

    # Optimiser
    lr : float,

    # loss function
    wt_style : float,
    wt_content : float,

    # Transfering Process
    num_epochs : int,
    img_saving_freq : int,

    output_path : str
) -> None:

    # Clean Output Directory
    if os.path.exists(output_path):
        shutil.rmtree(output_path) # Deletes the directory and all its contents
    os.makedirs(output_path) # Re-creates the empty directory

    input_image.requires_grad_(True)

    opt = optim.LBFGS([input_image], lr=lr)

    epoch_style_losses = []
    epoch_content_losses = []

    for curr_epoch in tqdm(range(1, num_epochs+1)):

        input_image.data.clamp_(0, 1)

        opt.zero_grad()

        epoch_style_loss = 0
        epoch_content_loss = 0

        x = input_image
        yc = content_image.detach()
        ys = style_image.detach()

        feature_maps_x = net(x)
        with torch.no_grad():
            feature_maps_yc = net(yc)
            feature_maps_ys = net(ys)

        for i,(f_x,f_yc,f_ys) in enumerate(zip(feature_maps_x,feature_maps_yc,feature_maps_ys)):
            if i in net.style_loss_indices:
                epoch_style_loss += F.mse_loss(gram_matrix(f_x), gram_matrix(f_ys.detach()).detach())
            if i in net.content_loss_indices:
                epoch_content_loss += F.mse_loss(f_x, f_yc.detach())

        epoch_style_loss *= wt_style
        epoch_content_loss *= wt_content

        total_loss = epoch_style_loss + epoch_content_loss
        total_loss.backward()

        def closure() -> torch.Tensor:
            return total_loss

        if curr_epoch % img_saving_freq == 0:
            display_image = input_image.data.clamp_(0, 1).squeeze(0).cpu().detach()
            torchvision.utils.save_image(
                display_image,
                f"{output_path}/image_{curr_epoch}.jpg"
            )

        opt.step(closure=closure)


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    input_dir = os.path.join(data_dir, 'inputs')
    output_dir = os.path.join(data_dir, 'outputs')

    parser = argparse.ArgumentParser()

    parser.add_argument("--content_img", type=str, help="content image filename under data/input", required=True)
    parser.add_argument("--style_img", type=str, help="style image filename under data/input", required=True)

    parser.add_argument("--use_gpu", type=bool, help="use GPU", default=False)

    parser.add_argument("--style_weight", type=float, help="weight of style loss", default=1e5)
    parser.add_argument("--content_weight", type=float, help="weight of content loss", default=2)

    parser.add_argument("--learning_rate", type=float, help="learning Rate", default=0.5)

    parser.add_argument("--init_mode", type=str, choices=['random', 'content'], default='random')
    
    parser.add_argument("--num_epochs", type=int, help="number of epochs the program run", default=500)
    parser.add_argument("--saving_freq", type=int, help="saving frequency of intermediate images", default=100)

    args = parser.parse_args()

    # Choose Device
    device = torch.device("cpu")
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.use_gpu and torch.backends.mps.is_available():
        device = torch.device("mps")
    
    # Prepare Model
    vgg19_model = torchvision.models.vgg19(weights=torchvision.models.vgg.VGG19_Weights.DEFAULT)
    feature_extractor=vgg19_model.features
    style_layer_names=["relu_1", "relu_2", "relu_3", "relu_4", "relu_5"]
    content_layer_names=["relu_4"]
    use_avgpool=False
    net = prepare_model(
        device,
        feature_extractor=feature_extractor,
        style_layer_names=style_layer_names,
        content_layer_names=content_layer_names,
        use_avgpool=use_avgpool
    )

    # Get Style and Content Tensors
    image_dimension = BIG_DIM if torch.cuda.is_available() else SMALL_DIM
    style_image = image_to_tensor(f"{input_dir}/{args.style_img}", image_dimension).to(device).detach()
    content_image = image_to_tensor(f"{input_dir}/{args.content_img}", image_dimension).to(device).detach()

    # Get Input Tensor
    if args.init_mode == "content":
        # initialize as the content image
        input_image = content_image.clone().to(device)
    else:
        input_image = torch.randn(content_image.data.size(), device=device)

    # Do it
    style_transfer(
        # Neural Network
        net,

        # Inputs
        input_image,
        content_image,
        style_image,

        # Optimiser
        args.learning_rate,

        # loss function
        args.style_weight,
        args.content_weight,

        # Transfering Process
        args.num_epochs,
        args.saving_freq,

        output_dir
    )
