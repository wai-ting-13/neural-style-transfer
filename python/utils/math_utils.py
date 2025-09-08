import torch

# Define gram matrix
def gram_matrix(ip : torch.Tensor) -> torch.Tensor:
    num_batch, num_channels, height, width = ip.size()
    feats = ip.view(num_batch * num_channels, width * height)
    gram_mat = torch.mm(feats, feats.t())
    return gram_mat.div(num_batch * num_channels * width * height)