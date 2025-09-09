import torch
import torch.nn as nn
import torchvision

"""Model Definition"""
class NSTNetwork(nn.Module):
    def __init__(
        self,
        feature_extractor : nn.Module,
        style_layer_names : list[str],
        content_layer_names : list[str],
        use_avgpool : bool = False
    ):
        super().__init__()

        # Get Indices
        self.style_loss_indices = [i for i, _ in enumerate(style_layer_names)]
        self.content_loss_indices = [i for i, name in enumerate(style_layer_names) if name in content_layer_names]

        # Define Normalisation Function
        self.normalise = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        slices : list[nn.Sequencial] = []
        slice = nn.Sequential()

        i = 0;
        for layer in feature_extractor.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
                layer = nn.AvgPool2d(layer.kernel_size, layer.stride, layer.padding) if use_avgpool else layer
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            slice.add_module(name, layer)
            
            if name in style_layer_names:
                slices.append(slice)
                slice = nn.Sequential()

        self.extractor = nn.Sequential()
        for i, slice in enumerate(slices,1):
            self.extractor.add_module(f"slice_{i}", slice)

    def forward(self, x) -> list[torch.Tensor]:
        x = self.normalise(x)
        feature_maps : list[torch.Tensor] = []
        for slice in self.extractor.children():
            x = slice(x)
            feature_maps.append(x)
        return feature_maps