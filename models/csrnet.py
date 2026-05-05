import torch
import torch.nn as nn
import torchvision.models as models


class CSRNet(nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()
        self.frontend_feat = [64, 64, "M", 128, 128, "M",
                              256, 256, 256, "M", 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.frontend = self.make_layers(self.frontend_feat)
        self.backend = self.make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        if load_pretrained:
            try:
                vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
                pretrained_dict = vgg.features.state_dict()
                frontend_dict = self.frontend.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items()
                                   if k in frontend_dict and v.size() == frontend_dict[k].size()}
                frontend_dict.update(pretrained_dict)
                self.frontend.load_state_dict(frontend_dict)
            except Exception as e:
                print(f"Couldn't load VGG16: {e}")
        self._initialize_weights()

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return self.output_layer(x)

    def make_layers(self, cfg, in_channels=3, batch_norm=False, dilation=False):
        layers = []
        d_rate = 2 if dilation else 1
        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3,
                                   padding=d_rate, dilation=d_rate)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.backend.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.output_layer.weight, std=0.01)
        if self.output_layer.bias is not None:
            nn.init.constant_(self.output_layer.bias, 0)
