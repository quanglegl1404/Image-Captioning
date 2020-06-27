import torch.nn as nn
import torchvision.models as models
import torch

#####################
# Encoder RESNET CNN
#####################
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        #resnet = models.resnet152(pretrained=True)
        resnet = models.resnet101(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))

    def forward(self, images):
        #print(f'Images: {self.resnet(images).shape}')
        out = self.adaptive_pool(self.resnet(images))
        #print(f'out size: {out.shape}')
        # batch_size, img size, imgs size, 2048
        out = out.permute(0, 2, 3, 1)
       # print(f'out size after permute: {out.shape}')
        return out