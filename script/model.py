
import torch
import torch.nn as nn
import torchvision.models as models

try:
    from torch.func import functional_call
except ImportError:  # pragma: no cover
    from torch.nn.utils.stateless import functional_call


class FeatureExtracter(nn.Module):
    '''
    Extract features Z from X
    '''

    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        # Load ResNet18
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None

        resnet = models.resnet18(weights=weights)

        # Extract ResNet18 up to layer 1, output channels = 64
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1

        # self.out_channels = 64

    def forward(self, x: torch.Tensor):
        '''
        Args:
            x: Input images [B, 3, H, W]

        Returns:
            features: Intermediate features Z [B, 64, H/4, W/4]
        '''
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)

        return x


class Classifier(nn.Module):
    '''
    Predict Y from Z
    '''
    def __init__(self, num_classes: int = 8, pretrained: bool = True):
        super().__init__()

        # Load ResNet18
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None

        resnet = models.resnet18(weights=weights)

        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        
        # Fully connected layer
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor):
        '''
        Args:
            x: Input features [B, 64, H/4, W/4]

        Returns:
            logits: Class predictions [B, num_classes]
        '''
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def functional_forward(self, x: torch.Tensor, params):
        params_and_buffers = dict(self.named_buffers())
        params_and_buffers.update(params)
        return functional_call(self, params_and_buffers, (x,))
        

class CICFModel(nn.Module):
    def __init__(self, num_classes: int = 8, pretrained: bool = True):
        super().__init__()

        self.h = FeatureExtracter(pretrained=pretrained)  # 1st Stage
        self.f = Classifier(num_classes=num_classes)  # 2nd Stage

        # self.f_params = list(self.f.parameters())  # store for virtual update

    def forward(self, x: torch.Tensor):
        '''
        Standard forward pass: X -> Z -> Y

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            logits: Class predictions [B, num_classes]
        '''
        z = self.h(x)
        return self.f(z)

    def forward_with_f_params(self, x: torch.Tensor, f_params):
        z = self.h(x)
        return self.f.functional_forward(z, f_params)

    def pooled_features(self, x: torch.Tensor):
        z = self.h(x)
        return z.mean(dim=[2, 3])
    
    def extract_features(self, x: torch.Tensor):
        with torch.no_grad():
            features = self.pooled_features(x)

        return features

        
