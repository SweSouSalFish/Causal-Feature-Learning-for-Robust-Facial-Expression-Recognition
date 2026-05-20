
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
    

if __name__ == '__main__':
    print("=" * 50)
    print("测试 CICFModel")
    print("=" * 50)
    
    # 创建模型
    model = CICFModel(num_classes=8, pretrained=True)
    print(f"\n✓ 模型创建成功")
    
    # 测试输入
    x = torch.randn(4, 3, 224, 224)
    print(f"  输入形状: {x.shape}")
    
    # 1. 测试前向传播
    logits = model(x)
    print(f"\n✓ 前向传播成功")
    print(f"  输出形状: {logits.shape}")  # 应该是 [4, 8]
    
    # 2. 测试 h 的输出
    z = model.h(x)
    print(f"\n✓ h 输出形状: {z.shape}")  # 应该是 [4, 64, 56, 56]
    
    # 3. 测试特征提取
    feats = model.extract_features(x)
    print(f"\n✓ 特征提取成功")
    print(f"  特征形状: {feats.shape}")  # 应该是 [4, 64]
    
    # 4. 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    h_params = sum(p.numel() for p in model.h.parameters())
    f_params = sum(p.numel() for p in model.f.parameters())
    
    print(f"\n✓ 参数量统计:")
    print(f"  h 参数: {h_params:,}")
    print(f"  f 参数: {f_params:,}")
    print(f"  总参数: {total_params:,}")
    
    # 5. 测试虚拟更新（验证 f 的参数可以被独立修改）
    print(f"\n✓ 测试虚拟更新:")
    original_weight = model.f.fc.weight.data.clone()  # fc 层的权重
    
    # 模拟 g_dagger
    fake_grad = torch.randn_like(original_weight) * 0.01
    with torch.no_grad():
        model.f.fc.weight.data -= 0.1 * fake_grad
    
    weight_changed = not torch.allclose(original_weight, model.f.fc.weight.data)
    print(f"  fc 层权重已改变: {weight_changed}")
    
    print("\n" + "=" * 50)
    print("所有测试通过！")
        
