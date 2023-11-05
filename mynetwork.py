import torch.nn as nn
import torchvision.models as models
from resnet import resnet50
class CustomResNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet, self).__init__()

        # 创建ResNet50骨干网络
        self.resnet = resnet50(pretrained=False)

        # 去掉ResNet的原始全连接层
        self.resnet = nn.Sequential(*list(self.resnet.children()))

        # 添加自定义分类层
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Flatten(),
            nn.Linear(2048, num_classes)  # 假设你有num_classes个类别
        )

    def forward(self, x):
        # 通过ResNet的骨干网络
        x = self.resnet(x)

        # 通过自定义分类层
        x = self.classifier(x)

        return x

