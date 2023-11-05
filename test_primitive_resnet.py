import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
from PIL import Image
import  numpy as np
import torch
from mynetwork import CustomResNet
# 读取图像
image = Image.open('image/1767_c6_f0136491.jpg')  # 替换为你的图像文件路径
rgb_img=np.array(image)
model =resnet50(False)
target_layers = [model.layer4[-1]]
input_tensor =torch.from_numpy(rgb_img).unsqueeze(0).permute(0, 3, 1, 2)
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

targets = [ClassifierOutputTarget(1)]

# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam = cam(input_tensor=input_tensor.to(torch.float), targets=targets)

# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(rgb_img.astype(np.float32)/255.0, grayscale_cam, use_rgb=True)
plt.imshow(visualization)
plt.show()