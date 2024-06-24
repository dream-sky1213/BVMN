import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from skimage import exposure

def rgb2gray(rgb):
    r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
    gray = 0.2989*r + 0.5870*g + 0.1140*b
    gray = torch.unsqueeze(gray, 1)
    return gray

def compute_gradient(image):
    # 将图像转换为灰度图并进行对比度增强
    image = image.astype(np.float32) / 255.0  # 将像素值范围调整到 0-1
    enhanced_image = exposure.equalize_adapthist(image) * 255.0

    # 转换为 torch 张量
    enhanced_image = torch.from_numpy(enhanced_image).permute(2, 0, 1).unsqueeze(0).float().cuda()

    gray_img = rgb2gray(enhanced_image)

    sobel_kernel_x = torch.tensor([[[[-1, 0, 1],
                                    [-2, 0, 2],
                                    [-1, 0, 1]]]], dtype=torch.float32).cuda()

    sobel_kernel_y = torch.tensor([[[[-1, -2, -1],
                                    [ 0,  0,  0],
                                    [ 1,  2,  1]]]], dtype=torch.float32).cuda()

    # 使用 F.conv2d 进行卷积操作
    grad_x = F.conv2d(gray_img, sobel_kernel_x, padding=1)
    grad_y = F.conv2d(gray_img, sobel_kernel_y, padding=1)

    # 计算梯度的幅度
    grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2).squeeze()

    return grad_magnitude.cpu().numpy()

# 读取图像
image = cv2.imread('/home/server-816/Data_Hardisk/wkk/munet/data/testimg/115.jpg')

# 计算梯度信息
magnitude = compute_gradient(image)
magnitude_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# 保存梯度幅值图像到本地
cv2.imwrite('gradient_magnitude_115.png', magnitude_normalized)

# 可视化梯度幅值
plt.imshow(magnitude, cmap='gray')
plt.title('Gradient Magnitude')
plt.show()
