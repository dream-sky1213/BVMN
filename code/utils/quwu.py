import cv2
import numpy as np

def dark_channel(image, window_size):
    # 计算每个像素点在指定窗口大小内的最小通道值（暗通道）
    return np.min(image, axis=2)

def estimate_atmospheric_light(image, dark_channel_image, percentile):
    # 根据暗通道图像中的像素点，估计全球大气光
    flat_dark_channel = dark_channel_image.flatten()
    num_pixels = flat_dark_channel.shape[0]
    num_top_pixels = int(num_pixels * (1.0 - percentile))
    indices = np.argpartition(flat_dark_channel, -num_top_pixels)[-num_top_pixels:]
    atmospheric_light = np.max(image.reshape(-1, 3)[indices], axis=0)
    return atmospheric_light

def estimate_transmission(dark_channel_image, atmospheric_light, omega, window_size):
    # 扩展大气光数组，使其与暗通道图像具有相同的形状
    atmospheric_light_extended = np.tile(atmospheric_light, (dark_channel_image.shape[0], dark_channel_image.shape[1], 1))
    
    # 分别计算每个通道的透射率
    transmission_r = 1.0 - omega * dark_channel_image / atmospheric_light_extended[:,:,0]
    transmission_g = 1.0 - omega * dark_channel_image / atmospheric_light_extended[:,:,1]
    transmission_b = 1.0 - omega * dark_channel_image / atmospheric_light_extended[:,:,2]
    
    # 对结果取平均
    transmission_avg = (transmission_r + transmission_g + transmission_b) / 3.0
    
    return transmission_avg



def refine_transmission(image, transmission_estimate, window_size):
    # 通过导向滤波器细化透射率
    guided_filter = cv2.ximgproc.createGuidedFilter(image.astype(np.float32), window_size, eps=0.001)
    return guided_filter.filter(transmission_estimate)

def recover_scene_radiance(image, atmospheric_light, transmission, t0=0.1):
    # 根据透射率和大气光恢复场景辐射
    return np.clip((image - atmospheric_light) / np.maximum(transmission, t0) + atmospheric_light, 0, 255)

def dehaze(image, omega=0.95, window_size=15, percentile=0.001):
    # 图像去雾
    dark_channel_image = dark_channel(image, window_size)
    atmospheric_light = estimate_atmospheric_light(image, dark_channel_image, percentile)
    transmission_estimate_val = estimate_transmission(dark_channel_image, atmospheric_light, omega, window_size)
    refined_transmission = refine_transmission(image, transmission_estimate_val, window_size)
    recovered_image = recover_scene_radiance(image, atmospheric_light, refined_transmission)
    return recovered_image.astype(np.uint8)


# 读取雾霾图像
hazy_image = cv2.imread("/home/server-816/Data_Hardisk/wkk/munet/code/utils/yt2.png")

# 对雾霾图像进行去雾
dehazed_image = dehaze(hazy_image)

# 保存去雾后的图像
cv2.imwrite("dehazed_image.jpg", dehazed_image)

print("Dehazing complete. Dehazed image saved successfully.")
