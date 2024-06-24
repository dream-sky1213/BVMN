import cv2
import numpy as np

# def preprocess_image(image):
#     # 降低分辨率
#     resized_image = cv2.resize(image, (640, 480))  # 将图像调整为 640x480 的分辨率

#     # 增强对比度
#     gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     enhanced_image = clahe.apply(gray_image)

#     # 降噪
#     blurred_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)
#     return np.squeeze(blurred_image)

#     # return blurred_image

# def detect_edges_ori(image):
#     # 将图像转换为灰度图
#     # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # 使用 Canny 边缘检测算法
#     edges = cv2.Canny(image, 100, 200)
#     return edges

# def detect_edges(image):
#     # 将图像转换为灰度图
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # 使用 Canny 边缘检测算法
#     edges = cv2.Canny(gray, 100, 200)
#     return edges

# def edge_difference(edge1, edge2):
#     # 计算两张边缘图像的差值
#     diff = cv2.absdiff(edge1, edge2)
#     return diff

# # 读取原图和篡改图
# original_image = cv2.imread("/home/server-816/Data_Hardisk/wkk/munet/code/utils/yt2.png")
# tampered_image = cv2.imread("/home/server-816/Data_Hardisk/wkk/munet/code/utils/zpt2.png")
# # 预处理原图
# preprocessed_original = preprocess_image(original_image)
# preprocessed_tampered= preprocess_image(tampered_image)
# # 对原图和篡改图进行边缘检测
# edges_original = detect_edges_ori(preprocessed_original)
# edges_tampered = detect_edges_ori(preprocessed_tampered)

# # 计算边缘图像的差值
# edge_diff = edge_difference(edges_original, edges_tampered)

# # 保存结果到本地
# cv2.imwrite("edges_original.png", edges_original)
# cv2.imwrite("edges_tampered.jpg", edges_tampered)
# cv2.imwrite("edge_difference.jpg", edge_diff)

# print("Images saved successfully.")

# import cv2
# import numpy as np

# def detect_edges_laplacian(image):
#     # 将图像转换为灰度图
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # 使用 Laplacian 算子进行边缘检测
#     edges = cv2.Laplacian(gray, cv2.CV_64F)
#     edges = np.uint8(np.absolute(edges))
#     return edges

# def edge_difference(edge1, edge2):
#     # 计算两张边缘图像的差值
#     diff = cv2.absdiff(edge1, edge2)
#     return diff

# # 读取原图和篡改图
# original_image = cv2.imread("/home/server-816/Data_Hardisk/wkk/munet/code/utils/yt2.png")
# tampered_image = cv2.imread("/home/server-816/Data_Hardisk/wkk/munet/code/utils/zpt2.png")

# # 对原图和篡改图进行边缘检测
# edges_original = detect_edges_laplacian(original_image)
# edges_tampered = detect_edges_laplacian(tampered_image)

# # 计算边缘图像的差值
# edge_diff = edge_difference(edges_original, edges_tampered)

# # 保存结果到本地
# cv2.imwrite("edges_original_laplacian.jpg", edges_original)
# cv2.imwrite("edges_tampered_laplacian.jpg", edges_tampered)
# cv2.imwrite("edge_difference_laplacian.jpg", edge_diff)

# print("Images saved successfully.")

#import cv2
import numpy as np

def detect_edges_sobel(image):
    # 将图像转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 使用 Sobel 算子进行边缘检测
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    # 计算梯度幅值
    edges = np.sqrt(sobelx**2 + sobely**2)
    edges = np.uint8(edges)
    return edges

def edge_difference(edge1, edge2):
    # 计算两张边缘图像的差值
    diff = cv2.absdiff(edge1, edge2)
    return diff

# 读取原图和篡改图
original_image = cv2.imread("/home/server-816/Data_Hardisk/wkk/munet/code/utils/yt2.png")
tampered_image = cv2.imread("/home/server-816/Data_Hardisk/wkk/munet/code/utils/zpt2.png")

# 对原图和篡改图进行边缘检测
edges_original = detect_edges_sobel(original_image)
edges_tampered = detect_edges_sobel(tampered_image)

# 计算边缘图像的差值
edge_diff = edge_difference(edges_original, edges_tampered)

# 保存结果到本地
cv2.imwrite("edges_original_sobel.jpg", edges_original)
cv2.imwrite("edges_tampered_sobel.jpg", edges_tampered)
cv2.imwrite("edge_difference_sobel.jpg", edge_diff)

print("Images saved successfully.")


