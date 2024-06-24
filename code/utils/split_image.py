from PIL import Image, ImageDraw, ImageFont
import math
import os

def split_image(image_path, rows, cols, output_folder, font_size=24):
    # 打开图像
    image = Image.open(image_path)
    # 获取图像尺寸
    width, height = image.size
    
    # 计算每个小图像的尺寸
    small_width = math.ceil(width / cols)
    small_height = math.ceil(height / rows)
    
    # 创建一个用于减弱背景的遮罩，通过降低亮度来减弱背景
    mask = Image.new('L', image.size, color=128)  # 灰度遮罩，128为半透明
    faded_image = Image.composite(image, Image.new('RGB', image.size), mask)
    
    # 设置字体大小
    font = ImageFont.truetype("arial.ttf", font_size)  # 你可以根据需要调整字体大小
    
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 拆分图像并保存
    number = 1
    for row in range(rows):
        for col in range(cols):
            # 计算小图像的边界
            left = col * small_width
            upper = row * small_height
            right = min((col + 1) * small_width, width)
            lower = min((row + 1) * small_height, height)
            
            # 剪切小图像
            box = (left, upper, right, lower)
            small_image = faded_image.crop(box)
            
            # 在小图像中央打印序号
            draw = ImageDraw.Draw(small_image)
            text = str(number)
            text_width, text_height = draw.textsize(text, font=font)
            # 设置文字位置
            text_x = (small_width - text_width) // 2
            text_y = (small_height - text_height) // 2
            # 在图像中央打印序号
            draw.text((text_x, text_y), text, font=font, fill='white')
            
            # 保存小图像
            small_image.save(os.path.join(output_folder, f'image_{number}.png'))
            number += 1

# 使用示例
split_image('/home/server-816/Data_Hardisk/wkk/munet/code/splitImage/image.jpg', 3, 3, '/home/server-816/Data_Hardisk/wkk/munet/code/splitResult', font_size=30)