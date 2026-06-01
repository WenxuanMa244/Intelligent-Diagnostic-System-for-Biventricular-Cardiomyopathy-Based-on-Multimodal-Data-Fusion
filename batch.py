import cv2
import numpy as np
import os

def adjust_brightness_contrast(img, brightness=1.0, contrast=1.0):
    """
    自适应调整图像的亮度和对比度
    :param img: 输入图像
    :param brightness: 亮度调整因子（1.0 表示不变）
    :param contrast: 对比度调整因子（1.0 表示不变）
    :return: 调整后的图像
    """
    # 应用亮度和对比度调整
    img = np.int16(img)
    img = img * contrast + brightness
    img = np.clip(img, 0, 255)
    return np.uint8(img)

def auto_detect_purple_range(hsv_img):
    """
    自动检测图像中紫色的范围
    :param hsv_img: HSV 颜色空间的图像
    :return: lower_purple, upper_purple
    """
    # 提取 H 通道（色调）
    h_channel = hsv_img[:, :, 0]

    # 统计紫色区域的色调值（紫色在 HSV 中的范围通常是 120-160）
    purple_hues = h_channel[(h_channel >= 120) & (h_channel <= 160)]

    if len(purple_hues) == 0:
        # 如果没有检测到紫色，使用默认范围
        return np.array([120, 50, 50]), np.array([160, 255, 255])

    # 计算紫色区域的色调均值和标准差
    hue_mean = np.mean(purple_hues)
    hue_std = np.std(purple_hues)

    # 动态调整紫色范围
    lower_purple = np.array([max(120, hue_mean - 2 * hue_std), 50, 50])
    upper_purple = np.array([min(160, hue_mean + 2 * hue_std), 255, 255])

    return lower_purple, upper_purple

def process_image(image_path, output_folder):
    """
    处理单张图像
    :param image_path: 图像路径
    :param output_folder: 输出文件夹路径
    """
    # 1. 加载图像
    img = cv2.imread(image_path)

    # 检查图像是否加载成功
    if img is None:
        print(f"Error: 图像加载失败，请检查路径是否正确: {image_path}")
        return

    # 2. 自适应调整亮度和对比度
    adjusted_img = adjust_brightness_contrast(img, brightness=30, contrast=1.2)

    # 3. 转换到 HSV 颜色空间
    hsv_img = cv2.cvtColor(adjusted_img, cv2.COLOR_BGR2HSV)

    # 4. 自动检测紫色范围
    lower_purple, upper_purple = auto_detect_purple_range(hsv_img)
    print(f"图像: {os.path.basename(image_path)}, 自动检测的紫色范围: lower={lower_purple}, upper={upper_purple}")

    # 5. 创建掩膜
    mask = cv2.inRange(hsv_img, lower_purple, upper_purple)

    # 6. 检查掩膜中是否存在紫色
    if np.any(mask == 255):
        print(f"图像: {os.path.basename(image_path)}, 存在紫色。")
    else:
        print(f"图像: {os.path.basename(image_path)}, 不存在紫色。")
        return  # 如果没有紫色，直接跳过

    # 7. 提取最大连通区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # 如果没有检测到连通区域（只有背景）
    if num_labels == 1:
        print(f"图像: {os.path.basename(image_path)}, 未检测到任何连通区域。")
        return

    # 找到最大的连通区域（忽略背景，背景标签为0）
    max_label = 1  # 初始化为第一个连通区域
    max_area = stats[max_label, cv2.CC_STAT_AREA]  # 初始化为第一个连通区域的面积

    for label in range(2, num_labels):  # 从第二个连通区域开始遍历
        area = stats[label, cv2.CC_STAT_AREA]
        if area > max_area:
            max_area = area
            max_label = label

    # 创建只包含最大连通区域的掩膜
    max_region_mask = (labels == max_label).astype(np.uint8) * 255

    # 8. 填充最大连通区域
    # 根据区域大小动态调整卷积核大小
    kernel_size = max(3, int(np.sqrt(max_area) / 10))  # 动态调整卷积核大小
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    filled_mask = cv2.morphologyEx(max_region_mask, cv2.MORPH_CLOSE, kernel)

    # 9. 保存填充后的掩膜
    output_path = os.path.join(output_folder, f"mask_{os.path.basename(image_path)}")
    cv2.imwrite(output_path, filled_mask)
    print(f"图像: {os.path.basename(image_path)}, 填充后的最大连通区域掩膜已保存到：{output_path}")

def process_folder(input_folder, output_folder):
    """
    处理文件夹中的所有图像
    :param input_folder: 输入文件夹路径
    :param output_folder: 输出文件夹路径
    """
    # 检查输出文件夹是否存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        # 检查文件是否为图像（支持常见格式）
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(input_folder, filename)
            process_image(image_path, output_folder)

if __name__ == "__main__":
    # 输入文件夹路径
    input_folder = r"C:\Users\32572\Desktop\dataset\test_lable\test_lable_5"
    # 输出文件夹路径
    output_folder = r"C:\Users\32572\Desktop\mask\test_lable_mask_5"

    # 处理文件夹中的所有图像
    process_folder(input_folder, output_folder)