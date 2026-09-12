"""
W2 实验：色彩空间转换演示
功能：
  - 读取图片，展示 BGR → RGB / Gray / HSV 的转换
  - HSV 阈值分割示例（提取指定颜色）
  - 生成对比图并保存

作者：魏勺子
日期：2026-09-14
"""

import cv2
import numpy as np
import os


def demo_color_conversion(image_path: str, output_dir: str = "output"):
    """
    演示色彩空间转换
    
    Args:
        image_path: 输入图片路径
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取图片（BGR 格式）
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    
    print(f"原始图片尺寸: {img_bgr.shape}")  # (height, width, channels)
    
    # 1. BGR → RGB（用于 matplotlib 正确显示）
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 2. BGR → 灰度
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    print(f"灰度图尺寸: {img_gray.shape}")  # (height, width)
    
    # 3. BGR → HSV
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    
    # 保存结果
    cv2.imwrite(f"{output_dir}/01_original_bgr.jpg", img_bgr)
    cv2.imwrite(f"{output_dir}/02_rgb.jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{output_dir}/03_gray.jpg", img_gray)
    cv2.imwrite(f"{output_dir}/04_hsv_h.jpg", h)
    cv2.imwrite(f"{output_dir}/05_hsv_s.jpg", s)
    cv2.imwrite(f"{output_dir}/06_hsv_v.jpg", v)
    
    print(f"色彩空间转换结果已保存到 {output_dir}/")
    
    return img_bgr, img_hsv


def demo_hsv_threshold(img_bgr: np.ndarray, img_hsv: np.ndarray, 
                       color_name: str = "blue",
                       output_dir: str = "output"):
    """
    演示 HSV 阈值分割
    
    Args:
        img_bgr: BGR 格式原图
        img_hsv: HSV 格式图像
        color_name: 要提取的颜色名称
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 预定义颜色范围
    color_ranges = {
        "red":    ([0, 50, 50], [10, 255, 255], [170, 50, 50], [179, 255, 255]),
        "orange": ([11, 50, 50], [25, 255, 255], None, None),
        "yellow": ([26, 50, 50], [34, 255, 255], None, None),
        "green":  ([35, 50, 50], [85, 255, 255], None, None),
        "blue":   ([100, 50, 50], [130, 255, 255], None, None),
        "purple": ([131, 50, 50], [160, 255, 255], None, None),
    }
    
    if color_name not in color_ranges:
        print(f"未知颜色 '{color_name}'，使用蓝色作为默认")
        color_name = "blue"
    
    ranges = color_ranges[color_name]
    
    # 创建掩码
    if ranges[2] is not None:  # 红色需要两个范围
        lower1, upper1 = np.array(ranges[0]), np.array(ranges[1])
        lower2, upper2 = np.array(ranges[2]), np.array(ranges[3])
        mask1 = cv2.inRange(img_hsv, lower1, upper1)
        mask2 = cv2.inRange(img_hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        lower, upper = np.array(ranges[0]), np.array(ranges[1])
        mask = cv2.inRange(img_hsv, lower, upper)
    
    # 应用掩码
    result = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
    
    # 保存结果
    cv2.imwrite(f"{output_dir}/07_mask_{color_name}.jpg", mask)
    cv2.imwrite(f"{output_dir}/08_result_{color_name}.jpg", result)
    
    # 统计
    total_pixels = mask.size
    selected_pixels = np.count_nonzero(mask)
    ratio = selected_pixels / total_pixels * 100
    
    print(f"\nHSV 阈值分割结果 ({color_name}):")
    print(f"  选中像素: {selected_pixels:,} / {total_pixels:,} ({ratio:.1f}%)")
    print(f"  结果已保存到 {output_dir}/")
    
    return mask, result


def create_comparison_grid(image_path: str, output_path: str = "output/comparison.jpg"):
    """
    创建色彩空间对比图（2x3 网格）
    
    Args:
        image_path: 输入图片路径
        output_path: 输出图片路径
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    
    # 转换各种色彩空间
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    
    # 统一为 3 通道以便拼接
    img_gray_3ch = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    h_3ch = cv2.applyColorMap(h, cv2.COLORMAP_HSV)  # H 用彩色显示
    s_3ch = cv2.cvtColor(s, cv2.COLOR_GRAY2BGR)
    v_3ch = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)
    
    # 添加标签
    def add_label(img, text):
        img_copy = img.copy()
        cv2.putText(img_copy, text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return img_copy
    
    # 拼接 2x3 网格
    row1 = np.hstack([
        add_label(img_bgr, "BGR (OpenCV)"),
        add_label(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), "RGB"),
        add_label(img_gray_3ch, "Grayscale")
    ])
    row2 = np.hstack([
        add_label(h_3ch, "HSV - Hue"),
        add_label(s_3ch, "HSV - Saturation"),
        add_label(v_3ch, "HSV - Value")
    ])
    
    grid = np.vstack([row1, row2])
    
    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, grid)
    print(f"\n对比图已保存: {output_path}")


def main():
    """主函数：演示色彩空间操作"""
    
    # 使用 W1 生成的 hello.png 作为示例图片
    # 如果没有，可以替换为任意图片路径
    sample_image = "../week01/hello.png"
    
    if not os.path.exists(sample_image):
        print(f"示例图片不存在: {sample_image}")
        print("请修改 sample_image 变量为实际图片路径")
        return
    
    print("=" * 50)
    print("色彩空间转换演示")
    print("=" * 50)
    
    # 1. 基础转换
    img_bgr, img_hsv = demo_color_conversion(sample_image)
    
    # 2. HSV 阈值分割（尝试提取蓝色）
    print("\n" + "-" * 50)
    demo_hsv_threshold(img_bgr, img_hsv, color_name="blue")
    
    # 3. 生成对比图
    print("\n" + "-" * 50)
    create_comparison_grid(sample_image)
    
    print("\n" + "=" * 50)
    print("演示完成！查看 output/ 目录下的结果")
    print("=" * 50)


if __name__ == "__main__":
    main()
