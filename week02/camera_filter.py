"""
W2 实战：摄像头实时滤镜
功能：
  - 打开摄像头，实时显示画面
  - 按键切换滤镜：原图 → 灰度 → HSV 通道分离 → 边缘检测
  - 按 'q' 退出，按 's' 保存当前帧

作者：魏勺子
日期：2026-09-14
"""

import cv2
import numpy as np
import os
from datetime import datetime


class CameraFilter:
    """摄像头实时滤镜类"""
    
    # 滤镜模式枚举
    MODE_ORIGINAL = 0
    MODE_GRAY = 1
    MODE_HSV = 2
    MODE_EDGE = 3
    MODE_COUNT = 4
    
    MODE_NAMES = {
        MODE_ORIGINAL: "Original",
        MODE_GRAY: "Grayscale",
        MODE_HSV: "HSV Channels",
        MODE_EDGE: "Edge Detection"
    }
    
    def __init__(self, camera_id=0, width=640, height=480):
        """
        初始化摄像头
        
        Args:
            camera_id: 摄像头设备号，默认 0
            width: 画面宽度
            height: 画面高度
        """
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        if not self.cap.isOpened():
            raise RuntimeError("无法打开摄像头，请检查设备连接")
        
        self.mode = self.MODE_ORIGINAL
        self.save_dir = "captures"
        os.makedirs(self.save_dir, exist_ok=True)
        
        print("=" * 50)
        print("摄像头实时滤镜 Demo")
        print("=" * 50)
        print("按键说明：")
        print("  [空格] 或 [m] : 切换滤镜模式")
        print("  [s]           : 保存当前帧")
        print("  [q] 或 [ESC]  : 退出程序")
        print("=" * 50)
        print(f"当前模式: {self.MODE_NAMES[self.mode]}")
    
    def apply_filter(self, frame):
        """
        对输入帧应用当前滤镜
        
        Args:
            frame: BGR 格式的输入图像
            
        Returns:
            处理后的图像（用于显示）
        """
        if self.mode == self.MODE_ORIGINAL:
            return frame.copy()
        
        elif self.mode == self.MODE_GRAY:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 转回 3 通道以便统一显示
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        elif self.mode == self.MODE_HSV:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            # 创建 3 通道显示：H 用彩色显示，S 和 V 用灰度
            h_color = cv2.applyColorMap(h, cv2.COLORMAP_HSV)
            s_bgr = cv2.cvtColor(s, cv2.COLOR_GRAY2BGR)
            v_bgr = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)
            
            # 水平拼接：H | S | V
            top_row = np.hstack([h_color, s_bgr])
            bottom_row = np.hstack([v_bgr, np.zeros_like(v_bgr)])
            combined = np.vstack([top_row, bottom_row])
            
            # 缩放到原始尺寸
            return cv2.resize(combined, (frame.shape[1], frame.shape[0]))
        
        elif self.mode == self.MODE_EDGE:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        return frame.copy()
    
    def save_frame(self, frame):
        """保存当前帧到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_name = self.MODE_NAMES[self.mode].lower().replace(" ", "_")
        filename = f"{self.save_dir}/capture_{timestamp}_{mode_name}.jpg"
        cv2.imwrite(filename, frame)
        print(f"已保存: {filename}")
    
    def run(self):
        """主循环"""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("无法读取画面，退出...")
                break
            
            # 应用滤镜
            display = self.apply_filter(frame)
            
            # 在画面上显示当前模式
            mode_text = f"Mode: {self.MODE_NAMES[self.mode]}"
            cv2.putText(display, mode_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 显示画面
            cv2.imshow("Camera Filter", display)
            
            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # q 或 ESC
                break
            elif key == ord(' ') or key == ord('m'):  # 空格或 m 切换模式
                self.mode = (self.mode + 1) % self.MODE_COUNT
                print(f"切换到模式: {self.MODE_NAMES[self.mode]}")
            elif key == ord('s'):  # s 保存
                self.save_frame(display)
        
        # 释放资源
        self.cap.release()
        cv2.destroyAllWindows()
        print("程序已退出")


def main():
    """入口函数"""
    try:
        app = CameraFilter(camera_id=0, width=640, height=480)
        app.run()
    except RuntimeError as e:
        print(f"错误: {e}")
    except KeyboardInterrupt:
        print("\n用户中断")


if __name__ == "__main__":
    main()
