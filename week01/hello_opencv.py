"""week01/hello_opencv.py — 环境验证 + 第一个 OpenCV 程序

运行方式（Anaconda Prompt 或 VSCode 终端）:
    conda activate vision
    python week01/hello_opencv.py

按 q 退出摄像头窗口。
"""
import cv2
import numpy as np


def main():
    print(f"OpenCV 版本: {cv2.__version__}")
    print(f"NumPy 版本: {np.__version__}")

    # 1. 画一张图，验证图像基础操作
    img = np.zeros((240, 480, 3), dtype=np.uint8)
    cv2.putText(img, "Hello, my-vision-journey!", (40, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(img, "press q to quit", (40, 190),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 1)
    cv2.imwrite("week01/hello.png", img)
    print("已保存 week01/hello.png")

    # 2. 打开摄像头，实时显示 + 边缘检测
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("摄像头打开失败（不影响环境验证，跳过）")
        return

    print("摄像头已打开，窗口中按 q 退出...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        cv2.imshow("camera", frame)
        cv2.imshow("edges (Canny)", edges)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("环境验证完成 ✔")


if __name__ == "__main__":
    main()
