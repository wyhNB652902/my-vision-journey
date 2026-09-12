# W2 学习笔记：OpenCV 基础 I/O + 色彩空间

> 时间：9/14–9/20
> 目标：摄像头实时滤镜 demo + 3B1B 线代本质 E1–E6

---

## 一、OpenCV 基础 I/O

### 1.1 图像读写

```python
import cv2
import numpy as np

# 读取图像
img = cv2.imread('path/to/image.jpg')          # BGR 格式（注意不是 RGB）
img_gray = cv2.imread('path/to/image.jpg', cv2.IMREAD_GRAYSCALE)

# 显示图像
cv2.imshow('window_name', img)
cv2.waitKey(0)          # 等待按键，0 表示无限等待
cv2.destroyAllWindows()

# 保存图像
cv2.imwrite('output.jpg', img)
```

**关键参数：**
- `cv2.IMREAD_COLOR`（默认）：BGR 三通道
- `cv2.IMREAD_GRAYSCALE`：单通道灰度
- `cv2.IMREAD_UNCHANGED`：包含 alpha 通道

### 1.2 视频流读取

```python
cap = cv2.VideoCapture(0)   # 0 表示默认摄像头

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

**关键方法：**
- `cap.isOpened()`：检查摄像头是否打开
- `cap.get(cv2.CAP_PROP_FRAME_WIDTH)`：获取宽度
- `cap.set(cv2.CAP_PROP_FPS, 30)`：设置帧率

### 1.3 绘图 API

```python
# 直线
cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

# 矩形
cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)

# 圆形
cv2.circle(img, (center_x, center_y), radius, (0, 0, 255), thickness=-1)  # -1 填充

# 文字
cv2.putText(img, 'Hello', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=1, color=(255, 255, 255), thickness=2)

# 多边形
pts = np.array([[x1, y1], [x2, y2], [x3, y3]], np.int32)
cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
```

---

## 二、色彩空间

### 2.1 RGB vs BGR

OpenCV 使用 **BGR** 顺序（历史原因，早期摄像头厂商标准）。

```python
# BGR → RGB（用于 matplotlib 显示）
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# BGR → 灰度
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
```

### 2.2 HSV 色彩空间

**HSV = Hue（色相）+ Saturation（饱和度）+ Value（明度）**

| 通道 | 范围 | 含义 |
|------|------|------|
| H | 0–179 | 颜色种类（红=0, 绿=60, 蓝=120）|
| S | 0–255 | 颜色纯度（0=灰色，255=纯色）|
| V | 0–255 | 亮度（0=黑色，255=最亮）|

```python
# BGR → HSV
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

# HSV 阈值分割（示例：提取蓝色）
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([130, 255, 255])
mask = cv2.inRange(img_hsv, lower_blue, upper_blue)
result = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
```

**为什么用 HSV？**
- RGB 对光照变化敏感，HSV 更稳定
- 颜色分割更直观（直接指定色相范围）

### 2.3 常用颜色范围参考

| 颜色 | H 范围 | S 范围 | V 范围 |
|------|--------|--------|--------|
| 红色 | 0–10, 170–179 | 50–255 | 50–255 |
| 橙色 | 11–25 | 50–255 | 50–255 |
| 黄色 | 26–34 | 50–255 | 50–255 |
| 绿色 | 35–85 | 50–255 | 50–255 |
| 蓝色 | 100–130 | 50–255 | 50–255 |
| 紫色 | 131–160 | 50–255 | 50–255 |

---

## 三、本周实战：摄像头实时滤镜

### 目标
- 打开摄像头，实时显示画面
- 按键切换滤镜：原图 → 灰度 → HSV 通道分离 → 边缘检测
- 按 'q' 退出，按 's' 保存当前帧

### 代码框架

见 `week02/camera_filter.py`

---

## 四、3B1B 线代本质 E1–E6 学习 Checklist

| 集数 | 主题 | 核心概念 | 自测问题 |
|------|------|----------|----------|
| E1 | 向量是什么 | 向量 = 空间中的箭头 / 有序数组 | 向量的几何意义和代数意义分别是什么？ |
| E2 | 线性组合、张成、基 | 线性组合、张成空间、线性相关/无关 | 如何判断一组向量是否线性无关？ |
| E3 | 矩阵与线性变换 | 矩阵 = 线性变换的数值表示 | 矩阵乘法的几何意义是什么？ |
| E4 | 矩阵乘法与复合变换 | 矩阵乘法 = 变换的复合 | 为什么矩阵乘法不满足交换律？ |
| E5 | 行列式 | 行列式 = 面积/体积的缩放因子 | 行列式为 0 意味着什么？ |
| E6 | 逆矩阵、秩、零空间 | 逆矩阵、列空间、秩、零空间 | 什么情况下矩阵没有逆？ |

**学习建议：**
- 每集 15 分钟左右，看完立刻用自己的话复述核心概念
- 把「几何直觉」和「代数计算」对应起来
- 遇到不懂的暂停，在纸上画一画

---

## 五、本周产出清单

- [ ] `week02/camera_filter.py`：摄像头实时滤镜 demo
- [ ] `week02/color_space_demo.py`：色彩空间转换实验（可选）
- [ ] 3B1B E1–E6 学习笔记（本文档勾选）
- [ ] 每日刷题 ≥1 题（Codeforces / LeetCode）

---

*下周预告：滤波与边缘检测（高斯/中值滤波、Sobel、Canny）*
