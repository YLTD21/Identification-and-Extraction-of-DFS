import os
import time
import math
import matplotlib.pyplot as plt
from PIL import Image
from tkinter import messagebox
import tkinter as tk

def onclick(event):
    # 获取点击位置的坐标
    global x, y
    x = event.xdata
    y = event.ydata
    print('x = %d, y = %d' % (x, y))

    # 在点击位置生成一个红色标记点，设置标记点的大小为100
    plt.scatter(x, y, color='red', s=100)

    # 重新绘制图形以显示更新后的标记点
    plt.draw()

    # 弹出对话框，确认是否选择该点
    root = tk.Tk()
    root.withdraw()
    result = messagebox.askyesno("Confirmation", "Do you want to confirm this point as the vertex?")
    root.destroy()

    if result:
        # 绘制连接线和标注
        for center_x, center_y in centers:
            plt.plot([x, center_x], [y, center_y], 'r--', linewidth=0.5)
            distance = math.sqrt((center_x - x) ** 2 + (center_y - y) ** 2)
            plt.text((x + center_x) / 2, (y + center_y) / 2, f"{distance:.2f}", color='black', fontsize=8, ha='center', va='center')

        plt.draw()

        return x, y
    else:
        return None, None

def crop_image(image_path, size, click_position):  # 这里的size是切割大小
    # 创建输出文件夹
    timestamp = time.strftime("%Y%m%d%H%M%S")
    folder_name = os.path.basename(image_path) + "_" + timestamp
    output_folder_path = os.path.join("D://", folder_name)
    os.makedirs(output_folder_path)

    # 打开原始图片
    image = Image.open(image_path)
    width, height = image.size

    # 计算格网切割的行数和列数
    rows = math.ceil(height / size)
    cols = math.ceil(width / size)

    # 切割图片并保存
    for row in range(rows):
        for col in range(cols):
            left = col * size
            top = row * size
            right = min(left + size, width)
            bottom = min(top + size, height)
            cropped_image = image.crop((left, top, right, bottom))

            # 计算中心点坐标
            center_x = (left + right) / 2
            center_y = (top + bottom) / 2

            # 保存切割后的图片
            output_file = os.path.join(output_folder_path, f"{row * cols + col}.png")
            cropped_image.save(output_file)

            print(f"切割后图片 {output_file} 的中心点坐标为 ({center_x}, {center_y})")

    print("图片保存完毕")
    print("生成图片文件夹位置为：" + output_folder_path)
    return output_folder_path

# 读取图片
img = plt.imread("D:\yjs\写写写\图片\DFS.png")

# 显示图片
plt.imshow(img)
plt.title('Click on the image')
plt.axis('on')

# 绘制格网线
grid_size = 400  # 格网大小
height, width, _ = img.shape
centers = []
for x in range(0, width, grid_size):
    plt.axvline(x, color='gray', linestyle='--', linewidth=0.5)
for y in range(0, height, grid_size):
    plt.axhline(y, color='gray', linestyle='--', linewidth=0.5)
    for x in range(0, width, grid_size):
        center_x = x + grid_size / 2
        center_y = y + grid_size / 2
        centers.append((center_x, center_y))
        plt.scatter(center_x, center_y, color='blue', s=50)

# 注册鼠标点击事件
plt.gcf().canvas.mpl_connect('button_press_event', onclick)

plt.show()

image_path = "D:\yjs\写写写\图片\DFS.png"  # 替换为你的图片路径
click_position = (x, y)  # 替换为你的点击坐标位置
output_folder_path = crop_image(image_path, 400, click_position)
