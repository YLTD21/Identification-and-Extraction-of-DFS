import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
import os
import time
import math
from PIL import Image
Image.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import tkinter as tk
from tkinter import messagebox
from scipy.interpolate import interp1d
import pandas as pd

Image.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

class LengthCurvature:
    def __init__(self, pixel_length, folder_path):
        self.pixel_length = pixel_length
        self.folder_path = folder_path
        self.river_lengths = []
        self.river_curvatures = []
        self.river_widths = []
        self.distances = []

    def preprocess_mask(self, mask_image):
        binary_image = cv2.threshold(mask_image, 128, 255, cv2.THRESH_BINARY)[1]
        kernel = np.ones((5, 5), np.uint8)
        processed_mask = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel, iterations=5)
        return processed_mask

    def extract_river_centerline(self, mask_image):
        processed_mask = self.preprocess_mask(mask_image)
        # processed_mask = cv2.cvtColor(processed_mask, cv2.COLOR_BGR2GRAY)
        skeleton = cv2.ximgproc.thinning(processed_mask)

        y_indices, x_indices = np.where(skeleton >= 0)
        if y_indices.size == 0 or x_indices.size == 0:
            raise ValueError("mask_image 中不包含任何河流区域（y_indices 或 x_indices 是空的）。")

        y_indices, x_indices = np.where(skeleton > 0)
        if y_indices.size == 0 or x_indices.size == 0:
            print("未检测到河流区域，默认生成一条竖线作为中心线。")
            y_indices = np.arange(processed_mask.shape[0])
            x_indices = np.full_like(y_indices, processed_mask.shape[1] // 2)
        try:
            poly_coeffs = Polynomial.fit(y_indices, x_indices, deg=3)
        except np.linalg.LinAlgError:
            print("SVD 在多项式拟合中未能收敛，跳过该中心线。")
            y_indices = np.arange(processed_mask.shape[0])
            x_indices = np.full_like(y_indices, processed_mask.shape[1] // 2)
            poly_coeffs = Polynomial.fit(y_indices, x_indices, deg=1)

        x_fit = poly_coeffs(y_indices)
        x_fit = np.clip(x_fit, 0, mask_image.shape[1] - 1)
        optimized_centerline = np.zeros_like(skeleton)
        optimized_centerline[y_indices, np.round(x_fit).astype(int)] = 255
        return optimized_centerline

    def calculate_river_length(self, centerline_image):
        y_indices, x_indices = np.where(centerline_image > 0)
        total_length = 0
        for i in range(1, len(x_indices)):
            dx = x_indices[i] - x_indices[i - 1]
            dy = y_indices[i] - y_indices[i - 1]
            length = np.sqrt(dx ** 2 + dy ** 2) * self.pixel_length
            total_length += length
        return total_length

    def calculate_curvature(self, centerline_image, river_length):
        y_indices, x_indices = np.where(centerline_image > 0)
        x_start, y_start = x_indices[0], y_indices[0]
        x_end, y_end = x_indices[-1], y_indices[-1]
        straight_distance = np.sqrt((x_end - x_start) ** 2 + (y_end - y_start) ** 2) * self.pixel_length
        total_length = 0
        for i in range(1, len(x_indices)):
            dx = x_indices[i] - x_indices[i - 1]
            dy = y_indices[i] - y_indices[i - 1]
            length = np.sqrt(dx ** 2 + dy ** 2) * self.pixel_length
            total_length += length
        curvature = total_length / straight_distance
        return curvature

    def measure_width(self, x, y, mask_image,pixel_length):
        min_widths = []

        for angle in range(0, 360, 5):
            angle_rad = np.deg2rad(angle)
            direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])

            for step in range(-200, 201):  # 调整步长以适应图像尺寸
                vertical_line = direction * step
                point = np.array([x, y]) + vertical_line
                pixel_x, pixel_y = point.astype(int)

                # 确保使用mask_image来检查像素值
                if (0 <= pixel_x < mask_image.shape[1]) and (0 <= pixel_y < mask_image.shape[0]):
                    if (mask_image[pixel_y, pixel_x] == 0):
                        distance = np.linalg.norm(vertical_line) * self.pixel_length
                        min_widths.append(distance)

        min_widths.sort()

        while len(min_widths) < 2:
            min_widths.append(self.pixel_length * min(mask_image.shape[1] - x, x, mask_image.shape[0] - y, y))

        width = min_widths[0] + min_widths[1]
        return width

    def calculate_river_width(self, centerline_image,mask_image):
        min_widths = []

        for y_index, x_index in zip(*np.where(centerline_image > 0)):
            width = self.measure_width(x_index, y_index, mask_image, self.pixel_length)
            min_widths.append(width)

        return sum(min_widths) / len(min_widths) if min_widths else 0

    def process(self, distances):
        files = os.listdir(self.folder_path)
        for distance, file in zip(distances, files):
            if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
                image_path = os.path.join(self.folder_path, file)
                mask_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                centerline_image = self.extract_river_centerline(mask_image)
                river_length = self.calculate_river_length(centerline_image)
                river_curvature = self.calculate_curvature(centerline_image, river_length)
                river_width = self.calculate_river_width(centerline_image, mask_image)

                self.river_lengths.append(river_length)
                self.river_curvatures.append(river_curvature)
                self.river_widths.append(river_width)
                self.distances.append(distance)

                print("图像文件：", file)
                print("河流长度：{} 米".format(river_length))
                print("河流弯曲度：{}".format(river_curvature))
                print("河流宽度：{} 米".format(river_width))

        self.generate_excel()

    def generate_excel(self):
        data = {
            'Distance from Vertex': self.distances,
            'River Length': self.river_lengths,
            'River Curvature': self.river_curvatures,
            'River Width': self.river_widths
        }
        df = pd.DataFrame(data)

        excel_filename = 'river_data.xlsx'
        df.to_excel(excel_filename, index=False)

        print(f"Data exported to {excel_filename} successfully.")

def crop_image(image_path, size, click_position):
    timestamp = time.strftime("%Y%m%d%H%M%S")
    folder_name = os.path.basename(image_path) + "_" + timestamp
    output_folder_path = os.path.join("D://", folder_name)
    os.makedirs(output_folder_path)
    image = Image.open(image_path)
    width, height = image.size
    rows = math.ceil(height / size)
    cols = math.ceil(width / size)
    distances = []

    for row in range(rows):
        for col in range(cols):
            left = col * size
            top = row * size
            right = min(left + size, width)
            bottom = min(top + size, height)
            cropped_image = image.crop((left, top, right, bottom))
            center_x = (left + right) / 2
            center_y = (top + bottom) / 2
            distance = math.sqrt((center_x - click_position[0]) ** 2 + (center_y - click_position[1]) ** 2)
            distances.append(distance)

            output_file = os.path.join(output_folder_path, f"{row * cols + col}.png")
            cropped_image.save(output_file)
            print(f"切割后图片 {output_file} 的中心点坐标为 ({center_x}, {center_y})，与点击位置的欧氏距离为 {distance}")
    print("图片保存完毕")
    print("生成图片文件夹位置为：" + output_folder_path)
    return output_folder_path, distances

def plot_length_curve(distances, river_lengths):
    valid_distances = []
    valid_lengths = []
    for d, l in zip(distances, river_lengths):
        if 0 <= l <= 2000:
            valid_distances.append(d)
            valid_lengths.append(l)

    plt.figure(figsize=(12, 6))
    plt.scatter(valid_distances, valid_lengths, color='blue', label='Data Points')
    plt.xlabel('Distance from Vertex')
    plt.ylabel('Length (m)')
    plt.title('River Length vs. Distance from Vertex')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_curvature_curve(distances, river_curvatures):
    plt.figure(figsize=(12, 6))
    plt.scatter(distances, river_curvatures, color='orange', label='Data Points')
    plt.xlabel('Distance from Vertex')
    plt.ylabel('Curvature')
    plt.title('River Curvature vs. Distance from Vertex')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 2)
    plt.show()

def onclick(event):
    global x, y
    x = event.xdata
    y = event.ydata
    print('x = %d, y = %d' % (x, y))
    plt.scatter(x, y, color='red', s=100)
    plt.draw()
    root = tk.Tk()
    root.withdraw()
    result = messagebox.askyesno("Confirmation", "Do you want to confirm this point as the vertex?")
    root.destroy()
    if result:
        for center_x, center_y in centers:
            plt.plot([x, center_x], [y, center_y], 'r--', linewidth=0.5)
            distance = math.sqrt((center_x - x) ** 2 + (center_y - y) ** 2)
            plt.text((x + center_x) / 2, (y + center_y) / 2, f"{distance:.2f}", color='black', fontsize=8, ha='center', va='center')
        plt.draw()
        return x, y
    else:
        return None, None
# Read image
img = plt.imread("G:\yemaolin\clipanother2.tif")

# Display image
plt.imshow(img)
plt.title('Click on the image')
plt.axis('on')

# Draw grid lines
grid_size = 400
height, width= img.shape
centers = []
for x in range(0, width, grid_size):
    plt.axvline(x, color='gray', linestyle='--', linewidth=0.5)
for y in range(0, height, grid_size):
    plt.axhline(y, color='gray', linestyle='--', linewidth=0.5)

# Register mouse click event
plt.gcf().canvas.mpl_connect('button_press_event', onclick)

plt.show()

# Crop image
image_path = r"G:\yemaolin\clipanother2.tif"
# image_path = "G:\yemaolin\segformer-pytorch-master\clip-pspnet2.png"
click_position = (x, y)
output_folder_path, distances = crop_image(image_path, 400, click_position)

# Process image
pixel_length = 1.19
length_curvature_calculator = LengthCurvature(pixel_length, output_folder_path)
length_curvature_calculator.process(distances)

# Plot curve
plot_length_curve(distances, length_curvature_calculator.river_lengths)
plot_curvature_curve(distances, length_curvature_calculator.river_curvatures)

# 示例 Excel 文件路径
excel_file = 'river_data.xlsx'

# 绘制参数曲线图
# plot_parameter_curve_from_excel(excel_file, distance_interval=100, parameter_column='River Curvature', ylim=(0, 2))