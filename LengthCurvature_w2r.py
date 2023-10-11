import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial

class LengthCurvature:
    def __init__(self, pixel_length, mask_image):
        self.pixel_length = pixel_length
        self.mask_image = mask_image

    def preprocess_mask(self, mask_image):
        # 1. 二值化
        binary_image = cv2.inRange(mask_image, (0, 0, 0), (0, 0, 255))

        # 2. 形态学操作 - 多次腐蚀和膨胀
        kernel = np.ones((5, 5), np.uint8)
        processed_mask = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel, iterations=10)  # 多次腐蚀和膨胀

        return processed_mask

    def extract_river_centerline(self, mask_image):
        # 1. 图像预处理
        processed_mask = self.preprocess_mask(mask_image)

        # 2. 寻找中心线
        skeleton = cv2.ximgproc.thinning(processed_mask)

        # 3. 中心线优化（这里采用直线拟合）
        y_indices, x_indices = np.where(skeleton > 0)
        poly_coeffs = Polynomial.fit(y_indices, x_indices, deg=2)
        x_fit = poly_coeffs(y_indices)
        # 4. 裁剪拟合得到的x坐标，确保在图像范围内
        x_fit = np.clip(x_fit, 0, mask_image.shape[1] - 1)

        optimized_centerline = np.zeros_like(skeleton)
        optimized_centerline[y_indices, np.round(x_fit).astype(int)] = 255

        return optimized_centerline

    def calculate_river_length(self, centerline_image):
        y_indices, x_indices = np.where(centerline_image > 0)

        # 计算相邻像素点之间的距离，并累加得到河流长度
        total_length = 0
        for i in range(1, len(x_indices)):
            dx = x_indices[i] - x_indices[i - 1]
            dy = y_indices[i] - y_indices[i - 1]
            length = np.sqrt(dx ** 2 + dy ** 2) * self.pixel_length
            total_length += length

        return total_length

    def calculate_curvature(self, centerline_image, river_length):
        y_indices, x_indices = np.where(centerline_image > 0)

        # 计算中心线的开始坐标和结束坐标
        x_start, y_start = x_indices[0], y_indices[0]
        x_end, y_end = x_indices[-1], y_indices[-1]

        # 计算中心线开始坐标到结束坐标的直线距离
        straight_distance = np.sqrt((x_end - x_start) ** 2 + (y_end - y_start) ** 2) * self.pixel_length

        # 计算弯曲度
        total_length = 0
        for i in range(1, len(x_indices)):
            dx = x_indices[i] - x_indices[i - 1]
            dy = y_indices[i] - y_indices[i - 1]
            length = np.sqrt(dx ** 2 + dy ** 2) * self.pixel_length
            total_length += length
        curvature = total_length / straight_distance

        return curvature

    def process(self):
        mask_image = cv2.imread(self.mask_image)

        # 提取河流中心线
        centerline_image = self.extract_river_centerline(mask_image)
        river_length = self.calculate_river_length(centerline_image)

        plt.imshow(centerline_image, cmap='gray')
        plt.title('centerline_imagebatter')
        print("river_length：{} 米".format(river_length))

        # 计算弯曲度
        river_curvature = self.calculate_curvature(centerline_image, river_length)

        plt.imshow(centerline_image, cmap='gray')
        plt.title('centerline_imagebatter')
        print("river curvature：{}".format(river_curvature))
        return str(river_length),str(river_curvature)
if __name__ == "__main__":
    pixel_length = 1.19
    mask_image = r"D:\yjs\253_red_black.png"
    length_curvature_calculator = LengthCurvature(pixel_length, mask_image)
    length_curvature_calculator.process()
