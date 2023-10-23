from numpy.polynomial import Polynomial
import cv2 as cv
import numpy as np
import skimage
from PIL import Image,ImageDraw
# from newcanshutiqu import extract_river_centerline,calculate_curvature,preprocess_mask
import LengthCurvature_w2r as lc
import matplotlib.pyplot as plt

class RiverAnalysis:
    def __init__(self, image_path, pixlength):
        self.pixlength = pixlength
        self._image_path = image_path
        self._image = cv.imread(image_path, cv.IMREAD_COLOR)  # 转换为灰度图像
        # 以灰度模式读取黑白掩膜

        # 图像预处理
        self.processed_mask = self.preprocess_mask(self._image)

        # 中心线提取
        self.centerline = self.extract_river_centerline(self.processed_mask)

        self.kernel = np.array([
            [135.0, 126.9, 116.6, 104.0, 90.0, 76.0, 63.4, 53.1, 45.0],
            [143.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 36.9],
            [153.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 26.6],
            [166.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 14.0],
            [180.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e-5],
            [194.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 346.0],
            [206.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 333.4],
            [216.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 323.1],
            [225.0, 233.1, 243.4, 256.0, 270.0, 284.0, 296.6, 306.9, 315.0]
            ]).astype(np.uint8)
    def preprocess_mask(self, mask_image):
        # 进行图像预处理的操作，例如二值化、腐蚀和膨胀等
        binary_image = cv.threshold(mask_image, 128, 255, cv.THRESH_BINARY)[1]

        # 2. 形态学操作 - 多次腐蚀和膨胀
        kernel = np.ones((5, 5), np.uint8)
        processed_mask = cv.morphologyEx(binary_image, cv.MORPH_CLOSE, kernel)
        processed_mask = cv.morphologyEx(processed_mask, cv.MORPH_OPEN, kernel, iterations=10)

        return processed_mask

    def extract_river_centerline(self, mask_image):
        # 1. 图像预处理
        processed_mask = self.preprocess_mask(mask_image)
        processed_mask = cv.cvtColor(processed_mask, cv.COLOR_BGR2GRAY)  # 将图像转换为灰度图像

        # 2. 寻找中心线
        skeleton = cv.ximgproc.thinning(processed_mask)

        # 3. 中心线优化（这里采用直线拟合）
        y_indices, x_indices = np.where(skeleton > 0)
        poly_coeffs = Polynomial.fit(y_indices, x_indices, deg=5)
        x_fit = poly_coeffs(y_indices)

        # 4. 裁剪拟合得到的x坐标，确保在图像范围内
        x_fit = np.clip(x_fit, 0, mask_image.shape[1] - 1)

        optimized_centerline = np.zeros_like(skeleton)
        optimized_centerline[y_indices, np.round(x_fit).astype(int)] = 255

        return optimized_centerline



    # def _stream_way_distance(self, image, x, y):  # 求掩模长度
    #     # 通过高阈值检测图像边缘
    #     border = cv.Canny(image, 100, 200, apertureSize=3)
    #     border_points = np.where(border == 255)
    #     print("bp:", border_points)
    #     border_points_list = []  # 中心线点的列表
    #     distance = 0
    #     print("x:", x, "y:", y)
    #     for point in zip(*border_points):
    #         if point[1] == y:  # 判断横着的
    #             print(point)
    #             border_points_list.append(point)
    #         if len(border_points_list) == 2:#这个地方刚开始没有缩进
    #             distance = abs(border_points_list[0][0] - border_points_list[1][0])
    #             return distance
    #     for point in zip(*border_points):
    #         if point[0] == x:  # 判断竖着
    #             print(point)
    #             border_points_list.append(point)
    #         if len(border_points_list) == 2:
    #             distance = abs(border_points_list[0][0] - border_points_list[1][0])
    #             return distance
    #         distance = 100  # 其他情况，默认距离为100
    #     if distance == 0:
    #         distance = 100
    #     return distance
    def compute_width(self, x, y):
        # 初始化宽度
        width = 0
        # 创建方向线
        direction_line = np.array([x, y])
        # 计算垂直线
        vertical_line = np.array([-y, x])

        for step in range(-200, 201):  # 调整步长以适应图像尺寸
            # 计算垂直线上的点
            point = np.array([x, y]) + step * vertical_line
            pixel_x, pixel_y = point.astype(int)

            if (0 <= pixel_x < self.processed_mask.shape[0]) and (0 <= pixel_y < self.processed_mask.shape[1]):
                # 如果像素在图像内，检查是否与河流边界交叉
                if (self.processed_mask[pixel_x, pixel_y] == 0).any():
                    # 计算距离并将其添加到宽度中
                    distance = np.sqrt((pixel_x - x) ** 2 + (pixel_y - y) ** 2) * self.pixlength
                    width += distance

        return width

    def measure_width(self, x, y):
        # 初始化最小宽度的列表
        min_widths = []

        for angle in range(0, 360, 5):  # 按5度的步长遍历方向
            # 计算当前角度下的方向向量
            angle_rad = np.deg2rad(angle)
            direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])

            # 从当前中心线像素坐标开始，生成垂直线
            for step in range(-200, 201):  # 调整步长以适应图像尺寸
                vertical_line = direction * step
                point = np.array([x, y]) + vertical_line
                pixel_x, pixel_y = point.astype(int)

                if (0 <= pixel_x < self.processed_mask.shape[0]) and (0 <= pixel_y < self.processed_mask.shape[1]):
                    # 如果像素在图像内，检查是否与河流边界交叉
                    if (self.processed_mask[pixel_x, pixel_y] == 0).any():
                        # 计算距离并添加到最小宽度列表
                        distance = np.linalg.norm(vertical_line) * self.pixlength
                        min_widths.append(distance)

        # 获取最小宽度并按升序排序
        min_widths.sort()

        # 如果没有足够的宽度值，将其设置为一个大的默认值
        while len(min_widths) < 2:
            # min_widths.append(400)
            min_widths.append(self.pixlength * min(self.processed_mask.shape[0] - x, x, self.processed_mask.shape[1] - y, y))
        # 取两个最小宽度值相加作为宽度
        width = min_widths[0]
        # width = min_widths[0] + min_widths[1]

        return width
    # def measure_width(self, x, y):
    #     kernel_size = self.kernel.shape[0]
    #     half_kernel = kernel_size // 2
    #
    #     width_values = []
    #
    #     for i in range(-half_kernel, half_kernel + 1):
    #         for j in range(-half_kernel, half_kernel + 1):
    #             dx = x + i
    #             dy = y + j
    #             if (
    #                     dx < 0
    #                     or dy < 0
    #                     or dx >= self.processed_mask.shape[1]
    #                     or dy >= self.processed_mask.shape[0]
    #             ):
    #                 continue  # 忽略超出图像边界的像素
    #             dist = self.compute_distance(x, y, dx, dy)
    #             width_values.append(dist * self.kernel[i + half_kernel, j + half_kernel])
    #         river_analysis.overlay_centerline_on_mask(self.kernel)
    #     width_values.sort()
    #     return sum(width_values)
    #
    # def compute_distance(self, x1, y1, x2, y2):
    #     # 计算两个像素之间的距离，乘以分辨率
    #     distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) * self.pixlength
    #     return distance

    # def get_river_length(self):
    #     return len(self.points_list)

    def get_river_area(self):
        gray = cv.cvtColor(self._image, cv.COLOR_BGR2GRAY)
        # 转化为二值图
        ret, binary = cv.threshold(gray, 0, 255, 0)
        # cv.imshow("skeleton", binary)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        area = np.sum(binary == 255)  # 14167
        return area

    def get_river_width(self):

        mask_image = self.preprocess_mask(self._image)
        centerline = self.extract_river_centerline(mask_image)

        # 初始化变量
        length_counts = 0
        width_counts = 0

        # 遍历中心线上的点
        for point in zip(*np.where(centerline > 0)):
            x, y = point
            width = self.measure_width(x, y)  # 修正参数顺序
            # width = self._stream_way_distance(self._image, x, y)  # 修正参数顺序

            # 进行宽度测量
            if width > 0:
                length_counts += width
                width_counts += 1

        # 计算平均宽度
        if width_counts == 0:
            width_counts = 1
        return round(length_counts / width_counts, 2) * 1.19

    def show_centerline(self):

        cv.imshow("River Centerline", self.centerline)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def show_processed_mask(self):

        cv.imshow("Processed Mask", self.processed_mask)
        cv.waitKey(0)
        cv.destroyAllWindows()
    def get_river_parameter(self):

        pixel_length = self.pixlength
        river=lc.LengthCurvature(pixel_length,self._image_path)
        river_length,river_curvature = river.process()



        # return river_length, self.get_river_area(), round(river_curvature, 2), self.get_river_width()
        return river_length, self.get_river_area(), river_curvature, self.get_river_width()

    def overlay_centerline_on_mask(self, kernel):
        # 创建一个具有相同形状的红色图像，用于中心线
        centerline_image = np.zeros((self.processed_mask.shape[0], self.processed_mask.shape[1], 3), dtype=np.uint8)
        centerline_image[self.centerline > 0] = [0, 0, 255]  # 将中心线像素标记为红色

        # 将中心线图像与掩膜图像相加
        overlay = cv.addWeighted(centerline_image, 0.5, self.processed_mask, 0.5, 0)

        # 创建一个与图像相同大小的图像来叠加kernel
        kernel_image = np.zeros((self.processed_mask.shape[0], self.processed_mask.shape[1]), dtype=np.uint8)
        kernel_image[100:109, 100:109] = kernel  # 使用适当的坐标来叠加kernel

        # 将kernel图像与叠加图像相加
        result_image = cv.addWeighted(overlay, 0.7, cv.cvtColor(kernel_image, cv.COLOR_GRAY2BGR), 0.3, 0)

        # 显示叠加后的图像
        cv.imshow("Mask with Centerline and Kernel", result_image)
        cv.waitKey(0)
        cv.destroyAllWindows()


if __name__ == '__main__':

    path = r"D:\image.png_20231019090731\154.png"
    # path = r"D:\yjs\253_red_black.png"
    pixlength = 1.19
    river_analysis = RiverAnalysis(path,pixlength)
    print("河流宽度为=",river_analysis.get_river_width())
    print("河流参数为=",river_analysis.get_river_parameter())
    river_analysis.show_centerline()
    river_analysis.show_processed_mask()
    # print(river_analysis.get_river_parameter())
    # river_analysis.overlay_centerline_on_mask()
