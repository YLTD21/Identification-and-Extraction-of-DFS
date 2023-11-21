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
        processed_mask = cv.morphologyEx(processed_mask, cv.MORPH_OPEN, kernel, iterations=4)
        # cv.imshow("Processed Mask", processed_mask)
        return processed_mask


               # cv.imshow("Processed Mask", processed_mask)

    def extract_river_centerline(self,image):
        # Convert the image to grayscale
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # Remove white pixels from the first and last rows
        gray[0, :] = 0
        gray[-1, :] = 0

        # Apply median blur to remove noise
        blurred = cv.medianBlur(gray, 5)

        # Enhance edges using an edge-preserving filter (e.g., bilateral filter)
        enhanced_edges = cv.bilateralFilter(blurred, 9, 75, 75)

        # Apply adaptive thresholding
        _, binary = cv.threshold(enhanced_edges, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        # Find contours in the binary image
        contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Find the contour with the maximum area (assuming it's the river)
        river_contour = max(contours, key=cv.contourArea)

        # Approximate the contour to reduce the number of points
        epsilon = 0.02 * cv.arcLength(river_contour, True)
        approx_contour = cv.approxPolyDP(river_contour, epsilon, True)

        # Create an empty mask to draw the river contour
        mask = np.zeros_like(gray)

        # Draw the river contour on the mask
        cv.drawContours(mask, [approx_contour], 0, (255), thickness=cv.FILLED)

        # Remove white pixels from the first and last rows in the mask
        mask[0, :] = 0
        mask[-1, :] = 0

        # Apply morphological operations to smooth the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

        # Find the skeleton of the mask
        skeleton = cv.ximgproc.thinning(mask)

        # Optionally, remove short branches from the skeleton
        pruned_skeleton = self.prune_skeleton(skeleton)

        # Process the edges of the skeleton
        processed_skeleton = self.process_edges(pruned_skeleton)

        return processed_skeleton

    def prune_skeleton(self,skeleton, min_branch_length=40):
        # Convert the skeleton to a binary image
        _, binary_skeleton = cv.threshold(skeleton, 200, 255, cv.THRESH_BINARY)

        # Find contours in the binary skeleton image
        contours, _ = cv.findContours(binary_skeleton, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Prune short branches from the skeleton
        pruned_skeleton = np.zeros_like(skeleton)
        for contour in contours:
            if cv.arcLength(contour, closed=False) > min_branch_length:
                cv.drawContours(pruned_skeleton, [contour], 0, (255), thickness=cv.FILLED)

        return pruned_skeleton

    def process_edges(self,skeleton):
        # Find the indices of white pixels in the first and last rows
        first_row_indices = np.where(skeleton[0, :] == 255)[0]
        last_row_indices = np.where(skeleton[-1, :] == 255)[0]

        # If there are multiple white pixels in the first row, keep only the middle one
        if len(first_row_indices) > 1:
            middle_index = len(first_row_indices) // 2
            skeleton[0, :] = 0
            skeleton[0, first_row_indices[middle_index]] = 255

        # If there are multiple white pixels in the last row, keep only the middle one
        if len(last_row_indices) > 1:
            middle_index = len(last_row_indices) // 2
            skeleton[-1, :] = 0
            skeleton[-1, last_row_indices[middle_index]] = 255
            # Process left edge
        left_edge_indices = np.where(skeleton[:, 0] == 255)[0]
        if len(left_edge_indices) > 1:
            middle_index = len(left_edge_indices) // 2
            skeleton[:, 0] = 0
            skeleton[left_edge_indices[middle_index], 0] = 255

        # Process right edge
        right_edge_indices = np.where(skeleton[:, -1] == 255)[0]
        if len(right_edge_indices) > 1:
            middle_index = len(right_edge_indices) // 2
            skeleton[:, -1] = 0
            skeleton[right_edge_indices[middle_index], -1] = 255
        return skeleton
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

    def overlay_centerline_on_mask(self):
        # 创建一个具有相同形状的红色图像，用于中心线
        centerline_image = np.zeros((self.processed_mask.shape[0], self.processed_mask.shape[1], 3), dtype=np.uint8)
        centerline_image[self.centerline > 0] = [0, 0, 255]  # 将中心线像素标记为红色
        center_line_thickness = 4
        kernel = np.ones((center_line_thickness, center_line_thickness), np.uint8)
        dilated_center_line = cv.dilate(centerline_image, kernel)
        # 将中心线图像与掩膜图像相加
        overlay = cv.addWeighted(dilated_center_line, 0.5, self.processed_mask, 0.5, 0)

        # # 创建一个与图像相同大小的图像来叠加kernel
        # kernel_image = np.zeros((self.processed_mask.shape[0], self.processed_mask.shape[1]), dtype=np.uint8)
        # kernel_image[100:109, 100:109] = kernel  # 使用适当的坐标来叠加kernel
        #
        # # 将kernel图像与叠加图像相加
        # result_image = cv.addWeighted(overlay, 0.7, cv.cvtColor(kernel_image, cv.COLOR_GRAY2BGR), 0.3, 0)

        # 显示叠加后的图像
        cv.imshow("Mask with Centerline and Kernel", overlay)
        cv.waitKey(0)
        cv.destroyAllWindows()


if __name__ == '__main__':

    # 识别河流为白，背景为黑的图像
    # path = r"D:\yjs\829\8.png"
    path = r"D:\yjs\826\253.png"
    # path = r"D:\image.png_20231019090731\134.png"


    # path = r"D:\yjs\253_red_black.png"
    pixlength = 1.19
    river_analysis = RiverAnalysis(path,pixlength)
    print("河流宽度为=",river_analysis.get_river_width())
    print("河流参数为=",river_analysis.get_river_parameter())
    river_analysis.show_centerline()
    river_analysis.show_processed_mask()
    # print(river_analysis.get_river_parameter())
    river_analysis.overlay_centerline_on_mask()
