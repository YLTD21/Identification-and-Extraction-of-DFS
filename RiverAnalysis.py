import cv2 as cv
# import cv2
import numpy as np
import skimage
from PIL import Image,ImageDraw
from newcanshutiqu import extract_river_centerline,calculate_curvature,preprocess_mask
import LengthCurvature_w2r as lc
class RiverAnalysis:
    def __init__(self, image_path, pixlength):
        self.pixlength = pixlength
        self._image_path = image_path
        self._image = cv.imdecode(np.fromfile(self._image_path, dtype=np.uint8), -1)
        kernel = np.array([
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
        # processed_mask = cv.cvtColor(self._image, cv.COLOR_BGR2GRAY)
        # _, processed_mask = cv.threshold(processed_mask, 127, 255, cv.THRESH_BINARY_INV)
        contours = skimage.morphology.skeletonize(self._image)#现在不是列表，是个array类型
        ab=np.where(contours!= 0)
        print("---------",ab)
        merged_list = [(ab[0][i], ab[1][i]) for i in range(len(ab[0]))]#这个把contours改为了列表
        print("merged_list=",merged_list)
        center_line = np.zeros_like(self._image) + contours


        line_gray = cv.cvtColor(center_line, cv.COLOR_RGB2GRAY)
        line_binary = cv.threshold(line_gray, 0, 1, 0)[1].astype(np.uint8)
        self.points_list, self.angles_list = self._image_mask(line_binary, kernel)
        print(self.points_list, self.angles_list, len(self.points_list), len(self.angles_list), sep='\n')

        # img = cv.drawContours(self._image, self.points_list, -1, (0, 255, 0), 3)
        # cv.imshow("drawing", img)
        image = Image.new('RGB', (400, 400), color='black')##中心线提取，新建的一个画布，merged_list为中心线的坐标列表
        draw =ImageDraw.Draw(image)
        for coord in merged_list:
            y, x = coord
            draw.point((x, y), fill='white')
        # image.show()
        image.save("D://imgsave//image.png")
    def _image_mask(self, image, mask):
        height, width = image.shape
        h, w = mask.shape
        height_new = height - h + 1
        width_new = width - w + 1
        points = np.where(image == 1)
        points_list = []
        points_ls = []
        angle_ls = []
        image_new = None
        for point in zip(*points):
            points_list.append(point)
        for i in range(4, height_new + 4):
            for j in range(4, width_new + 4):
                if (i, j) in points_list:
                    # print("起:", i - 4, j-4)
                    # print("终:", i + h - 5, j + w - 5)
                    # print("当前:", i, j)
                    image_new = image[i - 4:i + h - 4, j - 4:j + w - 4] * mask
                    ps = list(zip(*np.where(image_new != 0)))
                    # print(ps)
                    if len(ps) == 1:
                        angle = image_new[list(ps)[0]]
                        points_ls.append((i, j))
                        angle_ls.append(angle)
                    # else:
                    #     image[i, j] = 0
        return points_ls, angle_ls

    def _stream_way_distance(self, image, x, y):  # 求掩模长度
        # 通过高阈值检测图像边缘
        border = cv.Canny(image, 100, 200, apertureSize=3)
        border_points = np.where(border == 255)
        print("bp:", border_points)
        border_points_list = []  # 中心线点的列表
        distance = 0
        print("x:", x, "y:", y)
        for point in zip(*border_points):
            if point[1] == y:  # 判断横着的
                print(point)
                border_points_list.append(point)
            if len(border_points_list) == 2:#这个地方刚开始没有缩进
                distance = abs(border_points_list[0][0] - border_points_list[1][0])
                return distance
        for point in zip(*border_points):
            if point[0] == x:  # 判断竖着
                print(point)
                border_points_list.append(point)
            if len(border_points_list) == 2:
                distance = abs(border_points_list[0][0] - border_points_list[1][0])
                return distance
            distance = 30  # 其他情况，默认距离为100
        if distance == 0:
            distance = 30
        return distance

    def get_river_length(self):
        return len(self.points_list)

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
        # 对角度加90并判断
        img = cv.cvtColor(self._image, cv.COLOR_RGB2GRAY)
        X_left = 0
        X_right = img.shape[0]

        Y_left = 0
        Y_right = img.shape[1]
        length_counts = 0  # 总长度
        width_counts = 0  # 宽度个数
        # y=k0*x+y0-k0*x0
        for i in range(len(self.angles_list)):
            self.angles_list[i] = (self.angles_list[i] + 90) % 360
            k0, x0, y0 = np.tan(self.angles_list[i] * np.pi / 180), self.points_list[i][1], self.points_list[i][0]
            print("角度：", self.angles_list[i])
            print("斜率：", k0)
            print("(", self.points_list[i][1], self.points_list[i][0], ")")

            pts_list = []
            if k0!=0:
                # 当X=0时
                if 0 <= y0 - k0 * x0 <= img.shape[1]:
                    X = 0
                    Y = int(y0 - k0 * x0)
                    # print((X, Y))
                    pts_list.append((X, Y))
                if 0 <= img.shape[1] * k0 + y0 - k0 * x0 <= img.shape[1]:  # 当 X=img.shape
                    X = img.shape[1]
                    Y = int(img.shape[1] * k0 + y0 - k0 * x0)
                    # print((X, Y))
                    pts_list.append((X, Y))
                # 当Y=0时
                if 0 <= int((k0 * x0 - y0) / k0) <= img.shape[0]:
                    X = int((k0 * x0 - y0) / k0)
                    Y = 0
                    # print((X, Y))
                    pts_list.append((X, Y))
                if 0 <= int((img.shape[0] + k0 * x0 - y0) / k0) <= img.shape[0]:
                    X2 = int((img.shape[0] + k0 * x0 - y0) / k0)
                    Y2 = img.shape[0]
                    # print((X2, Y2))
                    pts_list.append((X2, Y2))
                try:
                    point1, point2 = pts_list
                except Exception:
                    point1, point2 = pts_list[0], pts_list[1]
            print("X, Y", point1, point2)
            empty_graph = np.zeros_like(self._image)
            cv.line(empty_graph, point1, point2, (255, 255, 255), 1)
            img_binary = cv.threshold(img, 0, 1, 0)[1].astype(np.uint8)
            slope_line = img_binary * empty_graph[:, :, 0]
            origin_image = cv.imread(self._image_path, 0)
            mask_distance = self._stream_way_distance(origin_image, self.points_list[i][1], self.points_list[i][0])

            print("mask_distance", mask_distance)
            mask_left_from = self.points_list[i][1] - mask_distance
            mask_left_to = self.points_list[i][1] + mask_distance
            mask_right_from = self.points_list[i][0] - mask_distance
            mask_right_to = self.points_list[i][0] + mask_distance
            mask = np.zeros_like(img, dtype="uint8")
            mask[mask_right_from:mask_right_to, mask_left_from:mask_left_to] = 1
            slope_line *= mask
            # total_img = self._image + cv.merge([slope_line, slope_line, slope_line])  # center_line
            # cv.circle(total_img, (self.points_list[i][1], self.points_list[i][0]), 1, (0, 255, 0), 2)
            width = np.where(slope_line == 255)
            width_len = len(list(zip(*width)))
            if width_len == 0:
                continue
            length_counts += width_len
            width_counts += 1
        if width_counts == 0:
            width_counts = 1
        return round(length_counts / width_counts, 2)*1.19

    def get_river_parameter(self):

        try:
            start_point = self.points_list[0]
        except Exception:
            return 0, 0, 0, 0
        end_point = self.points_list[-1]
        river_length = self.get_river_length()
        river_curvature = np.sqrt(
            (start_point[0]-end_point[0])**2 + (start_point[1]-end_point[1])**2
        )

        pixel_length = self.pixlength
        river=lc.LengthCurvature(pixel_length,self._image_path)
        river_length,river_curvature = river.process()



        # return river_length, self.get_river_area(), round(river_curvature, 2), self.get_river_width()
        return river_length, self.get_river_area(), river_curvature, self.get_river_width()


if __name__ == '__main__':
    
    path = r"D:\yjs\253_red_black.png"
    pixlength = 1.19
    river_analysis = RiverAnalysis(path,pixlength)
    print("River_width=",river_analysis.get_river_width())
    print("River_parameter=",river_analysis.get_river_parameter())
    # print(river_analysis.get_river_parameter())

