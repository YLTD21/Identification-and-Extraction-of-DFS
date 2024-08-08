import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 请将'path_to_your_excel_file.xlsx'替换为您的Excel文件的实际路径
excel_file_path = r"D:\yjs\宽度长度弯曲度2.5Excel.xlsx"
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签SimHei
# 读取Excel文件
df = pd.read_excel(excel_file_path)

# 设置阈值
threshold = 1000

# 初始化用于存储结果的列表
histogram_data = []

# 遍历DataFrame中的每一行，计算平均宽度和距离
for index, row in df.iterrows():
    # 获取当前行的距离
    current_distance = row['Distance']

    # 筛选出与当前行距离相差小于阈值的行
    mask = (np.abs(df['Distance'] - current_distance) < threshold)

    # 计算宽度和距离的平均值
    if df[mask].empty:
        average_width = np.nan  # 如果没有满足条件的行，则设置为NaN
        average_distance = np.nan
    else:
        # average_width = df.loc[mask, 'River Width'].mean()
        # average_width = df.loc[mask, 'River Length'].mean()
        average_width = df.loc[mask, 'River Curvature'].mean()
        average_distance = df.loc[mask, 'Distance'].mean()

    # 存储结果
    histogram_data.append((average_distance, average_width))

# 过滤掉NaN值
histogram_data = [item for item in histogram_data if not np.isnan(item[0]) and not np.isnan(item[1])]

# 绘制直方图
plt.figure(figsize=(10, 6))
plt.plot([data[0] for data in histogram_data], [data[1] for data in histogram_data], alpha=0.5)
# plt.scatter([data[0] for data in histogram_data], [data[1] for data in histogram_data], alpha=0.5)
plt.xlabel('Average Distance from Vertex (m)')
# plt.xlabel('距离顶点距离 (m)')

# plt.ylabel('Average River Width (m)')
# plt.ylabel('Average River Length (m)')

# plt.ylabel('河流长度')
plt.ylabel('Average River Curvature')
# plt.title('Average River Length vs. Distance from Vertex')
# plt.title('Average River Width vs. Distance from Vertex')
plt.title('Average River Curvature vs. Distance from Vertex')
# plt.title('据顶点不同距离河流长度变化')
plt.grid(True)
plt.show()