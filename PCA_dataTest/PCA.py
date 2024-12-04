import open3d as o3d
import numpy as np

# 加载点云数据
pcd = o3d.io.read_point_cloud("conferenceRoom_1_rgb.ply")
pcd_tree = o3d.geometry.KDTreeFlann(pcd)  # 构建KD树用于快速查询邻域点

# 输出点云的总点数
points = np.asarray(pcd.points)
num_points = len(points)
print(f"数据总点数: {num_points}")

# 设置邻域大小和阈值
k = 10  # 邻域内包含的点数
threshold = 0.01  # 平面度阈值
normal_threshold = 0.8  # 法线方向阈值

# 存储法向量和标签
normals = np.zeros((num_points, 3))
labels = np.zeros(num_points, dtype=int)

# 定义颜色列表：4种颜色（3组平面 + 非平面）
color_list = [
    (0, 1, 0),   # Green normalized
    (0, 0, 1),   # Blue normalized
    (1, 0, 0),   # Red normalized
    (0, 0, 0)      # 黑色：非平面区域
]

# 遍历每个点进行局部 PCA
for i in range(num_points):
    # 提取当前点的邻域点
    [_, idx, _] = pcd_tree.search_knn_vector_3d(points[i], k)
    local_points = points[idx]

    # 计算邻域点的协方差矩阵
    mean = np.mean(local_points, axis=0)
    centered_points = local_points - mean
    cov_matrix = np.dot(centered_points.T, centered_points) / (k - 1)

    # 特征值分解
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    normal = eigenvectors[:, 0]  # 法向量是对应最小特征值的特征向量
    normals[i] = normal  # 保存法向量

    # 计算法线与坐标轴的夹角
    angles = np.abs(np.dot(normal, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).T))

    # 添加平面度过滤条件
    if eigenvalues[0] / eigenvalues[1] < threshold:  # 平面度判断

        # 根据法线方向分类平面
        if angles[0] > normal_threshold:  # X轴平行（前后平面）
            labels[i] = 1
        elif angles[1] > normal_threshold:  # Y轴平行（左右平面）
            labels[i] = 2
        elif angles[2] > normal_threshold:  # Z轴平行（上下平面）
            labels[i] = 3
        else:
            labels[i] = 4  # 非平面区域

# 输出平面点数和非平面点数
plane_counts = [np.sum(labels == i) for i in range(1, 4)]
non_plane_count = np.sum(labels == 4)
print(f"前后平面点数: {plane_counts[0]}, 左右平面点数: {plane_counts[1]}, 上下平面点数: {plane_counts[2]}, 非平面点数: {non_plane_count}")

# 根据标签分配颜色
colors = np.zeros((num_points, 3))
for i in range(4):
    colors[labels == (i + 1)] = np.array(color_list[i]) # 归一化颜色

# 设置颜色到点云
pcd.colors = o3d.utility.Vector3dVector(colors)

# 可视化结果
o3d.visualization.draw_geometries([pcd])

# 保存结果
o3d.io.write_point_cloud("output_pca_planes_simplified.ply", pcd)
print("基于PCA的平面分割完成，结果已保存为 output_pca_planes_simplified.ply")
