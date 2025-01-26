import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


def estimate_normals(pc, radius=0.1):
    # 估计点云中每个点的法向量
    pc.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
    )


def custom_region_growing(pc, radius, angle_threshold):
    # 自定义区域生长算法，考虑法向量
    points = np.asarray(pc.points)
    normals = np.asarray(pc.normals)
    point_count = points.shape[0]
    labels = -1 * np.ones(point_count)  # 初始化所有点的标签为-1
    cluster_id = 0
    for i in range(point_count):
        if labels[i] == -1:
            # 如果当前点未被标记，则开始一个新的集合
            grow_region(i, points, normals, labels, cluster_id, radius, angle_threshold)
            cluster_id += 1


def grow_region(idx, points, normals, labels, cluster_id, radius, angle_threshold):
    stack = [idx]
    labels[idx] = cluster_id
    while stack:
        point_idx = stack.pop()
        for i in range(len(points)):
            if (
                labels[i] == -1
                and np.linalg.norm(points[point_idx] - points[i]) < radius
            ):
                angle = np.arccos(
                    np.clip(np.dot(normals[point_idx], normals[i]), -1.0, 1.0)
                )
                if angle < angle_threshold:
                    labels[i] = cluster_id
                    stack.append(i)


ply_path = "./reconstruct/hdbscan/label11.ply"
pc = o3d.io.read_point_cloud(ply_path)

# 估计法向量
estimate_normals(pc, radius=0.1)

# 参数设置
radius = 0.2  # 搜索半径
angle_threshold = np.pi / 6  # 法向量夹角阈值（例如30度）

# 进行区域生长
custom_region_growing(pc, radius, angle_threshold)

o3d.visualization.draw_geometries([pc])
