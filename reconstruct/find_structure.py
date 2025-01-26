"""
create by yushuo, 24.01.2025
This scripts aims to rebuild the room structure from all wall pointclouds, or find the boundary of each room
"""

import open3d as o3d
import numpy as np
import os
import glob
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import hdbscan


"""creat the wall  class for each wall ply"""


class Wall:
    def __init__(self, ply_path: str, id: int) -> None:
        self.pcd = o3d.io.read_point_cloud(ply_path)
        self.id = id
        self.__wall2plane__()
        self.calculate_xy_convex()

    # get the parameter of the wall plane
    def __wall2plane__(self):
        self.plane_model, inliers = self.pcd.segment_plane(
            distance_threshold=0.01,
            ransac_n=3,
            num_iterations=1000,
        )
        self.plane_points = self.pcd.select_by_index(inliers)

        a, b, c, d = self.plane_model
        print(f"Plane equation: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")

    # get the convex hull of the 2D wall
    def calculate_xy_convex(self):
        if self.plane_points is None:
            raise ValueError(
                "No plane has been extracted yet. Please call extract_plane() first."
            )
        points = np.asarray(self.plane_points.points)[:, :2]
        self.points_2d = points

        hull = ConvexHull(points)
        hull_points = points[hull.vertices]

        plt.figure()
        plt.plot(points[:, 0], points[:, 1], "o")
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], "k-")
        plt.plot(hull_points[:, 0], hull_points[:, 1], "r--", lw=2)
        plt.fill(hull_points[:, 0], hull_points[:, 1], "r", alpha=0.3)
        plt.title("XY Projection of the Plane")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.grid(True)
        plt.axis("equal")
        plt.show()

        return hull_points

    # get the line segment of the 2D wall


"""
path = "./mmde3d/preds/synth1_paconv/all_walls/planes0.ply"
test = Wall(path, id=1)
"""


"""
HDBSCAN for spliting the wall from all walls ply file
"""


def cluster_pcd_with_hdbscan(pcd_path: str, min_cluster_size=15, min_samples=10):
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples
    )
    labels = clusterer.fit_predict(points)

    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Number of clusters found: {num_clusters}")
    print("Number of noise points:", list(labels).count(-1))

    """noise: -1"""
    unique_labels = np.unique(labels)
    print(f"labels: {unique_labels}")

    # colormap for different labels
    cmap = plt.get_cmap("magma", len(unique_labels))
    colors = [np.array(cmap(i + 1)[:3], dtype=np.float64) for i in labels]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd])

    return pcd, labels


# only choose the useful points for following work, based on the labels from HDBSCAN
def select_pcd_with_labels(
    pcd, labels: np.ndarray, selected_labels: np.ndarray, flag: int = 0
):
    """pcd: point cloud in o3d format, labels: labels list generated from hdbscan,
    selected_labels: defined bu user, -1 is noise;
    flag: if 1:save filtered pts to a file"""
    points = np.asarray(pcd.points)
    assert len(labels) == len(
        points
    ), "The length of labels didnot match the number of points in the pc."

    mask = np.in1d(labels, selected_labels)
    new_pts = points[mask]
    new_labels = labels[mask]

    assert len(new_labels) == len(
        new_pts
    ), "The length of new labels didnot match the number of points in the filtered pc."

    unique_labels = np.unique(new_labels)
    print(f"filtered unique label: {unique_labels}")
    cmap = plt.get_cmap("magma", len(np.unique(labels)))
    new_colors = [np.array(cmap(i + 1)[:3], dtype=np.float64) for i in new_labels]

    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(new_pts)
    new_pcd.colors = o3d.utility.Vector3dVector(new_colors)

    o3d.visualization.draw_geometries([new_pcd])

    """save the each label to the file"""
    if flag:
        for index in selected_labels:
            imask = np.in1d(labels, index)
            pts = points[imask]

            saved_pcd = o3d.geometry.PointCloud()
            saved_pcd.points = o3d.utility.Vector3dVector(pts)
            o3d.io.write_point_cloud(
                f"./reconstruct/hdbscan/label{index}.ply", saved_pcd
            )

    return new_pcd


# beacause there is no order for hdbsacn result, we need generate the label list,
def get_labels_most_pts(labels: np.ndarray, threshold: int):
    """threshold:  labels must have at least 'threshold' points"""
    unique_labels, counts = np.unique(labels, return_counts=True)
    filtered_labels = unique_labels[counts >= threshold]

    """also want to know the order of the filtered pcd, from high to low"""
    filtered_counts = counts[counts >= threshold]
    sorted_indices = np.argsort(-filtered_counts)
    sorted_labels = filtered_labels[sorted_indices]

    print(f"output ordered label is {sorted_labels}")

    return filtered_labels


if __name__ == "__main__":
    path = "./mmde3d/preds/synth1_paconv/all_walls.ply"
    # path = "/media/fys/T7 Shield/AdvancedGIS/read_test/ge005_ds_paconv_walls.ply"

    filtered_pcd, labels = cluster_pcd_with_hdbscan(path)
    selected_labels = get_labels_most_pts(labels, threshold=4000)
    select_pcd_with_labels(filtered_pcd, labels, selected_labels, flag=1)
