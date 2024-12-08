import numpy as np
import open3d as o3d


def project2xy(inpath: str, outpath: str, removed=(18 / 19)):
    pcd = o3d.io.read_point_cloud(inpath)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # set the z to 0
    points[:, 2] = 0
    """
    new_colors = np.zeros_like(colors)
    new_colors[:, 0] = colors[:, 0]
    new_colors[:, 1] = colors[:, 0]
    new_colors[:, 2] = colors[:, 0]
    """

    tolerance = 0.05
    indices = np.where(abs(colors[:, 0] - removed) > tolerance)[0]

    new_points = points[indices]
    new_colors = colors[indices]

    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(new_points)
    new_pcd.colors = o3d.utility.Vector3dVector(new_colors)

    o3d.io.write_point_cloud(outpath, new_pcd)
    print("write done")


if __name__ == "__main__":

    inpath = "./mmde3d/preds/synth1_wall_plane.ply"
    outpath = "/media/fys/T7 Shield/AdvancedGIS/read_test/synth1/wall_2d.ply"
    project2xy(inpath, outpath)
