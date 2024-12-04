import open3d as o3d
import numpy as np


# 0: voxel downsample, 1: uniform downsample, 2: random downsample
def downsample(ply_path: str, out_path: str, flag: int = 0):
    pcd = o3d.io.read_point_cloud(ply_path)
    print(f"size of original pc: {len(pcd.points)}")

    # Voxel downsampling
    if flag == 0:
        voxel_size = 0.05  # size of voxel
        downsampled_pcd = pcd.voxel_down_sample(voxel_size)
    elif flag == 1:
        every_k_points = 150
        downsampled_pcd = pcd.uniform_down_sample(every_k_points)
    elif flag == 2:
        sample_ratio = 0.5
        num_points = np.int32(len(pcd.points) * sample_ratio)
        indices = np.random.choice(len(pcd.points), num_points, replace=False)
        downsampled_pcd = pcd.select_by_index(indices)

    o3d.io.write_point_cloud(out_path, downsampled_pcd)
    print(f"the file saved in {out_path}")

    print(f"size of original pc: {len(downsampled_pcd.points)}")


if __name__ == "__main__":
    """
    root_path = "/Volumes/T7 Shield/AdvancedGIS/read_test/UZH_dataset/2C05/"
    in_ply = "2C05 - Cloud.ply"
    out_ply = "2C05 - downsample.ply"
    """

    root_path = "/Users/fengys/Desktop/advancedGIS/uzh/"
    in_ply = "synth2.ply"
    out_ply = "synth2_downsample.ply"
    downsample(root_path + in_ply, root_path + out_ply, 0)
