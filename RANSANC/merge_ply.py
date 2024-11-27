import open3d as o3d
import os

root_path = "/Volumes/T7 Shield/AdvancedGIS/read_test/UZH_dataset/"


def merge_ply_files(input_files, output_file):
    combined_cloud = o3d.geometry.PointCloud()
    for file_path in input_files:
        cloud = o3d.io.read_point_cloud(root_path + file_path)
        combined_cloud += cloud

    o3d.io.write_point_cloud(root_path + output_file, combined_cloud)
    print(f"Combined point cloud saved as: {output_file}")


input_files = [
    "2C05/2C05_1 - Cloud.ply",
    "2C05/2C05_2 - Cloud.ply",
    "2C05/2C05_3 - Cloud.ply",
]
output_file = "2C05/2C05 - Cloud.ply"


merge_ply_files(input_files, output_file)
