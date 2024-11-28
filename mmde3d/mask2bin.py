import numpy as np
import json
import open3d as o3d
import matplotlib.pyplot as plt


def mask_bin_xyz(binpath: str, jsonpath: str, outpath: str):
    point_cloud = np.fromfile(binpath, dtype=np.float32).reshape(-1, 6)[:, :3]
    print(f"points in bin: {point_cloud.shape}")
    with open(jsonpath, "r") as f:
        mask = json.load(f)

    mask_ls = mask["pts_semantic_mask"]
    assert len(mask_ls) == len(
        point_cloud
    ), "the number of points in two files are different."

    cmap = plt.get_cmap("viridis", 21)
    colors = [np.array(cmap(i + 1)[:3], dtype=np.float64) for i in mask_ls]

    new_pcd = o3d.geometry.PointCloud()

    new_pcd.points = o3d.utility.Vector3dVector(point_cloud)
    new_pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(outpath, new_pcd)


if __name__ == "__main__":
    bin_path = "./mmde3d/preds/2C05_downsample.bin"
    json_path = "./mmde3d/preds/2C05_downsample.json"
    ply_path = "./mmde3d/preds/2C05_ds_sg.ply"

    mask_bin_xyz(bin_path, json_path, ply_path)

"""
    with open(json_path, "r") as f:
        data = json.load(f)  # data is a dict
    print(type(data["pts_semantic_mask"]))  # is a list
    print(len(data["pts_semantic_mask"]))  # number is 135216
    print(type(data["pts_semantic_mask"][2])) # class int
"""
