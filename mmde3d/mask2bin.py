import numpy as np
import json
import open3d as o3d
import matplotlib.pyplot as plt

"""
bincols: each row(point) contains how many attributes in bin
outputlabel: choose some label to filter
"""


def mask_bin(
    binpath: str, jsonpath: str, outpath: str, binCols: int, outputlabel: list
):
    point_cloud = np.fromfile(binpath, dtype=np.float32).reshape(-1, binCols)
    print(f"points in bin: {point_cloud.shape}")
    with open(jsonpath, "r") as f:
        mask = json.load(f)

    mask_ls = mask["pts_semantic_mask"]
    print(f"length of mask is {len(mask_ls)}")
    assert len(mask_ls) == len(
        point_cloud
    ), "the number of points in two files are different."

    cmap = plt.get_cmap("viridis", 21)
    colors = [np.array(cmap(i + 1)[:3], dtype=np.float64) for i in mask_ls]

    filtered_points = []
    filtered_colors = []
    for i, label in enumerate(mask_ls):
        """for S3DIS dataset, label ['ceiling', 'floor', 'wall', 'beam', ...]"""
        if label in outputlabel:  # Only keep points with labels 1 and 2.
            filtered_points.append(point_cloud[i, 0:3])
            filtered_colors.append(colors[i])

    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    print("points done")
    new_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

    o3d.io.write_point_cloud(outpath, new_pcd)


if __name__ == "__main__":
    """
    bin_path = "/media/fys/T7 Shield/AdvancedGIS/read_test/ge005_downsample.bin"
    json_path = "/media/fys/T7 Shield/AdvancedGIS/read_test/ge005_downsample.json"
    ply_path = "/media/fys/T7 Shield/AdvancedGIS/read_test/ge005_ds_paconv_walls.ply"
    """

    bin_path = "./mmde3d/preds/synth1_downsample.bin"
    json_path = "./mmde3d/preds/synth1_downsample.json"
    ply_path = "./mmde3d/preds/synth1_paconv_floor.ply"

    mask_bin(bin_path, json_path, ply_path, binCols=6, outputlabel=[1])

"""
    with open(json_path, "r") as f:
        data = json.load(f)  # data is a dict
    print(type(data["pts_semantic_mask"]))  # is a list
    print(len(data["pts_semantic_mask"]))  # number is 135216
    print(type(data["pts_semantic_mask"][2])) # class int
"""
