import numpy as np
import pandas as pd
from plyfile import PlyData
import open3d as o3d


def convert_ply(input_path, output_path):
    plydata = PlyData.read(input_path)
    data = plydata.elements[0].data
    data_pd = pd.DataFrame(data)
    print(f"the shape of input ply is: {data_pd.shape}")

    data_np = np.zeros(data_pd.shape, dtype=np.float32)
    property_names = data[0].dtype.names
    print(property_names)

    for i, name in enumerate(property_names):
        data_np[:, i] = data_pd[name]
    data_np.astype(np.float32).tofile(output_path)
    print("saved")


"""
sometimes pointcloud has more attributes than 6 or three
"""


# input: ply
# output: bin(num, 6) xyz rgb
def convert2bin_xyzrgb(input_path, output_path):
    plydata = PlyData.read(input_path)
    data = plydata.elements[0].data
    data_pd = pd.DataFrame(data)
    print(f"the shape of input ply is: {data_pd.shape}")

    data_np = np.zeros(
        (data_pd.shape[0], 6), dtype=np.float32
    )  # for pointnet, the format of the data is float32
    property_names = data[0].dtype.names[:6]  # only x y z and RGB writen to the bin
    print(property_names)

    for i, name in enumerate(property_names):
        data_np[:, i] = data_pd[name]

    print(f"the shape of bin is: {data_np.shape}")
    data_np.astype(np.float32).tofile(output_path)
    print("saved")


# input: ply
# output: bin(num, 6) xyz(shift to origin) rgb
def convert2bin_xyzrgb_shifted(input_path, output_path):
    plydata = PlyData.read(input_path)
    data = plydata.elements[0].data
    data_pd = pd.DataFrame(data)
    print(f"the shape of input ply is: {data_pd.shape}")

    data_np = np.zeros(
        (data_pd.shape[0], 6), dtype=np.float32
    )  # for pointnet, the format of the data is float32
    property_names = data[0].dtype.names[:6]  # only x y z and RGB writen to the bin
    print(property_names)

    for i, name in enumerate(property_names):
        data_np[:, i] = data_pd[name]

    for i in range(3):
        min_val = data_np[:, i].min()
        data_np[:, i] -= min_val  # xnew = xold - xmin

    print(f"the shape of bin is: {data_np.shape}")
    data_np.astype(np.float32).tofile(output_path)
    print("saved")


# input: ply
# output: bin(num, 9) xyz rgb Normalized XYZ
# this is same with the S3DIS dataset
def convert2bin_xyzrgbNXYZ(input_path, output_path):
    plydata = PlyData.read(input_path)
    data = plydata.elements[0].data
    data_pd = pd.DataFrame(data)
    print(f"the shape of input ply is: {data_pd.shape}")

    data_np = np.zeros(
        (data_pd.shape[0], 6), dtype=np.float32
    )  # for pointnet, the format of the data is float32
    property_names = data[0].dtype.names[:6]  # only x y z and RGB
    print(property_names)

    for i, name in enumerate(property_names):
        data_np[:, i] = data_pd[name]

    # cal normalized xyz
    xyz = data_np[:, :3]

    min_vals = np.min(xyz, axis=0)
    max_vals = np.max(xyz, axis=0)
    normalized_xyz = (xyz - min_vals) / (max_vals - min_vals)
    data_np = np.hstack((data_np, normalized_xyz))

    print(f"the shape of bin is: {data_np.shape}")
    data_np.astype(np.float32).tofile(output_path)
    print("saved")


def convert_bin_xyz(inpath, outpath):
    bin_pcd = np.fromfile(inpath, dtype=np.float32)
    points = bin_pcd.reshape((-1, 3))[:, 0:3]
    o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    o3d.io.write_point_cloud(outpath, o3d_pcd)


if __name__ == "__main__":

    """
    convert_ply(
        "/media/fys/T7 Shield/AdvancedGIS/read_test/GE006_downsample.ply",
        "/media/fys/T7 Shield/AdvancedGIS/read_test/GE006_downsample.bin",
    )
    convert_ply(
        "/media/fys/T7 Shield/AdvancedGIS/read_test/UZH_dataset/2C05/2C05 - downsample.ply",
        "/media/fys/T7 Shield/AdvancedGIS/read_test/2C05_downsample.bin",
    )

    """
    """
    convert2bin_xyzrgbNXYZ(
        "/media/fys/T7 Shield/AdvancedGIS/read_test/synth1/synth1_downsample.ply",
        "/media/fys/T7 Shield/AdvancedGIS/read_test/synth1_xyz.bin",
    )

    """

    convert2bin_xyzrgb_shifted(
        "/media/fys/T7 Shield/AdvancedGIS/read_test/GE_006/GE006_downsample.ply",
        "/media/fys/T7 Shield/AdvancedGIS/read_test/GE_006/ge006_downsample.bin",
    )
