import numpy as np
import open3d as o3d
import os
import glob


def read_arrays(file_path):
    array_list = []
    with open(file_path, "r") as file:
        for line in file:
            numbers = line.strip().split()
            num_array = np.array(list(map(int, numbers)))
            array_list.append(num_array)
    return array_list


def project2xy(inpath: str, outpath: str, flag=1):
    pcd = o3d.io.read_point_cloud(inpath)

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # set the z to 0
    points[:, 2] = 0

    if flag == 1:
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(points)
        new_pcd.colors = o3d.utility.Vector3dVector(colors)

        o3d.io.write_point_cloud(outpath, new_pcd)
        print("write done")
    elif flag == 0:
        return pcd


def remove_outliers(inpath: str, flag=1):
    pcd = o3d.io.read_point_cloud(inpath)
    colors = pcd.colors
    normals = pcd.normals

    points_2d = np.asarray(pcd.points)[:, :2]

    center = np.mean(points_2d, axis=0)
    distances = np.linalg.norm(points_2d - center, axis=1)
    threshold = 4 * np.std(distances)
    filtered_points = np.asarray(pcd.points)[distances < threshold]
    colors = np.asarray(colors)[distances < threshold]
    normals = np.asarray(normals)[distances < threshold]

    pcd.points = o3d.utility.Vector3dVector(filtered_points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    if flag == 1:
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(filtered_points)
        new_pcd.colors = o3d.utility.Vector3dVector(colors)
        new_pcd.normals = o3d.utility.Vector3dVector(normals)

        directory = os.path.dirname(inpath)
        filename = os.path.basename(inpath)
        outpath = directory + "/wall_clean/" + filename[:-4] + "_clean.ply"

        o3d.io.write_point_cloud(outpath, new_pcd)
    else:
        return pcd


def filter_wall(infloder="/media/fys/T7 Shield/AdvancedGIS/read_test/synth1/wall/"):
    pattern = os.path.join(infloder, "*.ply")
    for file_path in glob.glob(pattern):
        # print(file_path)
        remove_outliers(file_path, 0)


def wall2line(
    infloder="/media/fys/T7 Shield/AdvancedGIS/read_test/synth1/wall/wall_clean/",
):
    # remove wall 14;
    line_ls = []
    pattern = os.path.join(infloder, "*.ply")
    num_wall = 0
    for inpath in glob.glob(pattern):
        num_wall += 1
        pcd = o3d.io.read_point_cloud(inpath)
        points_2d = np.asarray(pcd.points)[:, :2]
        m, b = np.polyfit(points_2d[:, 0], points_2d[:, 1], 1)
        A = m
        B = -1
        C = -b

        line_ls.append([A, B, C])

    # build the matrix of the intersection;

    return line_ls


if __name__ == "__main__":

    # inpath = "./mmde3d/preds/synth1_wall_plane.ply"
    """
    inpath = (
        r"/media/fys/T7 Shield/AdvancedGIS/read_test/synth1/synth1_wall_clean_index.ply"
    )
    outpath = (
        "/media/fys/T7 Shield/AdvancedGIS/read_test/synth1/wall_clean_2d_index.ply"
    )
    project2xy(inpath, outpath)
    """

    line_ls = wall2line()
    print(line_ls)
