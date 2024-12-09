import numpy as np
import open3d as o3d


def read_arrays(file_path):
    array_list = []
    with open(file_path, "r") as file:
        for line in file:
            numbers = line.strip().split()
            num_array = np.array(list(map(int, numbers)))
            array_list.append(num_array)
    return array_list


def project2xy(
    inpath: str,
    outpath: str,
    index_txt=r"/media/fys/T7 Shield/AdvancedGIS/read_test/synth1/",
):
    pcd = o3d.io.read_point_cloud(inpath)
    index_ls = read_arrays(index_txt + "index_output.txt")
    pcd = pcd.select_by_index(index_ls[0])

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # set the z to 0
    points[:, 2] = 0

    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(points)
    new_pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(outpath, new_pcd)
    print("write done")


if __name__ == "__main__":

    # inpath = "./mmde3d/preds/synth1_wall_plane.ply"
    inpath = (
        r"/media/fys/T7 Shield/AdvancedGIS/read_test/synth1/synth1_wall_clean_index.ply"
    )
    outpath = (
        "/media/fys/T7 Shield/AdvancedGIS/read_test/synth1/wall_clean_2d_index.ply"
    )
    project2xy(inpath, outpath)
