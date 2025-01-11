"""
 extract the wall model from clean wall point cloud
Created by Yushuo
"""

import numpy as np
import open3d as o3d
import os
import glob
from shapely.geometry import Polygon, MultiPoint
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


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


def line_intersection(line1, line2):
    """
    calculate the intersections
    return: (x, y) or none
    """
    [A1, B1, C1] = line1
    [A2, B2, C2] = line2

    det = A1 * B2 - A2 * B1
    if np.isclose(det, 0):
        return None

    x = (B1 * C2 - B2 * C1) / det
    y = (A2 * C1 - A1 * C2) / det
    return [x, y]


def linePoints(a=0, b=0, c=0, ref=[200.0, -150.0]):
    """given a,b,c for straight line as ax+by+c=0,
    return a pair of points based on ref values
    e.g linePoints(-1,1,2) == [(-1.0, -3.0), (1.0, -1.0)]
    """
    if a == 0 and b == 0:
        raise Exception("linePoints: a and b cannot both be zero")
    return [(-c / a, p) if b == 0 else (p, (-c - a * p) / b) for p in ref]


def wall2line(
    infloder="/media/fys/T7 Shield/AdvancedGIS/read_test/synth1/wall/wall_clean/",
):
    # remove wall 14;
    line_ls = []
    all_walls = []
    color_walls = []

    pattern = os.path.join(infloder, "*.ply")
    num_wall = 0
    for inpath in glob.glob(pattern):
        num_wall += 1
        pcd = o3d.io.read_point_cloud(inpath)
        points_2d = np.asarray(pcd.points)[:, :2]
        colors = np.asarray(pcd.colors)[0, :]

        all_walls.append(points_2d)
        color_walls.append(colors)

        model = LinearRegression()
        x_data = points_2d[:, 0].reshape(-1, 1)
        y_data = points_2d[:, 1].reshape(-1, 1)
        model.fit(x_data, y_data)
        slope = model.coef_[0]
        intercept = model.intercept_

        """
        plt.scatter(x_data, y_data, color="blue", label="Data Points")
        plt.plot(x_data, slope * x_data + intercept, color="red", label="Fitted Line")
        plt.title(f"{inpath}")
        plt.show()
        """

        m, b = np.polyfit(points_2d[:, 0], points_2d[:, 1], 1)

        """print the fit value"""
        print(f"the fitted result is {slope},{intercept} && {m},{b}\n")

        A = m
        B = -1
        C = b

        line_ls.append([A, B, C])

    # print(f"colors: {color_walls}")

    # build the matrix of the intersection;
    # only iterate the half matrix
    intersections = [[np.nan for _ in range(num_wall)] for _ in range(num_wall)]
    for ii in range(num_wall - 1):
        for jj in range(ii + 1, num_wall):
            line1 = line_ls[ii]
            line2 = line_ls[jj]

            # two conditions: has intersection and the intersection is close to the original point cloud
            if line_intersection(line1, line2) is not None:
                intersections[jj][ii] = line_intersection(line1, line2)

    """draw the wall line and filtered intersections"""
    fig, ax = plt.subplots()
    for ii in range(num_wall):
        ax.axline(
            *linePoints(a=line_ls[ii][0], b=line_ls[ii][1], c=line_ls[ii][2]),
            color=color_walls[ii],
        )

    # restrict the area to a limited range(a little larger than bouding box)
    ax.set_xlim([-2, 7])
    ax.set_ylim([-15, 3])

    """filtered intersection from original matrix"""
    valid_intersections = np.array(
        [
            item
            for sublist in intersections
            for item in sublist
            if isinstance(item, list)
        ]
    )
    """
    np.array(
        [
            elem
            for sublist in intersections
            for elem in sublist
            if elem is not None and not np.isnan(elem)
        ]
    )"""
    # ax.plot(valid_intersections[:, 0], valid_intersections[:, 1], "ro")
    plt.show()

    return intersections


"""calculate the convex hull of the floor plan, as well as the bounding box"""


# i want to find the mimimum convex of the floor plan
def find_convex(
    infloder="/media/fys/T7 Shield/AdvancedGIS/read_test/synth1/wall/wall_clean/",
):
    pattern = os.path.join(infloder, "*.ply")
    num_wall = 0
    all_points = []  # store all points of walls, (num,3)
    all_walls = []  # store all points of each wall (len, num, 2) only xy
    for inpath in glob.glob(pattern):
        num_wall += 1
        pcd = o3d.io.read_point_cloud(inpath)
        xypts = np.asarray(pcd.points)[:, :2]
        all_walls.append(xypts)

    all_points = np.vstack(all_walls)
    print(all_points.shape)

    # calculate the convex hull
    hull = MultiPoint(all_points).convex_hull
    x_coords, y_coords = hull.exterior.coords.xy  # Separate x and y coordinates
    hull_coords = list(zip(x_coords, y_coords))
    print("Coords of convex hull is:", hull_coords)
    print(f"shapes of coords is {np.shape(hull_coords)}")

    # cal the bounding box of the floor plan
    mins = np.min(np.asarray(hull_coords), axis=0)
    maxs = np.max(np.asarray(hull_coords), axis=0)
    print(f"min_x = {mins[0]}, min_y = {mins[1]}")
    print(f"max_x = {maxs[0]}, max_y = {maxs[1]}")

    # visualization ----//------------------------------
    x, y = zip(*all_points)
    plt.scatter(x, y, color="blue", label="Original Points")
    hull_x, hull_y = zip(*hull_coords)
    plt.plot(hull_x, hull_y, color="red", label="Convex Hull")

    # Add labels and legend
    plt.title("Convex Hull Visualization")
    plt.legend()
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()


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

    # print(wall2line())
    wall2line()
    # find_convex()
