import open3d as o3d
import numpy as np
import math


def plane_detection(pcd, tolerance=50):

    current_pcd = pcd
    planes = []
    all_planes = []
    while len(current_pcd.points) > tolerance:
        plane_model, inliers = current_pcd.segment_plane(
            distance_threshold=0.01, ransac_n=3, num_iterations=1000
        )
        if len(inliers) < tolerance:
            break
        inlier_indices = np.asarray(inliers)
        inlier_cloud = current_pcd.select_by_index(inlier_indices)
        current_pcd = current_pcd.select_by_index(inlier_indices, invert=True)

        # color the plane
        inlier_cloud.paint_uniform_color(np.random.rand(3))
        all_planes.append(inlier_cloud)

        normal_vector = plane_model[:3]
        point_in_plane = (
            -normal_vector * plane_model[3] / np.linalg.norm(normal_vector) ** 2
        )
        endpoint = point_in_plane + normal_vector * 2

        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector([point_in_plane, endpoint])
        line.lines = o3d.utility.Vector2iVector([[0, 1]])

        planes.append(line)
        planes.append(inlier_cloud)

    return current_pcd, planes, all_planes


def calNorm_z(point_cloud):

    point_cloud.estimate_normals()
    normals = np.asarray(point_cloud.normals)
    average_normal = np.mean(normals, axis=0)
    average_normal /= np.linalg.norm(average_normal)

    z_axis = np.array([0, 0, 1])
    cos_theta = np.dot(average_normal, z_axis)
    angle = math.acos(cos_theta)
    return angle


def get_wall(
    plypath: str,
    outpath: str,
    # wall_path=r"/media/fys/T7 Shield/AdvancedGIS/read_test/synth1/wall/",
    wall_path: str,
    sigma: int,
):
    pcd = o3d.io.read_point_cloud(plypath)

    remain_pcd, planes_normal, all_planes = plane_detection(pcd, sigma)
    num_planes = len(all_planes)
    # print(np.array(planes).shape)
    combined_cloud = o3d.geometry.PointCloud()
    print(f"Detected {num_planes} wall planes.")

    num_walls = 0
    for i in range(num_planes):

        """notice: only ceiling and floor, only walls or all big planes"""
        # if i == 0 or i == 1:
        if True:
            # if i != 0 and i != 1:
            # combined_cloud += all_planes[i]

            # remove the strip according to the angle
            # angle = calNorm_z(all_planes[i])
            # if ((abs(angle) < 1.4 or abs(angle) > 1.6) and len(all_planes[i]) > 2200):
            if True:
                combined_cloud += all_planes[i]

                # save each wall to a single ply file in wall folder
                file = wall_path + f"planes{num_walls}.ply"
                o3d.io.write_point_cloud(file, all_planes[i])

                num_walls += 1

    o3d.io.write_point_cloud(outpath, combined_cloud)
    print("done")
    print(f"Final get {num_walls} walls(planes).")


if __name__ == "__main__":
    # main(r"./mmde3d/preds/synth1.ply", r"./mmde3d/preds/synth1_nonsub_plane.ply")
    """
    get_wall(
        r"/media/fys/T7 Shield/AdvancedGIS/read_test/synth1/synth1_sg_clean.ply",
        r"/media/fys/T7 Shield/AdvancedGIS/read_test/synth1/synth1_floor_clean_index.ply",
        # r"./mmde3d/preds/synth1_wall_plane.ply",
    )
    """

    """some test for paconv model: synth1"""
    get_wall(
        plypath="./mmde3d/preds/synth1_paconv.ply",
        outpath="./mmde3d/preds/synth1_paconv/all_planes.ply",
        wall_path="./mmde3d/preds/synth1_paconv/all_planes/",
        sigma=1200,
    )
