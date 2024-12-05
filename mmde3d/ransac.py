import open3d as o3d
import numpy as np


def plane_detection(pcd, tolerance=50):

    current_pcd = pcd
    planes = []
    all_planes = []
    while len(current_pcd.points) > tolerance:
        plane_model, inliers = current_pcd.segment_plane(
            distance_threshold=0.01, ransac_n=5, num_iterations=1000
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


def main(plypath, outpath):
    pcd = o3d.io.read_point_cloud(plypath)

    remain_pcd, planes_normal, all_planes = plane_detection(pcd, 1000)
    # print(np.array(planes).shape)
    combined_cloud = o3d.geometry.PointCloud()
    for i in range(len(all_planes)):
        combined_cloud += all_planes[i]
    o3d.io.write_point_cloud(outpath, combined_cloud)
    print("done")
    """
    for plane in planes:
        plane.paint_uniform_color(np.random.rand(3))


    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=5.0, origin=[0, 0, 0]
    )
    o3d.visualization.draw_geometries([remain_pcd, *planes, mesh_frame])
    """


# main(r"./mmde3d/preds/synth1.ply", r"./mmde3d/preds/synth1_nonsub_plane.ply")
main(
    r"/media/fys/T7 Shield/AdvancedGIS/read_test/synth1/synth1.ply",
    r"./mmde3d/preds/synth1_nonsub_plane.ply",
)
