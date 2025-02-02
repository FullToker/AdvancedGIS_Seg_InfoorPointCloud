import open3d as o3d
import numpy as np


def extract_largest_plane(
    pcd, distance_threshold=0.02, ransac_n=3, num_iterations=1000
):
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    [a, b, c, d] = plane_model
    print(f"Plane model: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")

    plane_cloud = pcd.select_by_index(inliers)
    return plane_cloud, plane_model


def extract_two_largest_planes(
    pcd, distance_threshold=0.05, ransac_n=3, num_iterations=1000
):
    planes = []
    plane_models = []

    temp_pcd = pcd

    for _ in range(2):
        plane_model, inliers = temp_pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )
        [a, b, c, d] = plane_model
        print(f"Plane model: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")

        plane_cloud = temp_pcd.select_by_index(inliers)
        planes.append(plane_cloud)
        plane_models.append(plane_model)

        temp_pcd = temp_pcd.select_by_index(inliers, invert=True)

    z_means = [np.mean(np.asarray(plane.points)[:, 2]) for plane in planes]
    ceiling_idx = np.argmax(z_means)
    floor_idx = np.argmin(z_means)

    ceiling_plane = planes[ceiling_idx]
    floor_plane = planes[floor_idx]

    print(f"Identified ceiling plane: Model - {plane_models[ceiling_idx]}")
    print(f"Identified floor plane: Model - {plane_models[floor_idx]}")

    return ceiling_plane, floor_plane


if __name__ == "__main__":
    input_file = "/media/fys/T7 Shield/AdvancedGIS/rebuild/GE005_new_second/GE_005_cds_CF - Cloud.ply"
    pcd = o3d.io.read_point_cloud(input_file)
    if not pcd.has_points():
        print("No points")
        exit(1)

    """
    largest_plane, plane_model = extract_largest_plane(pcd)
    output_file = (
        "/media/fys/T7 Shield/AdvancedGIS/rebuild/hdbscan_ge005/floor_clean.ply"
    )
    o3d.io.write_point_cloud(output_file, largest_plane)
    print(f"saved to {output_file}")
    """
    ceiling, floor = extract_two_largest_planes(pcd)
    o3d.io.write_point_cloud(
        "/media/fys/T7 Shield/AdvancedGIS/rebuild/GE005_new_second/GE005_ceiling.ply",
        ceiling,
    )
    o3d.io.write_point_cloud(
        "/media/fys/T7 Shield/AdvancedGIS/rebuild/GE005_new_second/GE005_floor.ply",
        floor,
    )
