import open3d as o3d


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


if __name__ == "__main__":
    input_file = "/media/fys/T7 Shield/AdvancedGIS/rebuild/hdbscan_ge005/floor.ply"
    pcd = o3d.io.read_point_cloud(input_file)
    if not pcd.has_points():
        print("No points")
        exit(1)

    largest_plane, plane_model = extract_largest_plane(pcd)
    output_file = (
        "/media/fys/T7 Shield/AdvancedGIS/rebuild/hdbscan_ge005/floor_clean.ply"
    )
    o3d.io.write_point_cloud(output_file, largest_plane)
    print(f"saved to {output_file}")
