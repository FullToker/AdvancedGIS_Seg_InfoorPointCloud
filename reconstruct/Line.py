import open3d as o3d
import numpy as np
from itertools import combinations

def fit_plane_ransac(point_cloud, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
    """Fit a plane to a point cloud using RANSAC."""
    plane_model, inliers = point_cloud.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )
    return plane_model, inliers

def compute_intersection_line(plane1, plane2):
    """Compute the intersection line between two planes."""
    n1 = np.array(plane1[:3])
    n2 = np.array(plane2[:3])
    direction = np.cross(n1, n2)
    
    if np.linalg.norm(direction) < 1e-6:  # Check if planes are parallel
        return None, None
    
    A = np.array([n1, n2, direction])
    d = np.array([-plane1[3], -plane2[3], 0])
    point_on_line = np.linalg.lstsq(A, d, rcond=None)[0]
    
    return point_on_line, direction

def is_line_intersecting_cloud(line_point, line_direction, point_cloud, num_samples=200, threshold=0.2):
    """
    Check if a line intersects with a point cloud.
    Sample multiple points along the line and check their distances to the point cloud.
    """
    # Generate sample points along the line
    t_values = np.linspace(-5, 5, num_samples)  # Generate points along the line
    sample_points = line_point + np.outer(t_values, line_direction)
    
    # Compute distances from sample points to the point cloud
    distances = point_cloud.compute_point_cloud_distance(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sample_points)))
    distances = np.asarray(distances)
    
    # Check if any sampled point is within the threshold distance
    return np.any(distances < threshold)

def crop_point_cloud_with_lines(point_cloud, intersection_lines, tolerance=0.2):
    """Crop the point cloud using intersection lines."""
    cropped_points = []
    points = np.asarray(point_cloud.points)
    
    for point, direction in intersection_lines:
        # Compute the distance from each point to the line
        line_direction = direction / np.linalg.norm(direction)
        projected_points = point + np.dot(points - point, line_direction[:, None]) * line_direction
        distances = np.linalg.norm(points - projected_points, axis=1)
        
        # Keep points close to the intersection line
        mask = distances < tolerance
        cropped_points.append(points[mask])
    
    if cropped_points:
        cropped_points = np.vstack(cropped_points)
        return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cropped_points))
    else:
        return None

def extract_wall_boundary(point_cloud):
    """Extract the boundary of a wall point cloud using Convex Hull."""
    if len(point_cloud.points) < 10:  # Check if point cloud has enough points
        return None
    hull = point_cloud.compute_convex_hull()[0]
    boundary_points = np.asarray(hull.vertices)
    return boundary_points

def reconstruct_wall_mesh(boundary_points):
    """Reconstruct a 3D wall mesh from boundary points."""
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(boundary_points))
    hull, _ = pc.compute_convex_hull()
    return hull

def load_point_clouds(ply_files):
    """Load multiple .ply files as Open3D point clouds."""
    point_clouds = []
    for file in ply_files:
        pcd = o3d.io.read_point_cloud(file)
        point_clouds.append(pcd)
    return point_clouds

def main(ply_files):
    point_clouds = load_point_clouds(ply_files)
    
    # Fit planes to all walls using RANSAC
    planes = []
    for i, pc in enumerate(point_clouds):
        plane_model, inliers = fit_plane_ransac(pc)
        planes.append((plane_model, pc))
        print(f"Wall {i + 1}: Plane model = {plane_model}")
    
    # Compute intersection lines and validate them based on wall boundaries
    valid_intersection_lines = []
    for (i, (plane1, pc1)), (j, (plane2, pc2)) in combinations(enumerate(planes), 2):
        point, direction = compute_intersection_line(plane1, plane2)
        if point is None:
            continue
        
        # Check if the intersection line is valid (intersects both wall point clouds)
        if is_line_intersecting_cloud(point, direction, pc1) and is_line_intersecting_cloud(point, direction, pc2):
            valid_intersection_lines.append((point, direction))
            print(f"Valid intersection line between wall {i + 1} and wall {j + 1}")
    
    # Extract and reconstruct wall models
    wall_models = []
    for pc in point_clouds:
        # Crop point cloud based on intersection lines
        cropped_pc = crop_point_cloud_with_lines(pc, valid_intersection_lines, tolerance=0.2)
        if cropped_pc is not None and len(cropped_pc.points) > 10:
            boundary_points = extract_wall_boundary(cropped_pc)
            if boundary_points is not None:
                wall_mesh = reconstruct_wall_mesh(boundary_points)
                wall_models.append(wall_mesh)
    
    if not wall_models:
        print("No wall models were generated. Try adjusting the tolerance or checking the input point clouds.")
    
    # Visualize walls and valid intersection lines
    lines = []
    line_set_points = []
    for point, direction in valid_intersection_lines:
        start_point = point - 5 * direction
        end_point = point + 5 * direction
        line_set_points.append(start_point)
        line_set_points.append(end_point)
        lines.append([len(line_set_points) - 2, len(line_set_points) - 1])
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_set_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in lines])
    
    # Visualize everything together
    o3d.visualization.draw_geometries(point_clouds + [line_set] + wall_models, window_name="3D Wall Reconstruction")
    o3d.visualization.draw_geometries([line_set] + wall_models, window_name="wall model")

if __name__ == "__main__":
    ply_files = [
        "wall0_clean.ply", "wall1_clean.ply", "wall2_clean.ply", "wall3_clean.ply",
        "wall4_clean.ply", "wall5_clean.ply", "wall6_clean.ply", "wall7_clean.ply",
        "wall8_clean.ply", "wall9_clean.ply", "wall10_clean.ply", "wall11_clean.ply",
        "wall12_clean.ply", "wall13_clean.ply", "wall15_clean.ply"
    ]
    main(ply_files)
