import open3d as o3d
import numpy as np

# Load the classified point cloud
pcd = o3d.io.read_point_cloud("output_pca_planes_simplified.ply")  # Replace with your file path

# Extract point colors and convert to numpy array
colors = np.asarray(pcd.colors)

# Total number of points
num_points = len(pcd.points)

# Define the color mappings (matching your classification colors)
color_map = {
    "Front-back plane": np.array([0, 1, 0]),  # Green
    "Left-right plane": np.array([0, 0, 1]),  # Blue
    "Top-bottom plane": np.array([1, 0, 0]),  # Red
    "Non-planar region": np.array([0, 0, 0])  # Black
}

# Tolerance for color matching
tolerance = 0.01

# Helper function to count points based on color
def count_points_by_color(colors, target_color, tolerance):
    diff = np.linalg.norm(colors - target_color, axis=1)
    return np.sum(diff < tolerance)

# Count points for each category
front_back_count = count_points_by_color(colors, color_map["Front-back plane"], tolerance)
left_right_count = count_points_by_color(colors, color_map["Left-right plane"], tolerance)
top_bottom_count = count_points_by_color(colors, color_map["Top-bottom plane"], tolerance)
non_plane_count = count_points_by_color(colors, color_map["Non-planar region"], tolerance)

# Print results
print(f"Total number of points: {num_points}")
print(f"Front-back plane points: {front_back_count}")
print(f"Left-right plane points: {left_right_count}")
print(f"Top-bottom plane points: {top_bottom_count}")
print(f"Non-planar points: {non_plane_count}")
