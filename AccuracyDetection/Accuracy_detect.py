import open3d as o3d
import numpy as np

# Load the segmented point cloud
pcd = o3d.io.read_point_cloud("output_hierarchical_classified.ply")  # Replace with your segmented file path

# Extract colors and convert to NumPy array
colors = np.asarray(pcd.colors)

# Define the color mappings (from segmentation step)
color_map = {
    "Surface Points": [0, 1, 0],   # Green
    "Edge Points": [1, 0, 0],      # Red
    "Corner Points": [0, 0, 1],    # Blue
    "Unlabeled Points": [0.5, 0.5, 0.5],  # Gray
}

# Helper function to count points by color
def count_points_by_color(colors, target_color, tolerance=0.01):
    diff = np.linalg.norm(colors - target_color, axis=1)
    return np.sum(diff < tolerance)

# Count points for each category
counts = {}
for label, target_color in color_map.items():
    counts[label] = count_points_by_color(colors, np.array(target_color))

# Print results
for label, count in counts.items():
    print(f"{label}: {count}")
