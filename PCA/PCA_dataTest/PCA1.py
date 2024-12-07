import open3d as o3d
import numpy as np

# Load point cloud
pcd = o3d.io.read_point_cloud("01164-GE009.ply")
points = np.asarray(pcd.points)
num_points = len(points)
pcd_tree = o3d.geometry.KDTreeFlann(pcd)

# Parameters
k = 50  # Larger neighborhood size for stable curvature estimation
surface_threshold = 0.03  # For surface points
edge_threshold = 0.6  # For edge points
corner_isotropy_threshold = 0.2  # For isotropic curvature (corner points)

# Labels: 1 for surface, 2 for edge, 3 for corner
labels = np.zeros(num_points, dtype=int)

# PCA function for local geometry analysis
def pca(points):
    mean = np.mean(points, axis=0)
    centered_points = points - mean
    covariance_matrix = np.cov(centered_points.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    return eigenvalues, eigenvectors, mean

# Classification logic
for i in range(num_points):
    [_, idx, _] = pcd_tree.search_knn_vector_3d(points[i], k)
    local_points = points[idx]

    # Perform PCA
    eigenvalues, eigenvectors, mean = pca(local_points)
    eig_sum = np.sum(eigenvalues)
    lambda1, lambda2, lambda3 = eigenvalues / eig_sum

    # Step 1: Detect surface points
    if lambda1 / lambda2 < surface_threshold:
        labels[i] = 1  # Surface

    # Step 2: Detect edge points
    elif lambda2 / lambda3 < edge_threshold:
        labels[i] = 2  # Edge

    # Step 3: Detect corner points (isotropic curvature + edge proximity)
    elif (
        abs(lambda1 - lambda2) < corner_isotropy_threshold and
        abs(lambda2 - lambda3) < corner_isotropy_threshold and
        abs(lambda1 - lambda3) < corner_isotropy_threshold
    ):
        labels[i] = 3  # Corner

# Fallback classification for unlabeled points
unlabeled_indices = np.where(labels == 0)[0]
for i in unlabeled_indices:
    [_, idx, _] = pcd_tree.search_knn_vector_3d(points[i], k)
    local_points = points[idx]

    # Perform PCA
    eigenvalues, eigenvectors, mean = pca(local_points)
    eig_sum = np.sum(eigenvalues)
    lambda1, lambda2, lambda3 = eigenvalues / eig_sum

    # Fallback: Assign to the closest category
    if lambda1 / lambda2 < surface_threshold:
        labels[i] = 1  # Surface
    elif lambda2 / lambda3 < edge_threshold:
        labels[i] = 2  # Edge
    else:
        labels[i] = 3  # Corner

# Define color map
color_map = {
    1: [0, 1, 0],        # Green: Surface
    2: [1, 0, 0],        # Red: Edge
    3: [0, 0, 1],        # Blue: Corner
    0: [0.5, 0.5, 0.5],  # Gray: Default (should be minimal after fallback)
}

# Assign colors
colors = np.array([color_map[label] for label in labels])
pcd.colors = o3d.utility.Vector3dVector(colors)

# Save and visualize
o3d.io.write_point_cloud("output_hierarchical_classified.ply", pcd)
o3d.visualization.draw_geometries([pcd])

# Print classification results
print(f"Total points: {num_points}")
print(f"Surface points: {np.sum(labels == 1)}")
print(f"Edge points: {np.sum(labels == 2)}")
print(f"Corner points: {np.sum(labels == 3)}")
print(f"Unlabeled points (after fallback): {np.sum(labels == 0)}")
