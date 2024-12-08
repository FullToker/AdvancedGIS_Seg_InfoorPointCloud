import open3d as o3d
import numpy as np

# Load point cloud
pcd = o3d.io.read_point_cloud("01164-GE009.ply")
points = np.asarray(pcd.points)
num_points = len(points)
pcd_tree = o3d.geometry.KDTreeFlann(pcd)

# Parameters
k = 100  # Larger neighborhood size for stable PCA
labels = np.zeros(num_points, dtype=int)

# PCA function for local geometry analysis
def pca(points):
    mean = np.mean(points, axis=0)
    centered_points = points - mean
    covariance_matrix = np.cov(centered_points.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    return eigenvalues, eigenvectors, mean

# Precompute eigenvalue ratios for all points
ratios = []
for i in range(num_points):
    [_, idx, _] = pcd_tree.search_knn_vector_3d(points[i], k)
    local_points = points[idx]

    # Perform PCA
    eigenvalues, _, _ = pca(local_points)
    lambda1, lambda2, lambda3 = eigenvalues
    if lambda1 + lambda2 + lambda3 == 0:  # Avoid division by zero
        ratios.append((0, 0))
        continue

    # Compute ratios
    ratios.append((lambda3 / lambda2, lambda2 / lambda1))

# Convert ratios to NumPy array for threshold computation
ratios = np.array(ratios)

# Robust threshold computation (exclude extreme outliers)
linear_threshold = np.percentile(ratios[:, 0], 95)
planar_threshold = np.percentile(ratios[:, 1][ratios[:, 1] < 1000], 95)

print(f"Adaptive Linear Threshold: {linear_threshold}")
print(f"Adaptive Planar Threshold: {planar_threshold}")

# Classify points based on dimensional features
for i in range(num_points):
    if ratios[i][0] == 0 and ratios[i][1] == 0:  # Noise or invalid eigenvalues
        labels[i] = 0
        continue

    d3_d2, d2_d1 = ratios[i]
    if d3_d2 > linear_threshold and d2_d1 > linear_threshold:
        labels[i] = 1  # Linear (Line)
    elif d3_d2 < planar_threshold and d2_d1 > planar_threshold:
        labels[i] = 2  # Planar (Plane)
    elif abs(d3_d2 - 1) < 0.3 and abs(d2_d1 - 1) < 0.3:  # Relaxed isotropy
        labels[i] = 3  # Isotropic (3D spread)
    else:
        labels[i] = 0  # Unclassified

# Fallback classification for unclassified points
for i in range(num_points):
    if labels[i] == 0:
        d3_d2, d2_d1 = ratios[i]
        if d3_d2 > d2_d1:
            labels[i] = 1  # Linear
        elif d2_d1 > d3_d2:
            labels[i] = 2  # Planar
        else:
            labels[i] = 3  # Isotropic

# Define color map
color_map = {
    0: [0.5, 0.5, 0.5],  # Gray: Noise or unclassified
    1: [1, 0, 0],        # Red: Linear
    2: [0, 1, 0],        # Green: Planar
    3: [0, 0, 1],        # Blue: Isotropic
}

# Assign colors
colors = np.array([color_map[label] for label in labels])
pcd.colors = o3d.utility.Vector3dVector(colors)

# Save and visualize
o3d.io.write_point_cloud("output_refined_classified.ply", pcd)

# Software-only rendering to address GLFW errors
try:
    o3d.visualization.draw_geometries([pcd])
except Exception as e:
    print(f"Rendering failed: {e}. Try updating GPU drivers or using software rendering.")

# Print classification results
print(f"Total points: {num_points}")
print(f"Linear points: {np.sum(labels == 1)}")
print(f"Planar points: {np.sum(labels == 2)}")
print(f"Isotropic points: {np.sum(labels == 3)}")
print(f"Unclassified points: {np.sum(labels == 0)}")
