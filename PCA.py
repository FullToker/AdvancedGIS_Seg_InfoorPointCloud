import numpy as np
import open3d as o3d

def load_point_cloud(file_path):
    """
    Load the point cloud data.
    """
    pcd = o3d.io.read_point_cloud(file_path)
    print(f"Point cloud has {np.asarray(pcd.points).shape[0]} points.")
    return pcd

def compute_covariance_matrix(points):
    """
    Compute the covariance matrix for a set of points.
    """
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    covariance_matrix = np.dot(centered_points.T, centered_points) / len(points)
    return covariance_matrix

def pca_eigenvalues_and_vectors(cov_matrix):
    """
    Compute eigenvalues and eigenvectors of the covariance matrix.
    """
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)
    return eigenvalues[sorted_indices], eigenvectors[:, sorted_indices]

def compute_curvature(eigenvalues):
    """
    Compute curvature (surface variation) using eigenvalues.
    """
    curvature = eigenvalues[0] / np.sum(eigenvalues)
    return curvature

def detect_edges_and_estimate_normals(pcd, k_neighbors=30, curvature_threshold=0.01):
    """
    Perform edge detection and normal estimation using PCA for the point cloud data.
    """
    points = np.asarray(pcd.points)
    kd_tree = o3d.geometry.KDTreeFlann(pcd)

    normals = []
    edge_points = []

    for i in range(len(points)):
        # Find the k nearest neighbors
        _, idx, _ = kd_tree.search_knn_vector_3d(points[i], k_neighbors)
        neighbors = points[idx]

        # Compute the covariance matrix of the neighborhood
        cov_matrix = compute_covariance_matrix(neighbors)
        eigenvalues, _ = pca_eigenvalues_and_vectors(cov_matrix)

        # Compute curvature
        curvature = compute_curvature(eigenvalues)

        # If curvature is high, the point is an edge point (sharp feature)
        if curvature > curvature_threshold:
            edge_points.append(i)

        # Store the normal (the eigenvector corresponding to the smallest eigenvalue)
        normal = eigenvalues[2]
        normals.append(normal)

    # Visualizing edge points
    print(f"Detected {len(edge_points)} edge points.")
    
    # Visualizing normals and edge points
    edge_colors = np.zeros_like(points)
    edge_colors[edge_points] = [1, 0, 0]  # Red for edges

    pcd.colors = o3d.utility.Vector3dVector(edge_colors)
    
    return np.array(normals), edge_points, pcd

def segment_point_cloud(pcd, k_neighbors=30, curvature_threshold=0.01):
    """
    Perform PCA-based segmentation on the point cloud data.
    """
    points = np.asarray(pcd.points)
    kd_tree = o3d.geometry.KDTreeFlann(pcd)

    visited = np.zeros(len(points), dtype=bool)
    segments = []
    
    for i in range(len(points)):
        if visited[i]:
            continue
        
        # Seed a new region
        region = []
        to_process = [i]
        while to_process:
            current_point = to_process.pop(0)
            if visited[current_point]:
                continue
            
            # Mark as visited
            visited[current_point] = True
            region.append(current_point)

            # Find neighbors
            _, idx, _ = kd_tree.search_knn_vector_3d(points[current_point], k_neighbors)
            neighbors = points[idx]

            # Compute PCA on neighbors
            cov_matrix = compute_covariance_matrix(neighbors)
            eigenvalues, _ = pca_eigenvalues_and_vectors(cov_matrix)
            curvature = compute_curvature(eigenvalues)

            # Add neighbors to region if curvature is below threshold
            for neighbor_idx in idx:
                if not visited[neighbor_idx] and curvature < curvature_threshold:
                    to_process.append(neighbor_idx)
        
        # Add region to segments
        if len(region) > 0:
            segments.append(region)
    
    return segments

def save_ply(pcd, filename):
    """
    Save the point cloud to a PLY file.
    """
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Point cloud saved to {filename}")

def visualize_segments(pcd, segments):
    """
    Visualize segmented regions of the point cloud with different colors.
    """
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    if len(colors) == 0:
        colors = np.zeros_like(points)

    for segment in segments:
        color = np.random.rand(3)
        for idx in segment:
            colors[idx] = color

    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    # Load point cloud data
    file_path = "conferenceRoom_1_rgb.ply"  # Replace with your file path
    pcd = load_point_cloud(file_path)

    # Perform edge detection and normal estimation
    normals, edge_points, pcd_with_edges = detect_edges_and_estimate_normals(pcd)

    # Visualizing the result
    print("Visualizing edge points and normals...")
    o3d.visualization.draw_geometries([pcd])  # Visualization of edge points

    # Save edge-detected result
    save_ply(pcd_with_edges, "edge_detected_output.ply")

    # Perform segmentation
    print("Performing segmentation...")
    segments = segment_point_cloud(pcd)

    # Visualize segmented point cloud
    print("Visualizing segmentation...")
    visualize_segments(pcd, segments)

    # Save the segmented point cloud
    save_ply(pcd, "segmented_output.ply")
