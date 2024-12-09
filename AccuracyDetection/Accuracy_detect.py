import numpy as np
import open3d as o3d

def extract_segments_from_ply_by_color(ply_file_path):
    """
    Extract segments from a .ply file based on unique color information.
    Each unique color represents a segment.
    """
    # Load the .ply file
    pcd = o3d.io.read_point_cloud(ply_file_path)
    
    # Convert point cloud to numpy arrays
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # Map unique colors to segment indices
    unique_colors = np.unique(colors, axis=0)
    segments = []
    for color in unique_colors:
        segment_indices = np.where((colors == color).all(axis=1))[0]
        segments.append(segment_indices.tolist())
    
    return segments, pcd

def calculate_iou(segments_pred, segments_gt):
    """
    Calculate Intersection over Union (IoU) for the segmented regions.
    """
    iou_scores = []
    
    for seg_pred in segments_pred:
        best_iou = 0
        for seg_gt in segments_gt:
            intersection = len(set(seg_pred).intersection(set(seg_gt)))
            union = len(set(seg_pred).union(set(seg_gt)))
            iou = intersection / union if union > 0 else 0
            best_iou = max(best_iou, iou)
        iou_scores.append(best_iou)

    return np.mean(iou_scores)

def visualize_segments(pcd, segments, title="Point Cloud Segments"):
    """
    Visualize segmented regions of the point cloud with different colors.
    """
    points = np.asarray(pcd.points)
    colors = np.zeros_like(points)  # Initialize colors with zeros

    # Assign a random color to each segment
    for segment in segments:
        random_color = np.random.rand(3)
        for idx in segment:
            colors[idx] = random_color

    pcd.colors = o3d.utility.Vector3dVector(colors)
    print(f"Visualizing: {title}")
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    # Define file paths for the predicted and ground truth .ply files
    predicted_file = "segmented_output.ply"  # Replace with your predicted segmentation file path
    ground_truth_file = "conferenceRoom_1_GT.ply"  # Replace with your ground truth file path

    # Extract segments from ground truth and predicted point clouds
    print("Extracting segments from ground truth...")
    segments_gt, pcd_gt = extract_segments_from_ply_by_color(ground_truth_file)
    print(f"Extracted {len(segments_gt)} segments from ground truth.")

    print("Extracting segments from prediction...")
    segments_pred, pcd_pred = extract_segments_from_ply_by_color(predicted_file)
    print(f"Extracted {len(segments_pred)} segments from prediction.")

    # Calculate IoU
    print("Calculating IoU...")
    iou_score = calculate_iou(segments_pred, segments_gt)
    print(f"IoU Score: {iou_score}")

    # Visualize ground truth and predicted segments
    visualize_segments(pcd_gt, segments_gt, title="Ground Truth Segments")
    visualize_segments(pcd_pred, segments_pred, title="Predicted Segments")
