from main import (
    readCloud,
    colorInlierOutlier,
    cloudVisualizer,
    keypoints2spheres,
    progressbar,
)
import open3d as o3d


pc_in = "/Volumes/T7 Shield/AdvancedGIS/read_test/UZH_dataset/2C05/"
pc_out = ""
filename = "2C05_1 - Cloud.ply"

winDims = [1000, 1000]
viewCtrl = [0, 0, 1, 0, -1, 0, 400.01, 0.00, 10, 0.08]

# Load Pointcloud
cloud_in = readCloud(pc_in, filename)

# Estimate Normals
cloud_in.estimate_normals()

# Perform RANSAC to find largest planar patch
# Set values
distance_threshold = 0.01
ransac_n = 3
num_iterations = 1000

plane_model, inliers = cloud_in.segment_plane(
    distance_threshold, ransac_n, num_iterations
)

# Set different colors for the found planar patch vs. rest of the cloud
cloud_colored = colorInlierOutlier(cloud_in, inliers)

# Visualize colored result
viewTitle = "RANSAC"
cloudVisualizer([cloud_colored], viewTitle, winDims, viewCtrl, 1)
o3d.io.write_point_cloud(
    "/Volumes/T7 Shield/AdvancedGIS/read_test/UZH_dataset/processed_2C05_1.ply",
    cloud_colored,
)
