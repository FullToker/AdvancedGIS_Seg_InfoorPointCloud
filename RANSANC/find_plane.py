from main import (
    readCloud,
    colorInlierOutlier,
    cloudVisualizer,
    keypoints2spheres,
    progressbar,
)
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np

"""
pc_in = "/Volumes/T7 Shield/AdvancedGIS/read_test/UZH_dataset/2C05/"
pc_out = ""
filename = "2C05 - downsample.ply"
"""
root_path = "/Volumes/T7 Shield/AdvancedGIS/read_test/"
in_ply = "GE006_downsample.ply"
output_file = "plane_GE006_downsample.ply"

winDims = [1000, 1000]
viewCtrl = [0, 0, 1, 0, -1, 0, 400.01, 0.00, 10, 0.08]

# Load Pointcloud
cloud_in = readCloud(root_path, in_ply)

# Estimate Normals
cloud_in.estimate_normals()

max_plane_idx = 6  # 6 planes
distance_threshold = 0.01
ransac_n = 3
num_iterations = 1000
min_points = 50


segment_models = {}  # stores the model parameters [a,b,c,d] for each plane
segments = {}  # will form a dictionary of planar segments found by RANSAC
rest = cloud_in  # input cloud per iteration, initialized with the full cloud

# iteration also using the progressbar - to deacttivate, replace the loop call with "for i in range(max_plane_idx)"
for i in progressbar(range(max_plane_idx), "Computing Plane: ", 40):

    colors = plt.get_cmap("Set1")(i)  # create color pallet

    segment_models[i], inliers = rest.segment_plane(
        distance_threshold, ransac_n, num_iterations
    )
    segments[i] = rest.select_by_index(inliers)
    segments[i].paint_uniform_color(list(colors[:3]))
    rest = rest.select_by_index(
        inliers, invert=True
    )  # select Outliers to create the input for the next iterative step

labels = np.array(
    rest.cluster_dbscan(eps=distance_threshold * 5, min_points=round(min_points / 2))
)  # call DBSCAN function for rest - note that you should adjust the parameters in the function call: ps=distance_threshold*5, min_points=round(min_points/2)

max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")

# Colorize found DBSCAN patches as before, again with setting 0 to all outliers
"""
colors = plt.get_cmap("Set3")(
    labels / (max_label if max_label > 0 else 1)
)  # create color pallet as before in DBSCAN
colors[labels < 0] = 0  # uniform color for outliers, see before
rest.colors = o3d.utility.Vector3dVector(colors[:, :3])

# visualize remaining Outliers as spheres using the keypoints2spheres function
outs = rest.select_by_index(np.where(labels < 0)[0])  # find outliers
out_sphere = keypoints2spheres(
    outs, 0.025, np.array([255, 0, 0])
)  # create colorized sphere for each outlier point
"""

"""
cloudVisualizer(
    [segments[i] for i in range(max_plane_idx)] + [rest] + [out_sphere],
    "Combined Segmentation",
    winDims,
    viewCtrl,
    1,
)  # SOLUTION OPTIONAL
"""


# write the result into the files
combined_cloud = o3d.geometry.PointCloud()
for i in range(max_plane_idx):
    combined_cloud += segments[i]
combined_cloud += rest
# combined_cloud += out_sphere

o3d.io.write_point_cloud(
    root_path + output_file,
    combined_cloud,
)

print(f"Combined point cloud saved as: {output_file}")
