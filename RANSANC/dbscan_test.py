from main import (
    readCloud,
    colorInlierOutlier,
    cloudVisualizer,
    keypoints2spheres,
    progressbar,
)
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

pc_in = "/Volumes/T7 Shield/AdvancedGIS/read_test/UZH_dataset/"
pc_out = ""
filename = "01164_GE006.ply"

winDims = [1000, 1000]
viewCtrl = [0, 0, 1, 0, -1, 0, 400.01, 0.00, 10, 0.08]

# ------------------------------------------------------############################################
fn2 = "processed_2C05_1.ply"
sample = readCloud(pc_in, fn2)

epsilon = 0.06  # TODO
min_points = 6  # TODO

# DBSCAN
labels = np.array(sample.cluster_dbscan(eps=epsilon, min_points=min_points))
max_label = labels.max()

# Assing a unique color to each found patch
colors = plt.get_cmap("tab20")(
    labels / (max_label if max_label > 0 else 1)
)  # Creates a set of colors following the given matplotlib colormap (here: tab20)
colors[labels < 0] = 0  # Set uniform color for outliers
sample.colors = o3d.utility.Vector3dVector(colors[:, :3])

# Visualize Result
viewTitle = "DBSCAN"
cloudVisualizer([sample], viewTitle, winDims, viewCtrl, 1)
o3d.io.write_point_cloud(
    "/Volumes/T7 Shield/AdvancedGIS/read_test/UZH_dataset/dbscan_2C05_1.ply", sample
)
