import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import sys


def readCloud(path, name):
    print("Reading Cloud titled: '" + name + "'.")
    pcd = o3d.io.read_point_cloud(path + name, remove_nan_points=True)
    print(pcd) if len(np.asarray(pcd.points)) > 0 else print("PointCloud empty.")
    return pcd


def colorInlierOutlier(fullCloud, indices):
    Inlier = fullCloud.select_by_index(indices)
    Inlier.paint_uniform_color([1, 0, 0])  # light grey
    Outlier = fullCloud.select_by_index(indices, invert=True)
    Outlier.paint_uniform_color([0.8, 0.8, 0.8])  # pure red
    ColoredCloud = Inlier + Outlier  # re-combine both clouds to one output
    print("Inlier: ")
    print(Inlier)
    print("Outlier: ")
    print(Outlier)
    return ColoredCloud


def cloudVisualizer(cloud, name, winDims, viewCtrl, normalTF):
    visual = o3d.visualization.Visualizer()
    visual.create_window(
        window_name=name, width=winDims[0], height=winDims[1], visible=True
    )
    for c in cloud:
        visual.add_geometry(c)

    renderOpts = visual.get_render_option()
    renderOpts.show_coordinate_frame = True
    rgb = np.asarray([53, 57, 62])
    renderOpts.background_color = rgb / 255
    renderOpts.point_size = 2
    if normalTF == 0:
        renderOpts.point_show_normal = True
    else:
        renderOpts.point_show_normal = False

    view_ctrl = visual.get_view_control()
    view_ctrl.set_up(viewCtrl[0:3])
    view_ctrl.set_front(viewCtrl[3:6])
    view_ctrl.set_lookat(viewCtrl[6:9])
    view_ctrl.set_zoom(viewCtrl[9])

    visual.run()
    visual.destroy_window()
    print("Viewer '" + name + "' terminated.")


def keypoints2spheres(keypoints, radius, color):
    spheres = o3d.geometry.TriangleMesh()
    for i in keypoints.points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(i)
        spheres += sphere
    spheres.paint_uniform_color((color) / 255)
    return spheres


def progressbar(it, pf="", s=60, out=sys.stdout):
    ct = len(it)

    def show(j):
        x = int(s * j / ct)
        tmp = " "
        print(
            f"{pf}[{u'█'*x}{('.'*(s-x))}] {j}/{ct} {tmp}",
            end="\r",
            file=out,
            flush=True,
        )

    show(0.1)  # avoid div/0
    for i, item in enumerate(it):
        yield item
        show(i + 1)
    print("\n", flush=True, file=out)
