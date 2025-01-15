import open3d as o3d
import numpy as np
import os

from itertools import combinations
from scipy.spatial import ConvexHull

def fit_plane_ransac(point_cloud, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
    """Fit a plane to a point cloud using RANSAC."""
    plane_model, inliers = point_cloud.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )
    inlier_cloud = point_cloud.select_by_index(inliers)
    return plane_model, inlier_cloud

def compute_bounding_rectangle(points_3d, plane_normal):
    """Compute a bounding rectangle that encloses the convex hull."""
    # Step 1: Compute the centroid of the points
    centroid = np.mean(points_3d, axis=0)
    
    # Step 2: Define a local 2D coordinate system on the plane
    u = np.cross(plane_normal, np.array([0, 0, 1]))
    if np.linalg.norm(u) < 1e-6:  # Handle case when plane is parallel to Z-axis
        u = np.cross(plane_normal, np.array([0, 1, 0]))
    u = u / np.linalg.norm(u)
    v = np.cross(plane_normal, u)
    
    # Step 3: Project points into the local 2D coordinate system
    points_2d = np.dot(points_3d - centroid, np.column_stack((u, v)))
    
    # Step 4: Compute the convex hull in 2D
    hull = ConvexHull(points_2d)
    
    # Step 5: Find the bounding rectangle of the convex hull
    min_x, min_y = np.min(points_2d[hull.vertices], axis=0)
    max_x, max_y = np.max(points_2d[hull.vertices], axis=0)
    
    # Step 6: Define the rectangle corners in 2D
    rectangle_2d = np.array([
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y]
    ])
    
    # Step 7: Map the 2D rectangle corners back to 3D
    rectangle_3d = centroid + rectangle_2d[:, 0][:, None] * u + rectangle_2d[:, 1][:, None] * v
    return rectangle_3d, plane_normal

    
    # Step 7: Map the 2D rectangle corners back to 3D
    rectangle_3d = centroid + np.dot(rectangle_2d, np.column_stack((u, v)))
    return rectangle_3d, plane_normal

def create_mesh_from_rectangle(rectangle_3d, plane_normal, thickness=0.05):
    """Create a rectangular mesh for a wall with thickness."""
    # Extrude the rectangle to create thickness
    vertices_top = rectangle_3d
    vertices_bottom = rectangle_3d - thickness * plane_normal
    
    # Combine top and bottom vertices
    vertices = np.vstack((vertices_top, vertices_bottom))
    num_vertices = len(vertices_top)
    
    # Create side faces
    triangles = []
    for i in range(num_vertices):
        next_i = (i + 1) % num_vertices
        triangles.append([i, next_i, num_vertices + i])
        triangles.append([next_i, num_vertices + next_i, num_vertices + i])
    
    # Create top and bottom faces
    top_faces = [[0, 1, 2], [0, 2, 3]]
    bottom_faces = [[num_vertices + 0, num_vertices + 2, num_vertices + 1],
                    [num_vertices + 0, num_vertices + 3, num_vertices + 2]]
    triangles.extend(top_faces)
    triangles.extend(bottom_faces)
    
    # Create the mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    return mesh

def load_point_clouds(ply_files):
    """Load multiple .ply files as Open3D point clouds."""
    point_clouds = []
    for file in ply_files:
        pcd = o3d.io.read_point_cloud(file)
        point_clouds.append(pcd)
    return point_clouds

def merge_obj_files(input_dir, output_file="merged_walls.obj"):
    """Merge multiple .obj files into a single .obj file."""
    vertices = []
    faces = []
    vertex_offset = 0
    
    # Step 1: Read all .obj files in the input directory
    obj_files = [f for f in os.listdir(input_dir) if f.endswith(".obj")]
    for obj_file in obj_files:
        mesh = o3d.io.read_triangle_mesh(os.path.join(input_dir, obj_file))
        
        # Append vertices
        vertices.extend(np.asarray(mesh.vertices))
        
        # Append faces, adjusting indices with vertex offset
        faces.extend(np.asarray(mesh.triangles) + vertex_offset)
        
        # Update vertex offset
        vertex_offset += len(mesh.vertices)
    
    # Step 2: Create a new mesh with merged vertices and faces
    merged_mesh = o3d.geometry.TriangleMesh()
    merged_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    merged_mesh.triangles = o3d.utility.Vector3iVector(faces)
    
    # Compute normals for the merged mesh
    merged_mesh.compute_vertex_normals()
    merged_mesh.compute_triangle_normals()
    
    # Step 3: Write the merged mesh to a single .obj file
    o3d.io.write_triangle_mesh(output_file, merged_mesh)
    print(f"Merged .obj file saved to {output_file}")

def main(ply_files, output_dir="output_models", visualize=False):
    point_clouds = load_point_clouds(ply_files)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    meshes = []
    for i, pc in enumerate(point_clouds):
        plane_model, inlier_cloud = fit_plane_ransac(pc)
        print(f"Wall {i + 1}: Plane model = {plane_model}")
        
        rectangle_3d, plane_normal = compute_bounding_rectangle(np.asarray(inlier_cloud.points), plane_model[:3])
        wall_mesh = create_mesh_from_rectangle(rectangle_3d, plane_normal)
        
        # 计算顶点和三角形法向量
        wall_mesh.compute_vertex_normals()
        wall_mesh.compute_triangle_normals()
        
        # 保存 .obj 文件，包含法向量信息
        output_path = os.path.join(output_dir, f"wall_{i + 1}.obj")
        o3d.io.write_triangle_mesh(output_path, wall_mesh, write_triangle_uvs=True)
        print(f"Wall {i + 1} mesh saved to {output_path}")
        
        meshes.append(wall_mesh)
    

    
    # Visualize point clouds and wall models
    o3d.visualization.draw_geometries(point_clouds, window_name="3D Wall Reconstruction")
    o3d.visualization.draw_geometries(meshes, window_name="Wall Models")

if __name__ == "__main__":
    ply_files = [
        "wall0_clean.ply", "wall1_clean.ply", "wall2_clean.ply", "wall3_clean.ply",
        "wall4_clean.ply", "wall5_clean.ply", "wall6_clean.ply", "wall7_clean.ply",
        "wall8_clean.ply", "wall9_clean.ply", "wall10_clean.ply", "wall11_clean.ply",
        "wall12_clean.ply", "wall13_clean.ply", "wall15_clean.ply"
    ]
    main(ply_files)

    input_dir = "output_models"  # Directory containing individual .obj files
    merge_obj_files(input_dir)
