import open3d as o3d
import numpy as np

def main():
    print("Testing mesh in Open3D...")
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.6, origin=[0.0,0.0,0.0], )
    mesh = o3d.io.read_triangle_mesh("../datasets/PROX/scenes/BasementSittingBooth.ply")
    print(mesh)
    print('Vertices:')
    print(np.asarray(mesh.vertices).shape)
    print('Triangles:')
    print(np.asarray(mesh.triangles).shape)

    print("Try to render a mesh with normals (exist: " +
          str(mesh.has_vertex_normals()) + ") and colors (exist: " +
          str(mesh.has_vertex_colors()) + ")")
    o3d.visualization.draw_geometries([mesh+mesh_frame],mesh_show_back_face = True)
    print("A mesh with no normals and no colors does not look good.")





if __name__ == '__main__':
    main()