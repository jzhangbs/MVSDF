import argparse
from email.policy import default
import open3d as o3d
import numpy as np
import os
from mesh_cut_ext import mesh_cut

p = argparse.ArgumentParser()
p.add_argument('in_file', type=str)
p.add_argument('out_file', type=str)
p.add_argument('--thresh', type=int, default=15)
p.add_argument('--smooth', type=int, default=10)
args = p.parse_args()

mesh_o = o3d.io.read_triangle_mesh(args.in_file)
mesh = o3d.geometry.HalfEdgeTriangleMesh.create_from_triangle_mesh(mesh_o)

hemesh_vertices = mesh.vertices
hemesh_triangles = mesh.triangles
face_vidxs = np.asarray(hemesh_triangles)
num_face = face_vidxs.shape[0]
vertex_colors = np.asarray(mesh.vertex_colors)
face_colors = vertex_colors[face_vidxs]
face_colors = face_colors[:,:,0].mean(axis=1)

color_thresh = args.thresh/255

smooth_cap = args.smooth
hes = mesh.half_edges
non_boundary_hes = [he for he in hes if not he.is_boundary() and not hes[he.twin].is_boundary()]
# unique_edges = {frozenset((he.triangle_index, hes[he.twin].triangle_index)) for he in non_boundary_hes}
unique_edges = [(he.triangle_index, hes[he.twin].triangle_index) for he in non_boundary_hes]
unique_edges = np.array([[i for i in e] + [smooth_cap] for e in unique_edges], dtype=np.uint32)

result = mesh_cut(face_colors>color_thresh, unique_edges)
mask = result.nonzero()[0]

print(f'[trim] num faces from {num_face} to {num_face-len(mask)}')

mesh_c = o3d.geometry.TriangleMesh(hemesh_vertices, hemesh_triangles)
mesh_c.remove_triangles_by_index(mask)
mesh_c = mesh_c.remove_unreferenced_vertices()
o3d.io.write_triangle_mesh(args.out_file, mesh_c)
