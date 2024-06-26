"""
Compute and store bounding boxes for object parts
    
Author: Tengyu Liu
Copyright, 2021, Tengyu Liu
"""

import numpy as np
import trimesh as tm
from scipy.spatial import ConvexHull
from pyquaternion.quaternion import Quaternion as Q


def get_bbox(obj_hull):
    """
    Intuition: An minimum obb should have at least one face aligned with a convex hull face
    Solution: For each convex hull face, de-rotate hull and compute aabb. Keep the one with smallest volume
    return: _ctr, _vec, _min, _max
    """
    # TODO
    # For each face, rotate object so that the face is parallel to X-Y plane. Mark rotation as Q1. 
    #     Mark the distance from the face to X-Y plane as T1 (Z-axis).
    #     Mark (the largest distance from vertex to X-Y plane - T1) as H
    #     Project all vertices onto X-Y plane, compute convex hull
    #     For each edge of the hull, rotate projection so that the edge is parallel to X-axis, mark rotation as Q2 (Z-axis rotation only). 
    #         Mark the distance from the edge to X-axis as T2 (Y-axis rotated by Q1)
    #         Mark (the largest distance from vertex to X-axis - T2) as W
    #         Project all vertices onto X-axis, mark the largest distance between vertices as D
    #         (D, W, H) is the obb extents
    #         Q1.inverse * Q2.inverse is the obb rotation
    #         T1 + Q1(T2) is the obb centroid

    obj_vertices = obj_hull.vertices[obj_hull.faces].reshape([-1,3])
    # xy_plane = tm.primitives.Box(extents=[1, 1, 0])
    best_obb_volume = float('inf')
    best_obb = None
    for face in obj_hull.faces:
        triangle = obj_hull.vertices[face]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        normal /= np.linalg.norm(normal)
        Q1 = compute_quaternion_from_vectors([0, 0, 1], normal).inverse
        q1_vertices = np.matmul(Q1.rotation_matrix, obj_hull.vertices.T).T
        T1 = q1_vertices[:,2].min()
        H = q1_vertices[:,2].max() - T1
        xy_projection = q1_vertices[:,:2]
        xy_projection_hull = ConvexHull(xy_projection)
        for i_edge in range(len(xy_projection_hull.vertices)):
            edge_0 = xy_projection[xy_projection_hull.vertices[i_edge-1], :]
            edge_1 = xy_projection[xy_projection_hull.vertices[i_edge], :]
            z_rotation = np.arctan((edge_1[1] - edge_0[1]) / (edge_1[0] - edge_0[0]))
            Q2 = Q(axis=[0,0,1], radians=z_rotation).inverse
            q2_vertices = np.matmul(Q2.rotation_matrix, q1_vertices.T).T
            T2 = q2_vertices[:,1].min()
            W = q2_vertices[:,1].max() - T2
            T3 = q2_vertices[:,0].min()
            D = q2_vertices[:,0].max() - T3
            volume = D * H * W
            if volume < best_obb_volume:
                best_obb_volume = volume
                best_obb = T1,T2,T3,D,W,H,Q1.inverse * Q2.inverse
    return best_obb

def compute_quaternion_from_vectors(v1, v2):
    dot = np.dot(v1, v2)
    if dot < -0.999999:
        quaternion = [0,0,1,0]
    elif dot > 0.999999:
        quaternion = [1,0,0,0]
    else:
        quaternion = np.zeros([4])
        quaternion[1:] = np.cross(v1, v2)
        quaternion[0] = 1 + dot
        quaternion /= np.linalg.norm(quaternion)
    return Q(quaternion)

if "__main__" == __name__:
    obj = tm.load_mesh("/usr/pydata/t2m/selfcontact/output/extremities/bone_0.obj")
    obj_hull = obj.convex_hull
    obb_params  = get_bbox(obj_hull)
    # 创建长方体
    obb_center = obb_params[0:3]
    obb_extents = obb_params[3:6]  # 长度在索引3,4,5处
    obb_rotation = obb_params[6]
    
    # 从obb_rotation创建一个四元数对象
    obb_quaternion = Q(obb_rotation)

    # 初始化一个单位的4x4齐次变换矩阵
    transform_matrix = np.eye(4)

    # 将原始局部坐标系的原点移动到OBB的中心点位置
    transform_matrix[:3, 3] = obb_center

    # 将局部坐标系根据OBB的旋转进行旋转
    rotation_matrix = obb_quaternion.rotation_matrix
    transform_matrix[:3, :3] = rotation_matrix
    
    obb_box = tm.primitives.Box(extents=obb_extents, transform=transform_matrix)

    # 保存OBB为obj文件
    # obb_save_path = "/usr/pydata/t2m/selfcontact/output/tmp/obb_0.obj"
    # obb_box.export(obb_save_path)
    
    # 将OBB长方体的顶点和面添加到原始mesh中
    combined_vertices = np.vstack([obj.vertices, obb_box.vertices])
    combined_faces = np.vstack([obj.faces, obb_box.faces + len(obj.vertices)])

    # 创建合并后的mesh对象
    combined_mesh = tm.Trimesh(vertices=combined_vertices, faces=combined_faces)
    
    # 保存合并后的mesh为obj文件
    save_path = "/usr/pydata/t2m/selfcontact/output/tmp/obb_2.obj"
    combined_mesh.export(save_path)
    