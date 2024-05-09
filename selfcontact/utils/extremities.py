import numpy as np
import torch

smplx_extremities_nums = [1,2,4,5,7,8,10,11,16,17,18,19,
    20,21,25,26,27,28,29,30,31,32,33,34,35,36,37,38,
    39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54
]
smplx_inner_toes = [5779, 5795, 5796, 5797, 5798, 5803, 5804, 5814, 5815, 
    5819, 5820, 5823, 5824, 5828, 5829, 5840, 5841, 5852, 5854, 5862, 5864, 
    8472, 8473, 8489, 8490, 8491, 8492, 8497, 8498, 8508, 8509, 8513, 8514,
    8517, 8518, 8522, 8523, 8531, 8532, 8533, 8534, 8535, 8536, 8546, 8548, 
    8556, 8558, 8565
]

def get_extremities(segmentation_path, include_toes=False):

    # extrimities vertex IDs
    smpl_bone_vert = np.load(segmentation_path)

    extremities = np.where(np.isin(smpl_bone_vert, smplx_extremities_nums))[0]
    # ignore inner toe vertices by default
    if not include_toes:
        extremities = extremities[~np.isin(extremities, smplx_inner_toes)]
    extremities = torch.tensor(extremities)

    return extremities

def get_extremities_assign_bones(segmentation_path, smplx_extremities_assign_nums, include_toes=False):

    # extrimities vertex IDs
    smpl_bone_vert = np.load(segmentation_path) # [10475]

    extremities = np.where(np.isin(smpl_bone_vert, smplx_extremities_assign_nums))[0]
    # ignore inner toe vertices by default
    if not include_toes:
        extremities = extremities[~np.isin(extremities, smplx_inner_toes)]
    extremities = torch.tensor(extremities)

    return extremities

def get_vertices_assign_bone(segmentation_path, bone_idx):
    """
    获取特定骨骼的顶点
    """
    smpl_bone_vert = np.load(segmentation_path) # [10475]
    bone_verts = np.where(smpl_bone_vert == bone_idx)
    return bone_verts[0]

def get_vertices_assign_bones(segmentation_path, bone_idxs):
    """
    获取指定的多个骨骼的顶点
    """
    smpl_bone_vert = np.load(segmentation_path) # [10475]
    bone_verts_all = []
    for idx in bone_idxs:
        bone_verts = np.where(smpl_bone_vert == idx)
        bone_verts_all.append(bone_verts)
    return bone_verts_all

# 获取骨骼（上下）边界的三角面索引
def find_connected_faces(start_index,connected_sets,processed_indices,bound_faces):
    # 创建一个新的集合来存储与目标三角面相互连接的三角面索引
    connected_set = set([start_index])
    
    # 创建一个队列，用于存储待处理的三角面索引
    queue = [start_index]
    
    while queue:
        # 弹出队列中的第一个元素作为当前目标三角面的索引
        current_index = queue.pop(0)
        
        # 将当前目标三角面设置为已处理
        processed_indices.add(current_index)
        
        # 获取当前目标三角面的顶点索引
        current_face = bound_faces[current_index]
        
        # 创建一个集合来存储当前目标三角面的顶点索引，方便后续比较
        current_face_set = set(current_face)
        
        # 遍历剩余的三角面
        for i, other_face in enumerate(bound_faces):
            # 如果这个三角面已经被处理过了，就跳过
            if i in processed_indices:
                continue
            
            # 检查当前目标三角面与其他三角面的顶点索引是否有交集
            if any(v in current_face_set for v in other_face):
                # 如果有交集，则将其添加到 connected_set 中
                connected_set.add(i)
                
                # 将与当前目标三角面相互连接的三角面的索引添加到队列中，等待处理
                queue.append(i)
                
                # 将与当前目标三角面相互连接的三角面的索引添加到集合中，方便后续比较
                current_face_set.update(other_face)
    
    # 将当前连接的三角面集合添加到 connected_sets 中
    connected_sets.append(list(connected_set))

def find_connected_faces_share(start_index, connected_sets, shared_vertices, processed_indices, bound_faces, vertices):
    # 创建一个新的列表来存储与目标三角面相互连接的三角面索引
    # connected_set = [start_index]
    
    # 创建一个队列，用于存储待处理的三角面索引

    queue = [start_index]
    # shared_vertices = []
    # for cur, other_face in enumerate(bound_faces):
    while queue:
        
        # 弹出队列中的第一个元素作为当前目标三角面的索引
        # current_index = cur
        
        current_index = queue.pop(0)
        
        processed_indices.add(current_index)
        
        # 获取当前目标三角面的顶点索引
        current_face = bound_faces[current_index]
        
        # 创建一个集合来存储当前目标三角面的顶点索引，方便后续比较
        current_face_set = set(current_face)
        
        # 创建一个列表用于存储当前目标三角面共享的顶点索引
        cur_shared_vertices = []
        
        # 遍历剩余的三角面
        for i, other_face in enumerate(bound_faces):
            # 如果是当前目标三角面本身，则跳过
            if i == current_index:
                continue
            # 检查当前目标三角面与其他三角面的顶点索引是否有交集
            if any(v in current_face_set for v in other_face):
                if i not in connected_sets:
                    queue.append(i)
                    connected_sets.append(i)
                # 将与当前目标三角面相互连接的三角面的顶点索引与当前目标三角面的顶点索引求交集
                shared_vertex = list(set(current_face).intersection(other_face))
                if len(shared_vertex) == 1:
                    cur_shared_vertices.append(shared_vertex[0])
                elif len(shared_vertex) == 2:
                    # 如果有两个共享的顶点，取其中模长较大的那个
                    sv = shared_vertex[0] if np.linalg.norm(vertices[0][shared_vertex[0]].cpu().detach().numpy()) > np.linalg.norm(vertices[0][shared_vertex[1]].cpu().detach().numpy()) else shared_vertex[1]
                    cur_shared_vertices.append(sv)
        
        if len(set(cur_shared_vertices)) == 2:
            shared_vertices.append(list(set(cur_shared_vertices)))
        elif len(set(cur_shared_vertices)) == 3:
            shared_vertices.append(list(set(cur_shared_vertices))[:2])
            shared_vertices.append(list(set(cur_shared_vertices))[1:])
        elif len(set(cur_shared_vertices)) == 1:
            # shared_vertices.append(cur_shared_vertices)
            print("Num Of Vertices == 1")
        elif len(set(cur_shared_vertices)) > 3:
            print("Num Of Vertices > 3")
        # 将当前目标三角面对应的两个共享顶点索引添加到 connected_sets 中
        # connected_sets.append(shared_vertices)
        
def get_connected_faces(vertices, bound_faces, faces):
    # 创建一个集合，用于存储已经被处理的三角面的索引
    processed_indices = set()
    
    vertices_wt = vertices
    bone_faces= bound_faces
    # 遍历所有三角面
    for i, face in enumerate(bound_faces):
        connected_sets = [i]
        shared_vertices = []
        
        # 如果这个三角面已经被处理过了，就跳过
        if i in processed_indices:
            continue
        
        # 将当前未处理的三角面设置为目标三角面，并进行处理
        find_connected_faces_share(i, connected_sets, shared_vertices , processed_indices, bound_faces, vertices)

        shared_vertices = np.array(shared_vertices)
        
        # 计算平均顶点位置
        # mean_vertex = np.mean(vertices[np.array(connected_sets)], axis=0)
        mean_vertex = torch.mean(vertices[:,np.unique(shared_vertices.flatten()),:], 1, # 嘴巴部位的平均顶点
                        keepdim=True)
        vertices_wt = torch.cat((vertices_wt, mean_vertex), 1)
        
        faces_wt = [[shared_vertices[i][0], shared_vertices[i][1],
                len(vertices_wt[0]) - 1] for i in range(len(shared_vertices))]
        # faces_wt = torch.tensor(np.array(faces_wt).astype(np.int64),dtype=torch.long,device=faces.device)
        # faces_wt = torch.cat((bound_faces, faces_wt), 0)        
        faces_wt = np.array(faces_wt).astype(np.int64)
        bone_faces = np.concatenate((bone_faces, faces_wt), 0)
        
    return vertices_wt,bone_faces
        # shared_vertices与mean_vertex组成新的三角面
        
    # find_connected_faces_share(shared_vertices,bound_faces)
    
    # shared_vertices = np.array(shared_vertices)
    # connected_faces_idx = [bound_faces[idxs] for idxs in connected_sets]
    return vertices_wt,faces_wt