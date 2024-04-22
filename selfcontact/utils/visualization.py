import numpy as np
import os
import trimesh
    
def save_smplx_mesh_bool(model, body, exterior, file_name = 'result', cfg = None, path = './output'):
    '''
    body是smplx模型的输出，exterior是一个布尔数组，表示每个顶点是否在模型外部
    '''
    mesh = trimesh.Trimesh(body.vertices[0].detach().cpu().numpy(), model.faces)
    verts_exterior_np = exterior.cpu().numpy()

    # color vertices inside red, outside green
    color = np.array(mesh.visual.vertex_colors)
    # 定义两种颜色
    color_false = np.array([255, 0, 0, 255], dtype=np.uint8)
    color_true = np.array([233, 233, 233, 255], dtype=np.uint8)
    
    # 使用verts_exterior_np布尔数组来更新color数组
    color[verts_exterior_np] = color_true
    color[~verts_exterior_np] = color_false

    # color[~verts_exterior_np[0], :] = [255, 0, 0, 255] # 内部红色
    # color[verts_exterior_np[0], :] = [0, 255, 0, 255]# [233, 233, 233, 255] # 外部绿色
    mesh.visual.vertex_colors = color

    # export mesh
    # 计算path目录中有多少文件
    file_path = os.path.join(cfg.output_folder, file_name)
    if os.path.exists(file_path) == False:
        os.makedirs(file_path, exist_ok=True)
    num = len(os.listdir(file_path))
    path =os.path.join(file_path, "{}_{}.obj".format(file_name,num))
    mesh.export(path)
    print("save mesh to %s" % path)
    
def save_smplx_mesh_index(model, body, extremities, path = './output'):
    '''
    body是smplx模型的输出，extremities是一个索引数组，表示每个顶点是否在模型外部
    '''
    mesh = trimesh.Trimesh(body.vertices[0].detach().cpu().numpy(), model.faces)
    # verts_exterior_np = extremities.cpu().numpy()

    # color vertices inside red, outside green
    colors = np.array(mesh.visual.vertex_colors)
    # 定义两种颜色
    color_extremity = np.array([255, 0, 0, 255], dtype=np.uint8)  # 灰色
    color_non_extremity = np.array([233, 233, 233, 255], dtype=np.uint8)  # 红色

    # 使用 extremities 数组进行条件索引，更新 colors 数组
    colors[extremities] = color_extremity
    colors[~np.isin(np.arange(len(colors)), extremities)] = color_non_extremity

    # color[~verts_exterior_np[0], :] = [255, 0, 0, 255] # 内部红色
    # color[verts_exterior_np[0], :] = [0, 255, 0, 255]# [233, 233, 233, 255] # 外部绿色
    mesh.visual.vertex_colors = colors

    # export mesh
    # 计算path目录中有多少文件
    if not path.endswith('.obj'):
        files = os.listdir(path)
        num = len(files)
        path = path + "/bone_%d.obj" % num
    mesh.export(path)
    print("save mesh to %s" % path)
    
def save_npz_file(output_folder, npz_file_orig, total_body_mesh, file_name, cfg):
    data_orig = np.load(npz_file_orig, allow_pickle=True)
    data_dict = dict(data_orig) # 需要拷贝一份才能修改
    
    body_poses = [cur_body.body_pose.clone().detach() for cur_body in total_body_mesh]

    if 'assign_frame_idx' in cfg and cfg.assign_frame_idx >= 0: # 修改指定帧的Pose
        data_dict['poses'][cfg.assign_frame_idx,3:66] = body_poses[0].cpu().detach().numpy().flatten().tolist()
    else: # 修改每帧对应的Pose
        frame_start = cfg.frame_start
        frame_end = cfg.frame_end
        for idx, cur_frame in enumerate(range(frame_start, frame_end+1)):
            pose_cpu = body_poses[idx].cpu().detach().clone()
            pose_numpy = pose_cpu.numpy().flatten().tolist()
            data_dict['poses'][cur_frame,3:66] = pose_numpy

    output_dir = os.path.join(output_folder, file_name ,'Anim')
    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir, exist_ok=True)
    file_count = len(os.listdir(output_dir))
    file_name = '{}_{}.npz'.format(file_name, file_count+1)
    output_dir = os.path.join(output_dir, file_name)
    np.savez_compressed(output_dir, **data_dict)
    print("result file save in {}".format(output_dir))