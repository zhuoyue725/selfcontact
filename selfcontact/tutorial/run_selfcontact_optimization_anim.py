# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2021 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import copy
import sys
import time
sys.path.append('/usr/pydata/t2m/selfcontact')

from selfcontact import SelfContact
import torch 
import trimesh 
import os.path as osp
import numpy as np
import argparse
import os 
import smplx
import yaml
import pickle
from selfcontact.losses import SelfContactOptiAnimAssignLoss
from selfcontact.fitting import SelfContactAnimOpti
from selfcontact.utils.parse import DotConfig
from selfcontact.utils.body_models import get_bodymodels
from selfcontact.utils.extremities import get_extremities,get_extremities_assign_bones,smplx_extremities_nums,get_vertices_assign_bone
from selfcontact.utils.visualization import save_smplx_mesh_with_obb, save_npz_file
from selfcontact.utils.SMPLXINFO import *
from selfcontact.utils.obb import build_obb
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True

# code contains non-deterministic parts, that may lead to
# different results when running the same script again. To
# run the determinisitc version use the following lines:
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
# torch.use_deterministic_algorithms(True) # 需要 torch >= 1.10.0

# def save_npz_file(output_folder, npz_file_orig, total_body_mesh, file_name, assign_frame_idx=-1):
#     data_orig = np.load(npz_file_orig, allow_pickle=True)
#     data_dict = dict(data_orig) # 需要拷贝一份才能修改
    
#     body_poses = [cur_body.body_pose.clone().detach() for cur_body in total_body_mesh]

#     if assign_frame_idx >= 0: # 修改特定帧对应的Pose
#         data_dict['poses'][assign_frame_idx,3:66] = body_poses[0].cpu().detach().numpy().flatten().tolist()
#     else: # 修改每帧对应的Pose
#         for i, pose in enumerate(body_poses):
#             pose_cpu = pose.cpu()
#             pose_numpy = pose_cpu.detach().numpy().flatten().tolist()
#             data_dict['poses'][i,3:66] = pose_numpy

#     file_count = len(os.listdir(output_folder))
#     file_name = '{}_{}.npz'.format(file_name, file_count)
#     file_path = os.path.join(output_folder, file_name)
#     np.savez_compressed(file_path, **data_dict)
#     print("result file save in {}".format(file_path))
# def save_npz_file(output_folder, npz_file_orig, total_body_mesh, file_name, cfg):
#     data_orig = np.load(npz_file_orig, allow_pickle=True)
#     data_dict = dict(data_orig) # 需要拷贝一份才能修改
    
#     body_poses = [cur_body.body_pose.clone().detach() for cur_body in total_body_mesh]

#     if 'assign_frame_idx' in cfg and cfg.assign_frame_idx >= 0: # 修改指定帧的Pose
#         data_dict['poses'][cfg.assign_frame_idx,3:66] = body_poses[0].cpu().detach().numpy().flatten().tolist()
#     else: # 修改每帧对应的Pose
#         frame_start = cfg.frame_range.frame_start
#         frame_end = cfg.frame_range.frame_end
#         for idx, cur_frame in enumerate(range(frame_start, frame_end+1)):
#             pose_cpu = body_poses[idx].cpu().detach().clone()
#             pose_numpy = pose_cpu.numpy().flatten().tolist()
#             data_dict['poses'][cur_frame,3:66] = pose_numpy

#     file_count = len(os.listdir(output_folder))
#     file_name = '{}_{}.npz'.format(file_name, file_count)
#     file_path = os.path.join(output_folder, file_name)
#     np.savez_compressed(file_path, **data_dict)
#     print("result file save in {}".format(file_path))
    
def load_model_param(npz_file, cfg, device='cuda'):
    assign_frame_idx = cfg.assign_frame_idx if 'assign_frame_idx' in cfg else -1
    models = get_bodymodels(
        model_path=cfg.model_folder, 
        model_type=cfg.body_model.model_type, 
        device=device,
        batch_size=cfg.batch_size, 
        num_pca_comps=cfg.body_model.num_pca_comps
    )
    
    # load data
    data = np.load(npz_file, allow_pickle=True)
    if 'mocap_frame_rate' in data.keys(): # 载入动画序列的某一帧
        fps = data['mocap_frame_rate'].item()
    elif 'mocap_framerate' in data.keys():
        fps = data['mocap_framerate'].item()
    if assign_frame_idx >= 0: # 仅添加特定帧
        frame_num = 1
        frame_start = 0
        frame_end = 0
    else:
        if 'frame_start' in cfg:
            frame_start = cfg.frame_start
            frame_end = cfg.frame_end
            frame_num = frame_end - frame_start + 1
        else:
            frame_num = data['poses'].shape[0] if 'poses' in data.keys() else 1
    
    print('frame_num: ', frame_num , 'fps: ', fps)
    
    if data['gender'].shape.__len__() > 1:
        gender = data['gender'][0].decode("utf-8")
    else:
        gender = data['gender'].item()
    
    body = models[gender]
        
    if len(data['betas']) > 10:
        betas = torch.from_numpy(data['betas'][:10]).unsqueeze(0).float()
    else:
        betas = torch.from_numpy(data['betas']).unsqueeze(0).float()
    # global_orient = torch.zeros(1,3) if 'global_orient' not in data.keys() \
    #     else data['global_orient'] # data['trans'] [N, 3]

    global_orient = torch.zeros(frame_num,3)
    # if 'trans' not in data.keys():
    #     global_orient = torch.zeros(frame_num,3)
    # else:
    #     global_orient = torch.Tensor(data['trans'])
        
    if 'poses' in data.keys(): # 载入动画序列的某一帧
        body_pose = torch.Tensor(data['poses'][:,3:66])
    else:
        body_pose = torch.Tensor(data['body_pose'][3:66]).unsqueeze(0)
    # most AMASS meshes don't have hand poses, so we don't use them here.
    #left_hand_pose = torch.Tensor(data['body_pose'][75:81]).unsqueeze(0)
    #right_hand_pose = torch.Tensor(data['body_pose'][81:]).unsqueeze(0)

    params = dict(
        betas = betas.to(device),
        global_orient = global_orient[0].to(device),
        body_pose = body_pose[0].to(device),
        #left_hand_pose = left_hand_pose,
        #right_hand_pose = right_hand_pose
    )
    
    total_body_mesh = [copy.deepcopy(body) for _ in range(frame_start, frame_end+1)]
    
    # reset params for each frame
    for idx, cur_frame in enumerate(range(frame_start, frame_end+1)):
        if assign_frame_idx >= 0: # 仅添加特定帧
            params['body_pose'] = body_pose[assign_frame_idx].to(device) # params_dict['poses']为list, 原55*3旋转，仅提取其中3:63，可以包含手指吗？
        else:
            params['body_pose'] = body_pose[cur_frame].to(device) # smplx的21个身体关节
        total_body_mesh[idx].reset_params(**params) # 设置对应帧的姿势参数， 暂未考虑全局位置
    
    params['body_pose'] = params['body_pose'].unsqueeze(0)
    params['global_orient'] = params['global_orient'].unsqueeze(0)
    
    return total_body_mesh, params

def main(cfg):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: ', device)
    
    # process arguments
    OUTPUT_DIR = cfg.output_folder
    HCP_PATH = osp.join(cfg.essentials_folder, 'hand_on_body_prior/smplx/smplx_handonbody_prior.pkl')
    SEGMENTATION_PATH = osp.join(cfg.essentials_folder, 'models_utils/smplx/smplx_segmentation_id.npy')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    npz_file = osp.join(cfg.essentials_folder, f'example_poses/pose1.npz') \
        if cfg.input_file == '' else cfg.input_file
    file_name = os.path.basename(npz_file)
    file_base_name, _ = os.path.splitext(file_name)
    print('Processing: ', npz_file)

    # models = get_bodymodels(
    #     model_path=cfg.model_folder, 
    #     model_type=cfg.body_model.model_type, 
    #     device=device,
    #     batch_size=cfg.batch_size, 
    #     num_pca_comps=cfg.body_model.num_pca_comps
    # )
        
    sc_module = SelfContact( 
        essentials_folder=cfg.essentials_folder,
        geothres=cfg.contact.geodesic_thres, 
        euclthres=cfg.contact.euclidean_thres, 
        model_type=cfg.body_model.model_type,
        test_segments=cfg.contact.test_segments,
        compute_hd=False,
        buffer_geodists=True,
    ).to(device) # 479 -> 1664->(no buffer_geodists) 1195

    if cfg.LowerBody:
        extremities_body_segment = LEFT_RIGHT_LEG_INDEX_55
        extremities_index_21 = LEFT_RIGHT_LEG_INDEX_21
    elif cfg.FullBody:
        extremities_body_segment = smplx_extremities_nums
        extremities_index_21 = range(21)
    else:
        extremities_body_segment = LEFT_RIGHT_ARM_INDEX_55
        extremities_index_21 = LEFT_RIGHT_ARM_INDEX_21
    extremities = get_extremities_assign_bones(
        SEGMENTATION_PATH, 
        extremities_body_segment,
        cfg.contact.test_segments
    )
    
    criterion = SelfContactOptiAnimAssignLoss( 
        contact_module=sc_module,
        inside_loss_weight=cfg.loss.inside_weight,
        outside_loss_weight=cfg.loss.outside_weight,
        contact_loss_weight=cfg.loss.contact_weight,
        hand_contact_prior_weight=cfg.loss.hand_contact_prior_weight,
        pose_prior_weight=cfg.loss.pose_prior_weight,
        hand_pose_prior_weight=cfg.loss.hand_pose_prior_weight,
        angle_weight=cfg.loss.angle_weight,
        smooth_angle_weight=cfg.loss.smooth_angle_weight,
        smooth_verts_weight=cfg.loss.smooth_verts_weight,
        smooth_joints_weight=cfg.loss.smooth_joints_weight,
        hand_contact_prior_path=HCP_PATH,
        downsample=extremities,
        use_hd=False,
        test_segments=cfg.contact.test_segments,
        device=device,
        fps=cfg.fps,
        frame_start=cfg.frame_start,
        frame_end=cfg.frame_end,
        frame_num=cfg.frame_end - cfg.frame_start + 1,
        cfg=cfg,
        file_base_name=file_base_name,
        extremities_index_21=extremities_index_21,
    )

    anim_opti = SelfContactAnimOpti(
        loss=criterion,
        optimizer_name=cfg.optimizer.name,
        optimizer_lr_body=cfg.optimizer.learning_rate_body,
        optimizer_lr_hands=cfg.optimizer.learning_rate_hands,
        max_iters=cfg.optimizer.max_iters,
        output_folder=OUTPUT_DIR,
        npz_file=npz_file,
        file_base_name=file_base_name,
        save_step=cfg.save_step,
        cfg=cfg,
    )

    # model = models[gender]
    total_body_model, params = load_model_param(npz_file,cfg,device) # 1642 -> 2536
    
    start_optim = time.time()
    total_output_body_mesh = anim_opti.run(total_body_model, params,cfg.assign_frame_idx)
    print('Total Optimization: {:5f}'.format(time.time() - start_optim))
    
    save_npz_file(OUTPUT_DIR, npz_file, total_output_body_mesh, file_base_name, cfg)
    # body = total_output_body_mesh[0] # 调试查看结果
    # mesh = trimesh.Trimesh(body.vertices[0].detach().cpu().numpy(), total_body_model[0].faces)
    # mesh.export(osp.join(OUTPUT_DIR, file_base_name+f'.obj'))

    # print('obj written to: '+osp.join(OUTPUT_DIR, file_base_name+f'.obj'))
    # out_dict = {}
    # for key, val in body.items():
    #     if val is not None:
    #         out_dict[key] = val.detach().cpu().numpy()
    # with open(osp.join(OUTPUT_DIR, file_base_name+f'.pkl'), 'wb') as f:
    #     pickle.dump(out_dict, f)
    # print('pkl written to: '+osp.join(OUTPUT_DIR, file_base_name+f'.pkl'))
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', 
        default='selfcontact/tutorial/configs/selfcontact_optimization_config_anim.yaml')
    parser.add_argument('--essentials_folder', required=True, 
        help='folder with essential data. Check Readme for download dir.')
    parser.add_argument('--output_folder', required=True, 
        help='folder where the example obj files are written to')
    parser.add_argument('--model_folder', required=True, 
        help='folder where the body models are saved.')
    parser.add_argument('--input_file', default='', 
        help='Input filename to be processed. Expects an npz file with the model parameters.')
    parser.add_argument('--assign_frame_idx', default=-1, type=int,
                        help='assign_frame_idx for test, -1 for all frames.')
    parser.add_argument('--frame_start', default=0, type=int,
                        help='frame start index.')
    parser.add_argument('--frame_end', default=10, type=int,
                        help='frame end index.')
    parser.add_argument('--LowerBody', action='store_true',default=False,
                        help='LowerBody or not.')
    parser.add_argument('--FullBody', action='store_true',default=False,
                        help='FullBody or not.')
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        cfg = DotConfig(
            input_dict=yaml.safe_load(stream)
        )
    
    cfg.update(vars(args))

    main(cfg)