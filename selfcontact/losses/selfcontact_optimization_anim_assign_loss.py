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

import pickle
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from selfcontact import SelfContact
import numpy as np
from ..utils.prior import L2Prior
from ..utils.mesh import compute_vertex_normals
from ..utils.visualization import save_smplx_mesh_bool,save_smplx_mesh_with_obb,save_smplx_mesh_bone
from ..utils.extremities import get_vertices_assign_bone,get_connected_faces
from ..utils.obb import build_obb
from ..utils.SMPLXINFO import *
import time

class SelfContactOptiAnimAssignLoss(nn.Module):
    def __init__(self,
        contact_module,
        inside_loss_weight,
        outside_loss_weight,
        contact_loss_weight,
        hand_contact_prior_weight,
        pose_prior_weight,
        hand_pose_prior_weight,
        angle_weight,
        smooth_angle_weight,
        smooth_joints_weight,
        smooth_verts_weight,
        hand_contact_prior_path,
        downsample,
        test_segments=False,
        use_hd=False,
        alpha1=0.04,
        alpha2=0.04,
        beta1=0.07,
        beta2=0.06,
        gamma1=0.01,
        gamma2=0.01,
        delta1=0.023,
        delta2=0.02,
        device='cuda',
        fps=30,
        frame_num=10,
        frame_start=0,
        frame_end=10,
        cfg = None,
        file_base_name = 'result',
        extremities_index_21 = list(range(21)),
    ):
        super().__init__()

        self.ds = downsample
        self.test_segments = test_segments 
        self.use_hd = use_hd

        # weights
        self.inside_w = inside_loss_weight
        self.contact_w = contact_loss_weight
        self.outside_w = outside_loss_weight
        self.hand_contact_prior_weight = hand_contact_prior_weight
        self.pose_prior_weight = pose_prior_weight
        self.hand_pose_prior_weight = hand_pose_prior_weight
        self.angle_weight = angle_weight 
        self.smooth_angle_weight = smooth_angle_weight 
        self.smooth_verts_weight = smooth_verts_weight
        self.smooth_joints_weight = smooth_joints_weight
        
        # hyper params
        self.a1 = alpha1
        self.a2 = alpha2
        self.b1 = beta1
        self.b2 = beta2
        self.c1 = gamma1
        self.c2 = gamma2
        self.d1 = delta1
        self.d2 = delta2

        # load hand-on-body prior
        with open(hand_contact_prior_path, 'rb') as f:
                    hand_contact_prior = pickle.load(f)
        lefthandids = torch.tensor(hand_contact_prior['left_hand_verts_ids']) # [778]
        righthandids = torch.tensor(hand_contact_prior['right_hand_verts_ids']) # [778]
        weights = torch.tensor(hand_contact_prior['mean_dist_hands_scaled']) # [778]
        self.hand_contact_prior = torch.cat((lefthandids,righthandids)).to(device) # [1556]
        self.hand_contact_prior_weights = torch.cat((weights, weights)).to(device) # [1556]

        # create hand pose prior
        self.hand_pose_prior = L2Prior().to(device)

        # self contact module
        self.cm = contact_module
        self.geodist = self.cm.geodesicdists
        
        self.fps = fps
        self.frame_num = frame_num
        self.device = device
        self.frame_start = frame_start
        self.frame_end = frame_end
        
        self.cfg = cfg
        self.file_base_name = file_base_name
        
        self.mse_loss = nn.MSELoss(reduction='sum').to(device=device)
        self.extremities_index_21 = extremities_index_21

    def configure(self, total_body_model, total_output_body_mesh):
        if self.contact_w <= 0 and self.outside_w <= 0: # 减少占用内存
            return
        init_poses = []
        init_verts = []
        init_verts_in_contact_idx = []
        self.init_verts_in_contact = []

        # 为每个 body_model 计算数据并存储
        for i, body_model in enumerate(total_body_model):
            # print(f"registering buffers for body_model {i}")
            # vertices = body_model(**params).vertices 
            vertices = total_output_body_mesh[i].vertices # 不用蒙皮，已经计算
            init_poses.append(body_model.body_pose.clone().detach())# .to(torch.float16)
            init_verts.append(vertices.clone().detach()) # .to(torch.float16)

            # print(f"init_verts_in_contact_idx for body_model {i}")
            with torch.no_grad():
                init_verts_in_contact_idx.append(self.cm.segment_vertices(init_verts[i], test_segments=False)[0][1][0]) # [10475] Mask 接触状态为True，此处保存初始的接触状态
                self.init_verts_in_contact.append(torch.where(init_verts_in_contact_idx[i])[0].cpu().numpy()) # [824] 接触状态的顶点索引
                # init_verts_in_contact.append(torch.where(init_verts_in_contact_idx[i])[0].clone().detach()) # [824] 接触状态的顶点索引
                # torch.cuda.empty_cache()

        # 将数据注册为二维数组
        self.register_buffer('init_poses', torch.stack(init_poses))
        self.register_buffer('init_verts', torch.stack(init_verts))
        # self.register_buffer('init_verts_in_contact_idx', torch.stack(init_verts_in_contact_idx))
        # self.register_buffer('init_verts_in_contact', np.stack(init_verts_in_contact))
    
    def single_forward(self, idx, body, model = None):
        """
            compute loss based on status of vertex
        """
        
        # initialize loss tensors
        device = body.vertices.device
        vertices = body.vertices
        vertices = vertices.to(torch.float16)
        _, nv, _ = vertices.shape

        loss = torch.tensor(0.0, device=device, requires_grad=True) # torch.float16
        insideloss = torch.tensor(0.0, device=device, requires_grad=True)
        contactloss = torch.tensor(0.0, device=device, requires_grad=True)
        outsideloss = torch.tensor(0.0, device=device, requires_grad=True)
        hand_contact_loss_inside = torch.tensor(0.0, device=device, requires_grad=True)
        hand_contact_loss_outside = torch.tensor(0.0, device=device, requires_grad=True)
        hand_contact_loss = torch.tensor(0.0, device=device, requires_grad=True)
        angle_loss = torch.tensor(0.0, device=device, requires_grad=True)
        left_hand_contact_loss_inside = torch.tensor(0.0, device=device, requires_grad=True)
        right_hand_contact_loss_inside = torch.tensor(0.0, device=device, requires_grad=True)
        left_hand_contact_loss_outside = torch.tensor(0.0, device=device, requires_grad=True)
        right_hand_contact_loss_outside = torch.tensor(0.0, device=device, requires_grad=True)
        pose_prior_loss = torch.tensor(0.0, device=device, requires_grad=True)
        hand_pose_prior_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # torch.cuda.empty_cache()
        v2v_ds_min = None
        v2v_ds_min_idx = None
        v2v_hand_min = None # 手部顶点最短距离
        
        v2v_hand_min = self.cm.get_hand_vertices_min(
            hand_idx = self.hand_contact_prior, # 左右手顶点索引
            hand_vertices=vertices[:, self.hand_contact_prior, :], # 左右手顶点
            vertices=vertices, # 所有顶点
        )
        
        if self.test_segments:
            v2v_min, v2v_min_idx, exterior \
                = self.cm.segment_vertices_scopti( # test_segments是只取指定segments的顶点？
                vertices=vertices,
                test_segments=self.test_segments,
            )
            # only select vertices on extremities
            exterior = exterior[:,self.ds]
        else:
            # v2v_min, v2v_min_idx, exterior = self.cm.segment_points_scopti( # 834MB  4794 -> 6047   2521 -> 6015 / 4647(CHATGPT) / 4120(KIMI)
            #     points=vertices[:, self.ds, :], # ds是选择的特定部位的顶点
            #     vertices=vertices,
            # ) # 3141-1262
            v2v_ds_min, v2v_ds_min_idx, exterior = self.cm.segment_points_scopti_ds( # 834MB  4794 -> 6047   2521 -> 6015 / 4647(CHATGPT) / 4120(KIMI)
                ds = self.ds, # 所有ds顶点
                ds_vertices=vertices[:, self.ds, :], # ds是选择的特定部位的顶点
                vertices=vertices,
            ) 
        # v2v_min = v2v_min.squeeze() # [10475]
        
        torch.cuda.empty_cache() # 清理缓存

        # 获取GPU显存的总量和已使用量
        # used_memory = torch.cuda.memory_allocated(torch.cuda.current_device()) / (1024 ** 3)  # 已使用显存(GB)
        # print(f"已使用的GPU显存：{used_memory:.2f} GB")

        # only extremities intersect
        inside = torch.zeros(nv).to(device).to(torch.bool)
        # rewritten, because of pytorch bug when torch.use_deterministic_algorithms(True)
        true_tensor = torch.ones((~exterior[0]).sum().item(), device=device, dtype=torch.bool)
        inside[self.ds[~exterior[0]]] = true_tensor

        if model is not None:
            pass
            # 测试: 生成OBB盒
            SEGMENTATION_PATH = '/usr/pydata/t2m/selfcontact/selfcontact-essentials/models_utils/smplx/smplx_segmentation_id.npy'
            for bone_idx in range(55):
                vert_idx = get_vertices_assign_bone(SEGMENTATION_PATH, bone_idx)
                box_verts, box_face = build_obb(vertices[:, vert_idx, :].squeeze(0))
                # target_faces = []
                # for face in self.cm.faces:
                #     if(face[0].item() in vert_idx and face[1].item() in vert_idx and face[2].item() in vert_idx):
                #         target_faces.append(np.array(face.cpu()))
                # target_faces = np.array(target_faces)
                
                bound_faces = []
                inner_faces = []
                for face in self.cm.faces:
                    count = sum(1 for vertex in face if vertex.item() in vert_idx) # 两个顶点在该部位，认为是边界三角形
                    if count == 2:
                        bound_faces.append(np.array(face.cpu()))
                    if count == 3:
                        inner_faces.append(np.array(face.cpu()))
                bound_faces = np.array(bound_faces)
                inner_faces = np.array(inner_faces)
                
                verts_with_bone_close, face_with_bone_close = get_connected_faces(vertices, bound_faces, self.cm.faces) # list
                # bound_faces = self.cm.get_connected_faces() # list
                # inner_and_bound_faces = np.concatenate((bound_faces, face_with_bone_close), axis=0)
                if len(face_with_bone_close) > 0:
                    inner_and_bound_faces = np.concatenate((inner_faces, face_with_bone_close), axis=0)
                save_smplx_mesh_bone(verts_with_bone_close, inner_and_bound_faces, file_name = self.file_base_name, cfg = self.cfg)
            # save_smplx_mesh_with_obb(model, body, box_verts, box_face, file_name = self.file_base_name, cfg = self.cfg)
            # save_smplx_mesh_bool(model, body,~inside, file_name = self.file_base_name, cfg = self.cfg)
            
        # ==== contact loss ==== 内部or外部接触损失
        # pull outside together
        if self.contact_w > 0 and (~inside).sum() > 0:
            # if len(self.init_verts_in_contact) > 0:
            #     gdi = self.geodist[:, self.init_verts_in_contact].min(1)[0]
            #     weights_outside = 1 / (5 * gdi + 1)
            if len(self.init_verts_in_contact[idx]) > 0:
                gdi = self.geodist[:, self.init_verts_in_contact[idx]].min(1)[0]
                weights_outside = 1 / (5 * gdi + 1)       
            else:
                weights_outside = torch.ones_like(self.geodist)[:,0].to(device)
            attaction_weights = weights_outside[self.ds][~inside[self.ds]]
            if v2v_ds_min is not None:
                v2voutside = v2v_ds_min[~inside[self.ds]] # 外部距离
            else:
                v2voutside = v2v_min[self.ds][~inside[self.ds]] # 指定部位中外部的顶点
            v2voutside = self.a1 * attaction_weights.to(self.device)  * torch.tanh(v2voutside/self.a2)
            contactloss = self.contact_w * v2voutside.mean() # 0.6687

        # push inside to surface
        if self.inside_w > 0 and inside.sum() > 0:
            if v2v_ds_min is not None:
                v2vinside = v2v_ds_min[inside[self.ds]]
            else:
                v2vinside = v2v_min[inside] # [176]
            v2vinside = self.b1 * torch.tanh(v2vinside / self.b2)
            insideloss = self.inside_w * v2vinside.mean()

        # ==== hand-on-body prior loss ==== hand-on-body先验损失
        if self.hand_contact_prior_weight > 0:
            ha = int(self.hand_contact_prior.shape[0] / 2) # 778 单只手部顶点数
            hand_verts_inside = inside[self.hand_contact_prior] # 1556 手部顶点在内部为True

            if (~hand_verts_inside).sum() > 0:
                if v2v_hand_min is not None:
                    left_hand_outside = v2v_hand_min[:ha][(~hand_verts_inside)[:ha]] # 取左手&在外部的最小距离 487
                    right_hand_outside = v2v_hand_min[ha:][(~hand_verts_inside)[ha:]] # 取右手&在外部的最小距离 487
                else:
                    left_hand_outside = v2v_min[self.hand_contact_prior[:ha]][(~hand_verts_inside)[:ha]] # 取左手&在外部的最小距离 487
                    right_hand_outside = v2v_min[self.hand_contact_prior[ha:]][(~hand_verts_inside)[ha:]] # 取右手&在外部的最小距离 487
                    
                # weights based on hand contact prior
                left_hand_weights = -0.1 * self.hand_contact_prior_weights[:ha].view(-1,1)[(~hand_verts_inside)[:ha]].view(-1,1) + 1.0  # 左手外部权重 [487, 1]
                right_hand_weights = -0.1 * self.hand_contact_prior_weights[ha:].view(-1,1)[(~hand_verts_inside)[ha:]].view(-1,1) + 1.0 # 右手外部权重 [487, 1]    
                if left_hand_outside.sum() > 0:
                    left_hand_contact_loss_outside = self.c1 * torch.tanh(left_hand_outside/self.c2)
                if right_hand_outside.sum() > 0:
                    right_hand_contact_loss_outside = self.c1 * torch.tanh(right_hand_outside/self.c2)
                hand_contact_loss_outside = (left_hand_weights * left_hand_contact_loss_outside.view(-1,1)).mean() + \
                                            (right_hand_weights * right_hand_contact_loss_outside.view(-1,1)).mean()

            if hand_verts_inside.sum() > 0:
                if v2v_hand_min is not None:
                    left_hand_inside = v2v_hand_min[:ha][(hand_verts_inside)[:ha]]
                    right_hand_inside = v2v_hand_min[ha:][(hand_verts_inside)[ha:]]
                else:
                    left_hand_inside = v2v_min[self.hand_contact_prior[:ha]][hand_verts_inside[:ha]]
                    right_hand_inside = v2v_min[self.hand_contact_prior[ha:]][hand_verts_inside[ha:]]
                if left_hand_inside.sum() > 0:
                    left_hand_contact_loss_inside = self.d1 * torch.tanh(left_hand_inside/self.d2)
                if right_hand_inside.sum() > 0:
                    right_hand_contact_loss_inside = self.d1 * torch.tanh(right_hand_inside/self.d2)
                hand_contact_loss_inside = left_hand_contact_loss_inside.mean() + right_hand_contact_loss_inside.mean()

            hand_contact_loss = self.hand_contact_prior_weight * (hand_contact_loss_inside + hand_contact_loss_outside)

        # ==== align normals of verts in contact ==== 接触顶点的法线对齐
        if self.angle_weight > 0:
            if v2v_ds_min_idx is not None: # 仅对ds中距离很近的顶点进行法线对齐
                verts_close = torch.where(v2v_ds_min < 0.01)[0] # v2v小于0.01的顶点索引
                if len(verts_close) > 0:
                    vertex_normals = compute_vertex_normals(body.vertices, self.cm.faces) # 1,10475,3 每个顶点的法线
                    dotprod_normals = torch.matmul(vertex_normals, torch.transpose(vertex_normals,1,2))[0] # 10475,10475 每个顶点的法线点乘
                    #normalsgather = dotprod_normals.gather(1, v2v_min_idx.view(-1,1))
                    normalsgather = dotprod_normals[np.arange(nv), v2v_ds_min_idx.view(-1,1)]
                    angle_loss = 1 + normalsgather[verts_close,:]
                    angle_loss = self.angle_weight * angle_loss.mean()
            else:
                verts_close = torch.where(v2v_min < 0.01)[0] # v2v小于0.01的顶点索引
                if len(verts_close) > 0:
                    vertex_normals = compute_vertex_normals(body.vertices, self.cm.faces) # 1,10475,3 每个顶点的法线
                    dotprod_normals = torch.matmul(vertex_normals, torch.transpose(vertex_normals,1,2))[0] # 10475,10475 每个顶点的法线点乘
                    #normalsgather = dotprod_normals.gather(1, v2v_min_idx.view(-1,1))
                    normalsgather = dotprod_normals[np.arange(nv), v2v_min_idx.view(-1,1)]
                    angle_loss = 1 + normalsgather[verts_close,:]# 这里是verts_close与所有其他顶点的法线对齐？
                    angle_loss = self.angle_weight * angle_loss.mean()

        # ==== penalize deviation from initial pose params ==== 惩罚偏离初始姿态参数
        # pose_prior_loss = self.pose_prior_weight * F.mse_loss(body.body_pose, self.init_pose, reduction='sum')
        if self.pose_prior_weight > 0:
            pose_prior_loss = self.pose_prior_weight * F.mse_loss(body.body_pose, self.init_poses[idx], reduction='sum') # .to(torch.float16)

        # ==== penalize deviation from mean hand pose ==== 惩罚偏离平均手姿势
        if self.hand_pose_prior_weight > 0:
            hand_pose_prior_loss = self.hand_pose_prior_weight * \
                (self.hand_pose_prior(body.left_hand_pose) + self.hand_pose_prior(body.right_hand_pose))

        # ==== pose regression loss / outside loss ==== 回归Loss/外部Loss
        if self.outside_w > 0:
            # outsidelossv2v = torch.norm(self.init_verts-body.vertices, dim=2)
            outsidelossv2v = torch.norm(self.init_verts[idx]-body.vertices, dim=2)
            # if self.init_verts_in_contact.sum() > 0:
            #     gd = self.geodist[:, self.init_verts_in_contact].min(1)[0]
            #     outsidelossv2vweights = (2 * gd.view(body.vertices.shape[0], -1))**2
            if self.init_verts_in_contact[idx].sum() > 0:
                gd = self.geodist[:, self.init_verts_in_contact[idx]].min(1)[0]
                outsidelossv2vweights = (2 * gd.view(body.vertices.shape[0], -1))**2
            else:
                outsidelossv2vweights = torch.ones_like(outsidelossv2v).to(device)
            outsidelossv2v = (outsidelossv2v * outsidelossv2vweights).sum()
            outsideloss = self.outside_w * outsidelossv2v

        # ==== Total loss ====
        loss = insideloss + contactloss + outsideloss + pose_prior_loss + hand_pose_prior_loss + angle_loss + hand_contact_loss
        loss_dict = {
            'Total': loss.item(),
            'Contact': contactloss.item(),
            'Inside': insideloss.item(),
            'Outside': outsideloss.item(),
            'Angles': angle_loss.item(),
            'HandContact': hand_contact_loss.item(),
            'HandPosePrior':hand_pose_prior_loss.item(),
            'BodyPosePrior': pose_prior_loss.item(),
        }
        # torch.cuda.empty_cache()
        return loss, loss_dict
    
    def forward(self, total_body, mesh = None): # 4790

        # initialize loss tensors
        device = total_body[0].vertices.device
        # vertices = total_body[0].vertices
        # _, nv, _ = vertices.shape
        
        loss = torch.tensor(0.0, device=device)
        selfcontact_loss = torch.tensor(0.0, device=device)
        smooth_angle_loss = torch.tensor(0.0, device=device)
        smooth_verts_loss = torch.tensor(0.0, device=device)
        smooth_joints_loss = torch.tensor(0.0, device=device)
        
        loss_dict_total = {
            'Total': 0,
            'Contact': 0,
            'Inside': 0,
            'Outside': 0,
            'Angles': 0,
            'HandContact': 0,
            'HandPosePrior':0,
            'BodyPosePrior': 0,
            'SmoothAngle': 0,
            'SmoothVerts': 0,
            'SmoothJoints': 0,
        }
        
        # 1.单帧优化       
        start_time = time.time()
        for idx, body in enumerate(total_body):
            sc_loss, loss_dict = self.single_forward(idx, body, mesh)
            selfcontact_loss += sc_loss
            for key in loss_dict:
                loss_dict_total[key] += loss_dict[key]
        selfcontact_loss = selfcontact_loss / len(total_body)
        for key in loss_dict:
            loss_dict_total[key] /= len(total_body)
        print('single_forward: {:5f}'.format(time.time() - start_time))
        
        
        # 2.平滑优化
        if self.smooth_angle_weight > 0 and len(total_body) > 1:
            body_poses_total = [cur_body.body_pose for cur_body in total_body] # 获取每帧的pose
            smooth_angle_loss = self.smooth_angle_weight * self.calculate_angle_acc_smooth_loss_total(body_poses_total) # 计算平滑loss
        
        if self.smooth_verts_weight > 0 and len(total_body) > 1:
            total_verts = [cur_body.vertices for cur_body in total_body]
            smooth_verts_loss = self.smooth_verts_weight * self.calculate_verts_vel_smooth_loss_total(total_verts) # 相邻帧顶点距离
            
        if self.smooth_joints_weight > 0 and len(total_body) > 1:
            total_joints_position = [cur_body.joints[:,:55,:] for cur_body in total_body] # 每帧的关节点位置 1, 55, 3 -> 1, 76, 3 -> 1, 127, 3 y轴向上
            smooth_joints_loss = self.smooth_joints_weight * self.calculate_joints_smooth_loss_total(total_joints_position) # 仅考虑相邻帧顶点距离
            
        loss_dict_total['SmoothAngle'] = smooth_angle_loss.item()
        loss_dict_total['SmoothVerts'] = smooth_verts_loss.item()
        loss_dict_total['SmoothJoints'] = smooth_joints_loss.item()
        
        loss = selfcontact_loss + smooth_angle_loss + smooth_verts_loss + smooth_joints_loss
        
        return loss, loss_dict_total
      
    def get_quat_from_rodrigues_tensor(self, rodrigues):
         # 计算罗德里格斯向量的长度，即旋转角度的弧度值
         angle_rad = torch.norm(rodrigues, dim=0)
         if angle_rad == 0:
             return torch.Tensor([1, 0, 0, 0])
         # 归一化罗德里格斯向量，得到旋转轴
         axis = rodrigues / angle_rad

         # 根据罗德里格斯公式计算四元数
         # 这里我们使用公式: q = cos(θ/2) + sin(θ/2) * (x * i + y * j + z * k)
         half_angle = angle_rad / 2
         cos_half_angle = torch.cos(half_angle)
         sin_half_angle = torch.sin(half_angle)
         quat = torch.stack([
             cos_half_angle,
             sin_half_angle * axis[0],
             sin_half_angle * axis[1],
             sin_half_angle * axis[2]
         ], dim=0)

         return quat
    
    def quaternion_multiply(self, quaternion1, quaternion0):
        w0, x0, y0, z0 = quaternion0.unbind(-1)
        w1, x1, y1, z1 = quaternion1.unbind(-1)
        result = torch.stack([
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0
        ], dim=-1)
        return result
    
    def quaternion_inverse(self, q):
        w, x, y, z = q.unbind(-1)
        conjugate = torch.stack([w, -x, -y, -z], dim=-1)
        norm = torch.norm(q, dim=-1, keepdim=True)
        # if torch.allclose(norm, torch.Tensor([0],device=device)):
        #     return torch.Tensor([1, 0, 0, 0])
        return conjugate / (norm ** 2)
    
    # 计算一个骨骼的角加速度
    def calculate_angular_acceleration(self, orientation):
        data_count = len(orientation)
        # time_delta = self.time[1:] - self.time[:-1] # time delta
        time_delta = 1 / self.fps
        ## Divided difference for dq_dt
        dq_dt = torch.zeros((data_count,4),device=self.device)
        # dq_dt[0][0] = 1 # 第0帧角速度设置为(1,0,0,0) 否则第一帧梯度为计算为nan
        # dq_dt[0][0] = 1 # 第0帧角速度设置为(1,0,0,0) 否则第一帧梯度为计算为nan
        dq_dt[1:] = (orientation[1:] - orientation[:-1]) / time_delta # 第1帧到最后一帧计算速度，1阶导
        dq_dt[0] = dq_dt[1] # 第0帧角速度设置为与第1帧相同，使得第1帧角加速度为0，避免优化
        ## Divided difference for d2q_dt2
        d2q_dt2 = torch.zeros((data_count,4),device=self.device)
        
        d2q_dt2[2:] = (dq_dt[2:] - dq_dt[:-2]) / time_delta # 速度的第2帧到最后2帧计算加速度，2阶导
        # d2q_dt2[0][0] = 1 
        # d2q_dt2[1][0] = 1 # 前两帧角加速度设置为 1,0,0,0
        ## Calculate velocity
        velocity = 2 * self.quaternion_multiply(dq_dt, self.quaternion_inverse(orientation))
        ## Calculate acceleration
        temp = self.quaternion_multiply(dq_dt, self.quaternion_inverse(orientation))
        acceleration = 2 * (self.quaternion_multiply(d2q_dt2, self.quaternion_inverse(orientation)) - self.quaternion_multiply(temp,temp))
        return acceleration, velocity
    
    # vert最小
    def calculate_verts_smooth_loss_total(self, total_verts):
        smooth_loss = torch.tensor(0.0, device=self.device)
        for i in range(1, len(total_verts)):
            # smooth_loss += F.mse_loss(total_verts[i], total_verts[i-1],reduce='sum')
            smooth_loss += self.mse_loss(total_verts[i], total_verts[i-1])
        smooth_loss = smooth_loss / len(total_verts)
        return smooth_loss
    
    # vert速度最小
    def calculate_verts_vel_smooth_loss_total(self, total_verts):
        smooth_loss = torch.tensor(0.0, device=self.device)
        for i in range(1, len(total_verts)):
            velocity = (total_verts[i] - total_verts[i - 1]) * self.fps
            smooth_loss += torch.mean(torch.norm(velocity, dim=-1))
        smooth_loss = smooth_loss / len(total_verts)
        return smooth_loss
    
    def calculate_joints_smooth_loss_total(self, body_joints_position_total):
        smooth_loss = torch.tensor(0.0, device=self.device)
        for i in range(1, len(body_joints_position_total)):
            # smooth_loss += F.mse_loss(total_verts[i], total_verts[i-1],reduce='sum')
            smooth_loss += self.mse_loss(body_joints_position_total[i], body_joints_position_total[i-1])
        smooth_loss = smooth_loss / len(body_joints_position_total)
        return smooth_loss
    
    def calculate_angle_acc_smooth_loss_total(self, body_poses_total):
        smooth_loss = torch.tensor(0.0, device=self.device)
        # for bone_idx in range(NUM_SMPLX_BODYJOINTS):
        target_bone_idx = self.extremities_index_21 # 21
        for bone_idx in target_bone_idx: # 右手、手臂、肩膀
            orientation = torch.zeros((len(body_poses_total), 4),device=self.device) # 每个骨骼全部帧的四元数
            for frame_index, frame_pose in enumerate(body_poses_total):
                current_pose = frame_pose.reshape(-1, 3)
                bone_quaternion = self.get_quat_from_rodrigues_tensor(current_pose[bone_idx])
                orientation[frame_index] = bone_quaternion
            acc, vel = self.calculate_angular_acceleration(orientation)
            # 1. 直接求全部的角加速度之和
            # bone_loss = torch.sum(acc.pow(2), dim=1).sqrt().sum()
            
            # 2.求大于阈值的元素之和
            # 找出平方和大于阈值的元素的索引
            # indices = squared_sums > acc_thres
            # 计算选定行的平方和
            # bone_loss = torch.sum(squared_sums)
            # 计算acc张量中每个元素的平方
            squared_sums =  torch.norm(acc, dim=1)  # 角加速度最小化 torch.sum(acc.pow(2), dim=1).sqrt()
            # squared_sums = torch.sum(vel.pow(2), dim=1).sqrt() # 角速度最小化，旋转趋于静止
            # 除了该骨骼，除了该帧，其他是否有变化，其他帧有变化，其他骨骼没影响
            # 3.求前k个最大值的和
            # k = 5
            # values, indices = torch.topk(squared_sums, k=k, largest=True, sorted=True)
            indices = range(0,self.frame_num) # 仅优化特定的帧
            # 计算前5个最大值的和
            sum_of_top_values = torch.sum(squared_sums[indices]) # 仅优化碰撞帧，但是感觉不如一个区间优化平滑
            sum_of_top_values /= len(indices)
            # print(indices)
            smooth_loss = smooth_loss + sum_of_top_values
        smooth_loss  = smooth_loss / len(target_bone_idx)
        return smooth_loss
