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
import torch.nn as nn
import torch.nn.functional as F
from selfcontact import SelfContact
import numpy as np
from ..utils.prior import L2Prior
from ..utils.mesh import compute_vertex_normals
from ..utils.visualization import save_smplx_mesh_bool
import time

class SelfContactOptiAnimLoss(nn.Module):
    def __init__(self,
        contact_module,
        inside_loss_weight,
        outside_loss_weight,
        contact_loss_weight,
        hand_contact_prior_weight,
        pose_prior_weight,
        hand_pose_prior_weight,
        angle_weight,
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
        frame_num=10
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
        lefthandids = torch.tensor(hand_contact_prior['left_hand_verts_ids'])
        righthandids = torch.tensor(hand_contact_prior['right_hand_verts_ids'])
        weights = torch.tensor(hand_contact_prior['mean_dist_hands_scaled'])
        self.hand_contact_prior = torch.cat((lefthandids,righthandids)).to(device)
        self.hand_contact_prior_weights = torch.cat((weights, weights)).to(device)

        # create hand pose prior
        self.hand_pose_prior = L2Prior().to(device)

        # self contact module
        self.cm = contact_module
        self.geodist = self.cm.geodesicdists
        
        self.fps = fps
        self.frame_num = frame_num
        self.device = device

    # def configure(self, body_model, params): # 耗时有时会崩

    #     # initial mesh and pose 
    #     print("registering buffers")
    #     vertices = body_model(**params).vertices
    #     self.register_buffer('init_pose', body_model[0].body_pose.clone().detach())
    #     self.register_buffer('init_verts', vertices.clone().detach())

    #     # get verts in contact in initial mesh
    #     print("init_verts_in_contact_idx")
    #     with torch.no_grad():
    #         self.init_verts_in_contact_idx = \
    #             self.cm.segment_vertices(self.init_verts, test_segments=False)[0][1][0]  # [10475] Mask 接触状态为True，此处保存初始的接触状态
    #         self.init_verts_in_contact = torch.where(self.init_verts_in_contact_idx)[0].cpu().numpy() # [824] 接触状态的顶点索引
    
    def configure(self, total_body_model, params):     
        self.init_poses = []
        self.init_verts = []
        init_verts_in_contact_idx = []
        self.init_verts_in_contact = []

        # 为每个 body_model 计算数据并存储
        for i, body_model in enumerate(total_body_model):
            # print(f"registering buffers for body_model {i}")
            vertices = body_model(**params).vertices
            self.init_poses.append(body_model.body_pose.to(torch.float16).clone().detach())
            self.init_verts.append(vertices.to(torch.float16).clone().detach())

            # print(f"init_verts_in_contact_idx for body_model {i}")
            with torch.no_grad():
                init_verts_in_contact_idx.append(self.cm.segment_vertices(self.init_verts[i], test_segments=False)[0][1][0]) # [10475] Mask 接触状态为True，此处保存初始的接触状态
                self.init_verts_in_contact.append(torch.where(init_verts_in_contact_idx[i])[0].cpu().numpy()) # [824] 接触状态的顶点索引
                # init_verts_in_contact.append(torch.where(init_verts_in_contact_idx[i])[0].clone().detach()) # [824] 接触状态的顶点索引

        # 将数据注册为二维数组
        # self.register_buffer('init_poses', torch.stack(init_poses))
        # self.register_buffer('init_verts', torch.stack(init_verts))
        # self.register_buffer('init_verts_in_contact_idx', torch.stack(init_verts_in_contact_idx))
        # self.register_buffer('init_verts_in_contact', np.stack(init_verts_in_contact))
    
    def single_forward(self, idx, body, model = None):
        """
            compute loss based on status of vertex
        """
        
        # initialize loss tensors
        device = body.vertices.device
        vertices = body.vertices
        _, nv, _ = vertices.shape

        loss = torch.tensor(0.0, device=device)
        insideloss = torch.tensor(0.0, device=device)
        contactloss = torch.tensor(0.0, device=device)
        outsideloss = torch.tensor(0.0, device=device)
        hand_contact_loss_inside = torch.tensor(0.0, device=device)
        hand_contact_loss_outside = torch.tensor(0.0, device=device)
        hand_contact_loss = torch.tensor(0.0, device=device)
        angle_loss = torch.tensor(0.0, device=device)
        left_hand_contact_loss_inside = torch.tensor(0.0, device=device)
        right_hand_contact_loss_inside = torch.tensor(0.0, device=device)
        left_hand_contact_loss_outside = torch.tensor(0.0, device=device)
        right_hand_contact_loss_outside = torch.tensor(0.0, device=device)
        # torch.cuda.empty_cache()
        if self.test_segments:
            v2v_min, v2v_min_idx, exterior \
                = self.cm.segment_vertices_scopti( # test_segments是只取指定segments的顶点？
                vertices=vertices,
                test_segments=self.test_segments,
            )
            # only select vertices on extremities
            exterior = exterior[:,self.ds]
        else:
            v2v_min, v2v_min_idx, exterior = self.cm.segment_points_scopti( # 4794 -> 6047
                points=vertices[:, self.ds, :], # ds是选择的特定部位的顶点
                vertices=vertices # 所有顶点
            )
        v2v_min = v2v_min.squeeze() # [10475]
        
        # only extremities intersect
        inside = torch.zeros(nv).to(device).to(torch.bool)
        # rewritten, because of pytorch bug when torch.use_deterministic_algorithms(True)
        true_tensor = torch.ones((~exterior[0]).sum().item(), device=device, dtype=torch.bool)
        inside[self.ds[~exterior[0]]] = true_tensor

        if model is not None:
            save_smplx_mesh_bool(model, body,~inside,"./output/iteration")
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
            v2voutside = v2v_min[self.ds][~inside[self.ds]]
            v2voutside = self.a1 * attaction_weights.to(self.device)  * torch.tanh(v2voutside/self.a2)
            contactloss = self.contact_w * v2voutside.mean()

        # push inside to surface
        if self.inside_w > 0 and inside.sum() > 0:
            v2vinside = v2v_min[inside]
            v2vinside = self.b1 * torch.tanh(v2vinside / self.b2)
            insideloss = self.inside_w * v2vinside.mean()

        # ==== hand-on-body prior loss ==== hand-on-body先验损失
        if self.hand_contact_prior_weight > 0:
            ha = int(self.hand_contact_prior.shape[0] / 2)
            hand_verts_inside = inside[self.hand_contact_prior]

            if (~hand_verts_inside).sum() > 0:
                left_hand_outside = v2v_min[self.hand_contact_prior[:ha]][(~hand_verts_inside)[:ha]]
                right_hand_outside = v2v_min[self.hand_contact_prior[ha:]][(~hand_verts_inside)[ha:]]
                # weights based on hand contact prior
                left_hand_weights = -0.1 * self.hand_contact_prior_weights[:ha].view(-1,1)[(~hand_verts_inside)[:ha]].view(-1,1) + 1.0
                right_hand_weights = -0.1 * self.hand_contact_prior_weights[ha:].view(-1,1)[(~hand_verts_inside)[ha:]].view(-1,1) + 1.0       
                if left_hand_outside.sum() > 0:
                    left_hand_contact_loss_outside = self.c1 * torch.tanh(left_hand_outside/self.c2)
                if right_hand_outside.sum() > 0:
                    right_hand_contact_loss_outside = self.c1 * torch.tanh(right_hand_outside/self.c2)
                hand_contact_loss_outside = (left_hand_weights * left_hand_contact_loss_outside.view(-1,1)).mean() + \
                                            (right_hand_weights * right_hand_contact_loss_outside.view(-1,1)).mean()

            if hand_verts_inside.sum() > 0:
                left_hand_inside = v2v_min[self.hand_contact_prior[:ha]][hand_verts_inside[:ha]]
                right_hand_inside = v2v_min[self.hand_contact_prior[ha:]][hand_verts_inside[ha:]]
                if left_hand_inside.sum() > 0:
                    left_hand_contact_loss_inside = self.d1 * torch.tanh(left_hand_inside/self.d2)
                if right_hand_inside.sum() > 0:
                    right_hand_contact_loss_inside = self.d1 * torch.tanh(right_hand_inside/self.d2)
                hand_contact_loss_inside = left_hand_contact_loss_inside.mean() + right_hand_contact_loss_inside.mean()

            hand_contact_loss = self.hand_contact_prior_weight * (hand_contact_loss_inside + hand_contact_loss_outside)

        # ==== align normals of verts in contact ==== 接触顶点的法线对齐
        verts_close = torch.where(v2v_min < 0.01)[0]
        if self.angle_weight > 0 and len(verts_close) > 0:
            vertex_normals = compute_vertex_normals(body.vertices, self.cm.faces)
            dotprod_normals = torch.matmul(vertex_normals, torch.transpose(vertex_normals,1,2))[0]
            #normalsgather = dotprod_normals.gather(1, v2v_min_idx.view(-1,1))
            normalsgather = dotprod_normals[np.arange(nv), v2v_min_idx.view(-1,1)]
            angle_loss = 1 + normalsgather[verts_close,:]
            angle_loss = self.angle_weight * angle_loss.mean()

        # ==== penalize deviation from initial pose params ==== 惩罚偏离初始姿态参数
        # pose_prior_loss = self.pose_prior_weight * F.mse_loss(body.body_pose, self.init_pose, reduction='sum')
        pose_prior_loss = self.pose_prior_weight * F.mse_loss(body.body_pose, self.init_poses[idx], reduction='sum')

        # ==== penalize deviation from mean hand pose ==== 惩罚偏离平均手姿势
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
            outsidelossv2v = (outsidelossv2v * outsidelossv2vweights.to(device)).sum()
            outsideloss = self.outside_w * outsidelossv2v

        # ==== Total loss ====
        loss = contactloss + insideloss + outsideloss + pose_prior_loss + hand_pose_prior_loss + angle_loss + hand_contact_loss
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

        return loss, loss_dict
    
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
        dq_dt = torch.zeros((data_count,4),device=self.device,
                                  dtype=torch.float32)
        dq_dt[0][0] = 1 # 第0帧角速度设置为(1,0,0,0) 否则第一帧梯度为计算为nan
        dq_dt[1:] = (orientation[1:] - orientation[:-1]) / time_delta # 第1帧到最后一帧计算速度，1阶导
        # dq_dt[0] = dq_dt[1] # 第0帧角速度设置为与第1帧相同，使得第1帧角加速度为0，避免优化
        ## Divided difference for d2q_dt2
        d2q_dt2 = torch.zeros((data_count,4),device=self.device,
                                  dtype=torch.float32)
        
        d2q_dt2[2:] = (dq_dt[2:] - dq_dt[:-2]) / time_delta # 速度的第2帧到最后2帧计算加速度，2阶导
        # d2q_dt2[0][0] = 1 
        # d2q_dt2[1][0] = 1 # 前两帧角加速度设置为 1,0,0,0
        ## Calculate velocity
        velocity = 2 * self.quaternion_multiply(dq_dt, self.quaternion_inverse(orientation))
        ## Calculate acceleration
        temp = self.quaternion_multiply(dq_dt, self.quaternion_inverse(orientation))
        acceleration = 2 * (self.quaternion_multiply(d2q_dt2, self.quaternion_inverse(orientation)) - self.quaternion_multiply(temp,temp))
        return acceleration, velocity
    
    def calculate_smooth_loss_total(self, body_poses_total):
        smooth_loss = torch.tensor(0, device=self.device,
                              dtype=torch.float32)
        # for bone_idx in range(NUM_SMPLX_BODYJOINTS):
        target_bone_idx = [13,16,18,14,11]# UPPER_BOFY_INDEX
        for bone_idx in target_bone_idx: # 右手、手臂、肩膀
            orientation = torch.zeros((len(body_poses_total), 4),device=self.device, # 每个骨骼全部帧的四元数
                      dtype=torch.float32)
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
            smooth_loss += sum_of_top_values
        # smooth_loss /= NUM_SMPLX_BODYJOINTS
        return smooth_loss / len(target_bone_idx)
    
    def forward(self, total_body, mesh = None): # 4790
        # initialize loss tensors
        device = total_body[0].vertices.device
        vertices = total_body[0].vertices
        _, nv, _ = vertices.shape
        
        loss = torch.tensor(0.0, device=device)
        selfcontact_loss = torch.tensor(0.0, device=device)
        smooth_loss = torch.tensor(0.0, device=device)
        
        loss_dict = {
            'Total': loss.item(),
            'Contact': 0,
            'Inside': 0,
            'Outside': 0,
            'Angles': 0,
            'HandContact': 0,
            'HandPosePrior':0,
            'BodyPosePrior': 0,
            'Smooth': smooth_loss.item(),
        }
        start_time = time.time()
        for idx, body in enumerate(total_body):
            loss, loss_dict = self.single_forward(idx, body) # 每帧自己的loss
            selfcontact_loss += loss
            for key in loss_dict:
                loss_dict[key] += loss_dict[key]
        selfcontact_loss /= len(total_body)
        print("calculate selfcontact loss: ", time.time()-start_time)
        
        # start_time = time.time()
        # body_poses_total = [cur_body.body_pose for cur_body in total_body] # 获取每帧的pose
        # smooth_loss = self.smooth_weight * self.calculate_smooth_loss_total(body_poses_total) # 计算平滑loss
        # print("calculate smooth loss: ", time.time()-start_time)
        
        loss_dict['Smooth'] = smooth_loss.item()
        
        loss = selfcontact_loss + smooth_loss
        
        return loss, loss_dict