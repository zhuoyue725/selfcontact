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

import torch 
import trimesh
import os

import numpy as np

from ..utils.output import printlosses
from ..utils.visualization import save_npz_file
import time

class SelfContactAnimOpti():
    def __init__(
        self,
        loss,
        optimizer_name='adam',
        optimizer_lr_body=0.01,
        optimizer_lr_hands=0.01,
        max_iters=100,
        loss_thres=1e-5,
        patience=5,
        output_folder=None,
        npz_file=None,
        file_base_name = None,
        save_step = 10,
        cfg = None
    ):
        super().__init__()

        # create optimizer
        self.optimizer_name =  optimizer_name 
        self.optimizer_lr_body = optimizer_lr_body
        self.optimizer_lr_hands = optimizer_lr_hands
        self.max_iters = max_iters
        self.loss_thres = loss_thres
        self.patience = patience

        # self-contact optimization loss
        self.loss = loss
        
        # save animation tmp result
        self.output_folder = output_folder
        self.npz_file = npz_file
        self.file_base_name = file_base_name
        
        self.save_step = save_step
        
        self.cfg = cfg
        
    def get_optimizer(self, model):
        if self.optimizer_name == 'adam':
            optimizer = torch.optim.Adam([
                {'params': model.body_pose, 'lr': self.optimizer_lr_body},
                {'params': [model.left_hand_pose, model.right_hand_pose], 'lr': self.optimizer_lr_hands},
            ])
        return optimizer

    def get_anim_optimizer(self, total_body_model):
        if self.optimizer_name == 'adam':
            # optimizer = torch.optim.Adam([
            #     {'params': [cur_body.body_pose for cur_body in total_body_model], 'lr': self.optimizer_lr_body}
            #     # {'params': [[cur_body.left_hand_pose, cur_body.right_hand_pose] for cur_body in total_body_model], 'lr': self.optimizer_lr_hands},
            # ])
            optimizer = torch.optim.Adam([cur_body.body_pose for cur_body in total_body_model], lr=self.optimizer_lr_body)
        return optimizer
    
    def run(self, total_body_model, params, assign_frame_idx = -1):

        # create optimizer 
        optimizer = self.get_anim_optimizer(total_body_model)
        
        # initial total body
        # body = body_model(
        #     get_skin=True,
        #     global_orient=params['global_orient'],
        #     betas=params['betas']
        # )
        total_output_body_mesh = [cur_body(
            get_skin=True,
            global_orient=params['global_orient'],
            betas=params['betas'] # 所有帧都相同
        ) for cur_body in total_body_model]
        
        print('start configure loss with initial mesh')
        start = time.time()
        # configure loss with initial mesh
        self.loss.configure(total_body_model, total_output_body_mesh) # 2529 -> 4876
        # torch.cuda.synchronize()
        print('configure loss with initial mesh: {:5f}'.format(time.time() - start))
        
        # Initialize optimization
        step = 0
        criterion_loss = True
        loss_old = 0
        count_loss = 0

        # run optimization
        while step < self.max_iters and criterion_loss:
            
            start_step = time.time()
            optimizer.zero_grad()

            # get current total model
            total_output_body_mesh = [cur_body(
                get_skin=True,
                global_orient=params['global_orient'],
                betas=params['betas'] # 所有帧都相同
            ) for cur_body in total_body_model]

            # compute loss
            if assign_frame_idx >=0 and (step % 10 == 0 or step == self.max_iters-1):
                total_loss, loss_dict = self.loss(total_output_body_mesh,total_body_model[0])
            elif assign_frame_idx < 0 and (step % self.save_step == 0 or step == self.max_iters-1) and step > 0:
                save_npz_file(self.output_folder, self.npz_file, total_output_body_mesh, self.file_base_name, self.cfg)
                total_loss, loss_dict = self.loss(total_output_body_mesh)
            else:
                total_loss, loss_dict = self.loss(total_output_body_mesh)
                
            print(step, printlosses(loss_dict))

            # =========== stop criterion based on loss ===========
            with torch.no_grad():
                count_loss = count_loss + 1 if abs(total_loss - loss_old) < self.loss_thres else 0
                if count_loss >= self.patience:
                    criterion_loss = False

                loss_old = total_loss
                
                step += 1

            # back prop
            start_time = time.time()
            total_loss.backward(retain_graph=False)
            print('back prop: {:5f}'.format(time.time() - start_time))
            # for i in range(11):
            #     print(total_output_body_mesh[i].body_pose.grad)
            optimizer.step()
            torch.cuda.empty_cache()
            print('Step Optimization: {:5f}'.format(time.time() - start_step))

        return total_output_body_mesh # 蒙皮后的bodys
