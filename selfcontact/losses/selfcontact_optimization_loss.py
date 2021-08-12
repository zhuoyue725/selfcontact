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

class SelfContactOptiLoss(nn.Module):
    def __init__(self,
        contact_module,
        inside_loss_weight,
        outside_loss_weight,
        contact_loss_weight,
        hand_contact_prior_weight,
        hand_contact_prior_path,
        downsample,
        device,
    ):
        super().__init__()

        self.ds = downsample

        # weights
        self.inside_w = inside_loss_weight
        self.contact_w = contact_loss_weight
        self.outside_w = outside_loss_weight
        self.hand_contact_prior_weight = hand_contact_prior_weight

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

    def configure(self, body_model, params):

        # initial mesh and pose 
        vertices = body_model(**params).vertices
        self.register_buffer('init_pose', body_model.body_pose.clone().detach())
        self.register_buffer('init_verts', vertices.clone().detach())

        # get verts in contact in initial mesh
        with torch.no_grad():
            self.init_verts_in_contact_idx = \
                self.cm.segment_vertices(self.init_verts, test_segments=False)[0][1][0]
            self.init_verts_in_contact = torch.where(self.init_verts_in_contact_idx)[0].cpu().numpy()

    def forward(self, body):
        """
            compute loss based on status of vertex
        """

        cols = 255 * np.ones((body.vertices.shape[1], 4))
        #cols[self.loss.init_verts_in_contact, :2] = 1
           

        # initialize loss tensors
        device = body.vertices.device
        vertices = body.vertices

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

        (v2v_min, v2v_min_idx, verts_in_contact, exterior), \
        (hd_v2v_min, hd_exterior, hd_points, hd_faces_in_contact) \
        = self.cm.segment_vertices(
            vertices,
            compute_hd=False, #self.use_hd,
            test_segments=False, #test_segments=self.test_segments,
            use_pytorch_norm=True,
            return_v2v_min_idx=True
        )
        v2v_min = v2v_min.squeeze()

        # only extrimities intersect
        inside = torch.zeros_like(exterior).to(exterior.device).to(torch.bool)
        inside[:,self.ds[~exterior[0, self.ds]]] = True
        inside = inside.squeeze()

        # ==== contact loss ====
        # compute attraction weight depending on geodesic distance to inital verts in contact
        if inside.sum() > 0:
            v2vinside = v2v_min[inside]
            v2vinside = 0.07 * torch.tanh(v2vinside / 0.06)
            insideloss = 0.5 * self.inside_w * v2vinside.mean()

        if (~inside).sum() > 0:
            if len(self.init_verts_in_contact) > 0:
                weights_outside = 1 / (5 * self.geodist[:, self.init_verts_in_contact].min(1)[0] + 1)
            else:
                weights_outside = torch.ones_like(self.geodist)[:,0].to(v2vinside.device)
            #cols[:, 0] = 255 * weights_outside.cpu().numpy()
            attaction_weights = weights_outside[self.ds][~inside[self.ds]]
            v2voutside = v2v_min[self.ds][~inside[self.ds]]
            v2voutside = 0.04 * attaction_weights  * torch.tanh(v2voutside/0.04)
            contactloss = 0.5 * self.contact_w * v2voutside.mean()

        # ==== hand-on-body prior loss ====
        ha = int(self.hand_contact_prior.shape[0] / 2)
        hand_verts_inside = inside[self.hand_contact_prior]
        if hand_verts_inside.sum() > 0:
            left_hand_inside = v2v_min[self.hand_contact_prior[:ha]][hand_verts_inside[:ha]]
            right_hand_inside = v2v_min[self.hand_contact_prior[ha:]][hand_verts_inside[ha:]]
            if left_hand_inside.sum() > 0:
                left_hand_contact_loss_inside = 0.023 * torch.tanh(left_hand_inside/0.02)
            if right_hand_inside.sum() > 0:
                right_hand_contact_loss_inside = 0.023 * torch.tanh(right_hand_inside/0.02)
            hand_contact_loss_inside = left_hand_contact_loss_inside.mean() + right_hand_contact_loss_inside.mean()
        if (~hand_verts_inside).sum() > 0:
            left_hand_outside = v2v_min[self.hand_contact_prior[:ha]][(~hand_verts_inside)[:ha]]
            right_hand_outside = v2v_min[self.hand_contact_prior[ha:]][(~hand_verts_inside)[ha:]]
            left_hand_weights = (0.1*(-1*(self.hand_contact_prior_weights[:ha].view(-1,1)[(~hand_verts_inside)[:ha]].view(-1,1))+1)+0.9)
            right_hand_weights = (0.1*(-1*(self.hand_contact_prior_weights[ha:].view(-1,1)[(~hand_verts_inside)[ha:]].view(-1,1))+1)+0.9)
            if left_hand_outside.sum() > 0:
                left_hand_contact_loss_outside = 0.01 * torch.tanh(left_hand_outside/0.01)
            if right_hand_outside.sum() > 0:
                right_hand_contact_loss_outside = 0.01 * torch.tanh(right_hand_outside/0.01)
            hand_contact_loss_outside = (left_hand_weights * left_hand_contact_loss_outside.view(-1,1)).mean() + \
                                        (right_hand_weights * right_hand_contact_loss_outside.view(-1,1)).mean()
        hand_contact_loss = 0.2 * self.hand_contact_prior_weight * (0.5 * hand_contact_loss_inside + 0.5 * hand_contact_loss_outside)

        # ==== align normals of verts in contact ====
        verts_close = torch.where(v2v_min < 0.01)[0]
        if len(verts_close) > 0:
            vertex_normals = compute_vertex_normals(body.vertices, self.cm.faces)
            dotprod_normals = torch.matmul(vertex_normals, torch.transpose(vertex_normals,1,2))[0]
            normalsgather = dotprod_normals.gather(1, v2v_min_idx.view(-1,1))
            angle_loss = 1 + normalsgather[verts_close,:]
            angle_loss = 0.001 * angle_loss.mean()

        # ==== penalize deviation from initial pose params ====
        pose_prior_loss = F.mse_loss(body.body_pose,
                            self.init_pose, reduction='sum')

        hand_pose_prior_loss = 0.001 * (self.hand_pose_prior(body.left_hand_pose) + \
                               self.hand_pose_prior(body.right_hand_pose))

        # ==== pose regression loss / outside loss ====
        outsidelossv2v = torch.norm(self.init_verts-body.vertices, dim=2)
        if self.init_verts_in_contact.sum() > 0:
            outsidelossv2vweights = (2 * self.geodist[:, self.init_verts_in_contact].min(1)[0].view(body.vertices.shape[0], -1))**2
        else:
            outsidelossv2vweights = torch.ones_like(outsidelossv2v).to(device)
        outsidelossv2v = (outsidelossv2v * outsidelossv2vweights).sum()
        outsideloss = self.outside_w * outsidelossv2v

        # ==== Total loss ====
        loss =  contactloss + insideloss + outsideloss + angle_loss + pose_prior_loss + hand_pose_prior_loss + hand_contact_loss
        loss_dict = {
            'Total': loss.item(),
            'Outside': outsideloss.item(),
            'Inside': insideloss.item(),
            'Contact': contactloss.item(),
            'Angles': angle_loss.item(),
            'HandPosePrior':hand_pose_prior_loss.item(),
            'PosePrior': pose_prior_loss.item(),
        }

        return loss, loss_dict, inside, cols