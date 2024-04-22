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

import sys
import torch
import torch.nn as nn
import numpy as np
import pickle
from .utils.mesh import batch_face_normals, \
                       batch_pairwise_dist, \
                       winding_numbers
from .body_segmentation import BatchBodySegment
from .utils.sparse import sparse_batch_mm

import os.path as osp
import time

def get_size_tensor(v2v):
    # 获取单个元素所占用的字节数
    element_size_bytes = v2v.element_size()

    # 获取张量的总元素个数
    total_elements = v2v.numel()
    # 计算总共占用的内存大小（以字节为单位）
    total_memory_bytes = element_size_bytes * total_elements
    total_memory_mb = total_memory_bytes / (1024 * 1024)
    print("张量 占用内存：{:.2f} MB".format(total_memory_mb))
        
class SelfContact(nn.Module):
    def __init__(self,
        geodesics_path='',
        hd_operator_path='',
        point_vert_corres_path='',
        segments_folder='',
        faces_path='',
        essentials_folder=None,
        geothres=0.3,
        euclthres=0.02,
        model_type='smplx',
        test_segments=True,
        compute_hd=False,
        buffer_geodists=False,
    ):
        super().__init__()

        # contact thresholds
        self.model_type = model_type
        self.euclthres = euclthres
        self.geothres = geothres
        self.test_segments = test_segments
        self.compute_hd = compute_hd

        if essentials_folder is not None:
            geodesics_path = osp.join(essentials_folder, 'geodesics', 
                model_type, f'{model_type}_neutral_geodesic_dist.npy')
            hd_operator_path = osp.join(essentials_folder, 'hd_model', 
                model_type, f'{model_type}_neutral_hd_vert_regressor_sparse.npz')
            point_vert_corres_path = osp.join(essentials_folder, 'hd_model', 
                model_type, f'{model_type}_neutral_hd_sample_from_mesh_out.pkl')
            faces_path =  osp.join(essentials_folder, 'models_utils', 
                model_type, f'{model_type}_faces.npy')
            segments_folder = osp.join(essentials_folder, 'segments', 
                model_type)
            segments_bounds_path = f'{segments_folder}/{model_type}_segments_bounds.pkl'

        # create faces tensor
        faces = np.load(faces_path)
        if type(faces) is not torch.Tensor:
            faces = torch.tensor(faces.astype(np.int64), dtype=torch.long)
        self.register_buffer('faces', faces)

        # create extra vertex and faces to close back of the mouth to maske
        # the smplx mesh watertight.
        if self.model_type == 'smplx':
            inner_mouth_verts_path = f'{segments_folder}/smplx_inner_mouth_bounds.pkl'
            vert_ids_wt = np.array(pickle.load(open(inner_mouth_verts_path, 'rb')))
            self.register_buffer('vert_ids_wt', torch.from_numpy(vert_ids_wt))
            faces_wt = [[vert_ids_wt[i+1], vert_ids_wt[i],
                faces.max().item()+1] for i in range(len(vert_ids_wt)-1)]
            faces_wt = torch.tensor(np.array(faces_wt).astype(np.int64),
                dtype=torch.long)
            faces_wt = torch.cat((faces, faces_wt), 0)
            self.register_buffer('faces_wt', faces_wt)

        # geodesic distance mask
        if geodesics_path is not None:
            geodesicdists = torch.Tensor(np.load(geodesics_path))
            if buffer_geodists:
                self.register_buffer('geodesicdists', geodesicdists)
            geodistmask = geodesicdists >= self.geothres
            self.register_buffer('geomask', geodistmask)

        # create batch segmentation here
        if self.test_segments:
            sxseg = pickle.load(open(segments_bounds_path, 'rb'))
            self.segments = BatchBodySegment(
                [x for x in sxseg.keys()], faces, segments_folder, self.model_type
            )

        # load regressor to get high density mesh
        if self.compute_hd:
            hd_operator = np.load(hd_operator_path)
            hd_operator = torch.sparse.FloatTensor(
                torch.tensor(hd_operator['index_row_col']),
                torch.tensor(hd_operator['values']),
                torch.Size(hd_operator['size']))
            self.register_buffer('hd_operator',
                torch.tensor(hd_operator).float())

            with open(point_vert_corres_path, 'rb') as f:
                hd_geovec = pickle.load(f)['faces_vert_is_sampled_from']
            self.register_buffer('geovec',
                torch.tensor(hd_geovec))
            self.register_buffer('geovec_verts', self.faces[self.geovec][:,0])

    def triangles(self, vertices):
        # get triangles (close mouth for smplx) # [1, 20940, 3, 3]
        if self.model_type == 'smpl':
            triangles = vertices[:,self.faces,:]
        elif self.model_type == 'smplx':
            mouth_vert = torch.mean(vertices[:,self.vert_ids_wt,:], 1, # 嘴巴部位的平均顶点
                        keepdim=True)
            vertices_mc = torch.cat((vertices, mouth_vert), 1) # 拼接 [1, 10475+1, 3]
            triangles = vertices_mc[:,self.faces_wt,:] # faces_wt [20940, 3]
        return triangles

    def get_intersection_mask(self, vertices, triangles, test_segments=True):
        """
            compute status of vertex: inside, outside, or colliding
        """

        bs, nv, _ = vertices.shape

        # split because of memory into two chunks
        exterior = torch.zeros((bs, nv), device=vertices.device, # [1, 10475] 在外部则为True 
            dtype=torch.bool)
        # exterior[:, :5000] = winding_numbers(vertices[:,:5000,:],
        #     triangles).le(0.99)
        # exterior[:, 5000:] = winding_numbers(vertices[:,5000:,:],
        #     triangles).le(0.99)

        N = 5  # 分成 N 段，可以调整cuda线程？
        chunk_size = vertices.shape[1] // N  # 计算每段的大小
        for i in range(N):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < N - 1 else vertices.shape[1]
            exterior[:, start_idx:end_idx] = winding_numbers(vertices[:, start_idx:end_idx, :], triangles).le(0.99)


        # check if intersections happen within segments
        if test_segments and not self.test_segments:
            assert('Segments not created. Create module with segments.')
            sys.exit()
        if test_segments and self.test_segments:
            for segm_name in self.segments.names:
                segm_vids = self.segments.segmentation[segm_name].segment_vidx
                for bidx in range(bs):
                    if (exterior[bidx, segm_vids] == 0).sum() > 0: # 该部位有顶点在内部
                        segm_verts = vertices[bidx, segm_vids, :].unsqueeze(0)
                        # 该部位外部顶点
                        segm_ext = self.segments.segmentation[segm_name] \
                            .has_self_isect_points(
                                segm_verts.detach(),
                                triangles[bidx].unsqueeze(0)
                        )
                        mask = ~segm_ext[bidx]
                        segm_idxs = torch.masked_select(segm_vids, mask) # 自交顶点的索引
                        true_tensor = torch.ones(segm_idxs.shape, device=segm_idxs.device, dtype=torch.bool)
                        exterior[bidx, segm_idxs] = true_tensor # 忽略这些自交顶点

        return exterior

    def get_hd_intersection_mask(self, points, triangles,
        faces_ioc_idx, hd_verts_ioc_idx, test_segments=False):
        """
            compute status of vertex: inside, outside, or colliding
        """
        bs, np, _ = points.shape
        assert bs == 1, 'HD points intersections only work with batch size 1.'

        # split because of memory into two chunks
        exterior = torch.zeros((bs, np), device=points.device,
            dtype=torch.bool)
        exterior[:, :6000] = winding_numbers(points[:,:6000,:],
            triangles).le(0.99)
        exterior[:, 6000:] = winding_numbers(points[:,6000:,:],
            triangles).le(0.99)

        return exterior

    def get_pairwise_dists(self, verts1, verts2, squared=False):
        """
            compute pairwise distance between vertices
        """
        v2v = batch_pairwise_dist(verts1, verts2, squared=squared)
        return v2v

    def segment_vertices(self, vertices, compute_hd=False, test_segments=True):
        """
            get self-intersecting vertices and pairwise distance, 获取自交顶点和顶点间距离
        """
        bs = vertices.shape[0]

        triangles = self.triangles(vertices.detach()) # [1, 20940, 3, 3]

        # start = time.time()
        # get inside / outside segmentation
        exterior = self.get_intersection_mask(
                vertices.detach(),
                triangles.detach(),
                test_segments
        )
        # torch.cuda.synchronize()
        # print('get intersection vertices: {:5f}'.format(time.time() - start))

        # get pairwise distances of vertices
        v2v = self.get_pairwise_dists(vertices, vertices, squared=False) # [1, 10475, 10475] 每帧都计算？
        # v2v.to(torch.float16)
        v2v_mask = v2v.detach().clone()
        # inf_tensor = float('inf') * torch.ones((1,(~self.geomask).sum().item()), device=v2v.device) # [1, 41639239] 不满足测地距离，设置顶点之间距离为inf
        v2v_mask[:, ~self.geomask] = float('inf')# inf_tensor # 测地距离小的设置为Inf，避免计算
        _, v2v_min_index = torch.min(v2v_mask, dim=1) # 每个顶点最近的顶点索引
        v2v_min = torch.gather(v2v, dim=2,
            index=v2v_min_index.view(bs,-1,1)).squeeze(-1) # 每个顶点最近的顶点之间的距离（唯一）
        incontact = v2v_min < self.euclthres # [1, 10475] 接触Mask

        hd_v2v_min, hd_exterior, hd_points, hd_faces_in_contact = None, None, None, None
        if compute_hd:
            hd_v2v_min, hd_exterior, hd_points, hd_faces_in_contact = \
                self.segment_hd_points(
                    vertices, v2v_min, incontact, exterior, test_segments)

        v2v_out = (v2v_min, incontact, exterior)
        hd_v2v_out = (hd_v2v_min, hd_exterior, hd_points, hd_faces_in_contact)

        return v2v_out, hd_v2v_out

    def segment_vertices_scopti(self, vertices, test_segments=True):
        """
            指定是否过滤特定部位
            get self-intersecting vertices and pairwise distance 
            for self-contact optimization. This version is determinisic.
        """
        bs, nv, _ = vertices.shape
        if bs > 1:
            sys.exit('Please use batch size one or set use_pytorch_norm=False')

        # get pairwise distances of vertices
        v2v = vertices.squeeze().unsqueeze(1).expand(nv, nv, 3) - \
                vertices.squeeze().unsqueeze(0).expand(nv, nv, 3)
        v2v = torch.norm(v2v, dim=2).unsqueeze(0)
        # v2v.to(torch.float16)
        with torch.no_grad():
            triangles = self.triangles(vertices.detach())

            start = time.time()
            # get inside / outside segmentation
            exterior = self.get_intersection_mask(
                    vertices.detach(),
                    triangles.detach(),
                    test_segments
            )
            # torch.cuda.synchronize()
            # print('get intersection vertices: {:5f}'.format(time.time() - start))

            v2v_mask = v2v.detach().clone()
            #v2v_mask[:, ~self.geomask] = float('inf')
            inf_tensor = float('inf') * torch.ones((1,(~self.geomask).sum().item()), device=v2v.device) # 不满足测地距离，设置顶点之间距离为inf
            v2v_mask[:, ~self.geomask] = inf_tensor
            _, v2v_min_index = torch.min(v2v_mask, dim=1)

        #v2v_min = torch.gather(v2v, dim=2,
        #    index=v2v_min_index.view(bs,-1,1)).squeeze(-1)
        v2v_min = v2v[:, np.arange(nv), v2v_min_index[0]]

        return (v2v_min, v2v_min_index, exterior)

    def segment_points_scopti_orig(self, points, vertices):
        """
            只计算指定顶点是否穿模，默认不过滤特定部位
            get self-intersecting points (vertices on extremities) and pairwise distance
            for self-contact optimization. This version is determinisic.
        """
        bs, nv, _ = vertices.shape
        if bs > 1:
            sys.exit('Please use batch size one or set use_pytorch_norm=False')

        start = time.time()
        v2v = (vertices.squeeze().unsqueeze(1).expand(nv, nv, 3) - \
                vertices.squeeze().unsqueeze(0).expand(nv, nv, 3)) # 6053->7308
        v2v = torch.norm(v2v, dim=2).unsqueeze(0) # 已计算过？
        print('get v2v distance: {:5f}'.format(time.time() - start))
        
        # find closest vertex in contact
        with torch.no_grad():
            triangles = self.triangles(vertices.detach()) # [1, 20940, 3, 3] 已计算过？

            start = time.time()
            # get inside / outside segmentation  # 7311 -> 6469
            exterior = self.get_intersection_mask( # [1, 3889] 外部为True
                    vertices=points.detach(), # 顶点 [3889, 3]
                    triangles=triangles.detach(), # 面 [1, 20940, 3, 3]
                    test_segments=False
            )            
            # torch.cuda.synchronize()
            print('get intersection vertices: {:5f}'.format(time.time() - start))

            v2v_mask = v2v.detach().clone()# 6479 -> 6896
            #v2v_mask[:, ~self.geomask] = float('inf')
            inf_tensor = float('inf') * torch.ones((1,(~self.geomask).sum().item()), device=v2v.device) # 6896 -> 7730
            v2v_mask[:, ~self.geomask] = inf_tensor
            _, v2v_min_index = torch.min(v2v_mask, dim=1)

        # first version is better, but not deterministic
        #v2v_min = torch.gather(v2v, dim=2,
        #    index=v2v_min_index.view(bs,-1,1)).squeeze(-1)
        v2v_min = v2v[:, np.arange(nv), v2v_min_index[0]] # [1, 10475]
        
        return (v2v_min, v2v_min_index, exterior)

    def segment_points_scopti_CHAT1(self, points, vertices):
        """
        get self-intersecting points (vertices on extremities) and pairwise distance
        for self-contact optimization. This version is determinisic.
        """
        bs, nv, _ = vertices.shape
        if bs > 1:
            sys.exit('Please use batch size one or set use_pytorch_norm=False')

        start = time.time()
        v2v_min_list = []
        v2v_min_index_list = []

        for i in range(nv):
            diff = vertices.squeeze() - vertices.squeeze()[i]
            v2v = torch.norm(diff, dim=1)
            v2v_min_index = torch.argmin(v2v)
            v2v_min = v2v[v2v_min_index]
            v2v_min_list.append(v2v_min)
            v2v_min_index_list.append(v2v_min_index)

        v2v_min = torch.tensor(v2v_min_list).unsqueeze(0)
        v2v_min_index = torch.tensor(v2v_min_index_list).unsqueeze(0)

        print('get v2v distance: {:5f}'.format(time.time() - start))

        # find closest vertex in contact
        with torch.no_grad():
            triangles = self.triangles(vertices.detach())  # [1, 20940, 3, 3]

            start = time.time()
            # get inside / outside segmentation
            exterior = self.get_intersection_mask(
                vertices=points.detach(),  # 顶点 [3889, 3]
                triangles=triangles.detach(),  # 面 [1, 20940, 3, 3]
                test_segments=False
            )
            print('get intersection vertices: {:5f}'.format(time.time() - start))

        return (v2v_min, v2v_min_index, exterior)
    
    def segment_points_scopti(self, points, vertices):
        """
            只计算指定顶点是否穿模，默认不过滤特定部位
            get self-intersecting points (vertices on extremities) and pairwise distance
            for self-contact optimization. This version is determinisic.
        """
        bs, nv, _ = vertices.shape
        if bs > 1:
            sys.exit('Please use batch size one or set use_pytorch_norm=False')

        start = time.time()
        # vertices = vertices.to(torch.float16)
        # v2v = (vertices.squeeze().unsqueeze(1).expand(nv, nv, 3) - \
        #         vertices.squeeze().unsqueeze(0).expand(nv, nv, 3)) # 6053->7308          600MB
        # v2v = torch.norm(v2v, dim=2).unsqueeze(0) # 10475*10475*4 / (1024*1024) ，使用float16也要约418MB显存  400MB
        # 计算两两顶点之间的欧式距离
        v1 = vertices.unsqueeze(2)  # 添加一个维度，形状变为 (1, 10475, 1, 3)
        v2 = vertices.unsqueeze(1)  # 添加一个维度，形状变为 (1, 1, 10475, 3)
        v2v = torch.norm(v1 - v2, dim=-1)  # 计算欧式距离，dim=-1 表示沿着最后一个维度计算
        
        # print('get v2v distance: {:5f}'.format(time.time() - start))
        
        # find closest vertex in contact
        with torch.no_grad():
            triangles = self.triangles(vertices.detach()) # [1, 20940, 3, 3] 已计算过？

            # start = time.time()
            # get inside / outside segmentation  # 7311 -> 6469
            exterior = self.get_intersection_mask( # [1, 3889] 外部为True
                    vertices=points.detach(), # 顶点 [3889, 3]
                    triangles=triangles.detach(), # 面 [1, 20940, 3, 3]
                    test_segments=False
            )            
            # torch.cuda.synchronize()
            # print('get intersection vertices: {:5f}'.format(time.time() - start))

            v2v_mask = v2v.detach().clone() # 6479 -> 6896
            #v2v_mask[:, ~self.geomask] = float('inf')
            # inf_tensor = float('inf') * torch.ones((1,(~self.geomask).sum().item()), device=v2v.device) # 6896 -> 7730
            v2v_mask[:, ~self.geomask] = float('inf')# inf_tensor
            _, v2v_min_index = torch.min(v2v_mask, dim=1)
            v2v_min_index.to(torch.int16)
        # first version is better, but not deterministic
        # v2v_min = torch.gather(v2v, dim=2,
        #    index=v2v_min_index.view(bs,-1,1)).squeeze(-1)
        
        # v2v = v2v.to(torch.float16)
        v2v_min = v2v[:, np.arange(nv), v2v_min_index[0]] # [1, 10475]
        return (v2v_min, v2v_min_index, exterior)
        # return v2v_min
       
    def segment_points_scopti_ds(self, ds, ds_vertices, vertices):
        nv_ds = ds.shape[0] # ds中顶点的数量
        # ds_vertices_flat = ds_vertices.view(-1, 3)
        # # ds_vertices_flat = points.view(-1, 3)
        # vertices_flat = vertices.view(-1, 3)

        # 计算顶点之间的距离差的平方
        # ds_vertices_flat_expanded = ds_vertices_flat.unsqueeze(1)  # 形状变为[460, 1, 3]
        # vertices_flat_expanded = vertices_flat.unsqueeze(0)        # 形状变为[1, 10475, 3]
        v2v_ds = torch.norm(ds_vertices.view(-1, 3).unsqueeze(1) - vertices.view(-1, 3).unsqueeze(0), dim=2)
        # distances_squared = ((ds_vertices_flat.unsqueeze(1) - vertices_flat.unsqueeze(0)) ** 2).sum(dim=2)
        # # 计算距离的平方根
        # v2v_ds = torch.sqrt(distances_squared)
        
        with torch.no_grad():
            triangles = self.triangles(vertices.detach()) # [1, 20940, 3, 3] 已计算过？

            # start = time.time()
            # get inside / outside segmentation  # 7311 -> 6469
            exterior = self.get_intersection_mask( # [1, 3889] 外部为True
                    vertices=ds_vertices.detach(), # 顶点 [3889, 3]
                    triangles=triangles.detach(), # 面 [1, 20940, 3, 3]
                    test_segments=False
            )            
            # torch.cuda.synchronize()
            # print('get intersection vertices: {:5f}'.format(time.time() - start))

            ds_geomask = self.geomask[ds] # ds中满足测地距离 [460, 10475]
            v2v_ds_mask = v2v_ds.detach().clone()
            v2v_ds_mask[~ds_geomask] = float('inf') # [460, 10475]
            _, v2v_ds_min_index = torch.min(v2v_ds_mask, dim=1) # 没有grad_fn=<IndexBackward>
        v2v_ds_min = v2v_ds[np.arange(nv_ds), v2v_ds_min_index] # [1, 10475] 多了个grad_fn=<IndexBackward>
        
        return (v2v_ds_min, v2v_ds_min_index, exterior)
    
    def get_hand_vertices_min(self, hand_idx, hand_vertices, vertices):
        """
            手部顶点对应的最短距离
        """
        nv_hand = hand_idx.shape[0] # ds中顶点的数量
        v2v_hand = torch.norm(hand_vertices.view(-1, 3).unsqueeze(1) - vertices.view(-1, 3).unsqueeze(0), dim=2)
        with torch.no_grad():
            ds_geomask = self.geomask[hand_idx] # ds中满足测地距离 [460, 10475]
            v2v_hand_mask = v2v_hand.detach().clone()
            v2v_hand_mask[~ds_geomask] = float('inf') # [460, 10475]
            _, v2v_hand_min_index = torch.min(v2v_hand_mask, dim=1)
        v2v_hand_min = v2v_hand[np.arange(nv_hand), v2v_hand_min_index]
        return v2v_hand_min
    
    def segment_points_scopti_KIMI(self, points, vertices):
        """
            get self-intersecting points (vertices on extremities) and pairwise distance
            for self-contact optimization. This version is deterministic.
        """
        bs, nv, _ = vertices.shape
        if bs > 1:
            sys.exit('Please use batch size one or set use_pytorch_norm=False')

        start = time.time()
        # 计算顶点之间的距离矩阵，避免重复计算
        vertices_expanded = vertices.clone().detach().squeeze().unsqueeze(1).expand(nv, nv, 3)
        v2v = torch.norm(vertices_expanded - vertices_expanded.transpose(0, 1), dim=2).unsqueeze(0)
        print('get v2v distance: {:5f}'.format(time.time() - start))

        with torch.no_grad():
            triangles = self.triangles(vertices.detach()) # [1, 20940, 3, 3]

            start = time.time()
            # 使用更小的数据类型，如果精度要求允许
            exterior = self.get_intersection_mask(
                vertices=points.detach().float(), # 顶点 [3889, 3]
                triangles=triangles.detach(),
                test_segments=False
            )
            print('get intersection vertices: {:5f}'.format(time.time() - start))

            # 减少中间变量的存储，直接在 v2v 上操作
            v2v_mask = v2v.cpu().clone().detach()
            v2v_mask[:, ~self.geomask] = float('inf')
            _, v2v_min_index = torch.min(v2v_mask, dim=1)

            # 使用 in-place 操作减少内存使用
            v2v_min = v2v[:, np.arange(nv), v2v_min_index[0]].clamp_min_(0) # [1, 10475]

        del v2v_mask, v2v  # 释放不再使用的变量

        return (v2v_min, v2v_min_index, exterior)
    
    def segment_hd_points(self, vertices, v2v_min, incontact, exterior, test_segments=True):
        """
            compute hd points from vertices and compute their distance
            and inside / outside segmentation
        """
        bs, nv, _ = vertices.shape

        # select all vertices that are inside or in contact
        verts_ioc = incontact | ~exterior
        verts_ioc_idxs = torch.nonzero(verts_ioc)
        verts_ioc_idx = torch.nonzero(verts_ioc[0]).view(-1)

        # get hd points for inside or in contact vertices

        # get hd points for inside or in contact vertices
        hd_v2v_mins = []
        hd_exteriors = []
        hd_points = []
        hd_faces_in_contacts = []
        for idx in range(bs):
            verts_ioc_idx = torch.nonzero(verts_ioc[idx]).view(-1)
            exp1 = verts_ioc_idx.expand(self.faces.flatten().shape[0], -1)
            exp2 = self.faces.flatten().unsqueeze(-1).expand(-1, verts_ioc_idx.shape[0])
            nzv = (exp1 == exp2).any(1).reshape(-1, 3).any(1)
            faces_ioc_idx = torch.nonzero(nzv).view(-1)
            hd_verts_ioc_idx = (self.geovec.unsqueeze(-1) == \
                faces_ioc_idx.expand(self.geovec.shape[0], -1)).any(1)


            # check is points were samples from the vertices in contact
            if hd_verts_ioc_idx.sum() > 0:
                hd_verts_ioc = sparse_batch_mm(self.hd_operator, vertices[[idx]])[:,hd_verts_ioc_idx,:]
                triangles = self.triangles(vertices[[idx]])
                face_normals = batch_face_normals(triangles)[0]
                with torch.no_grad():
                    hd_v2v = self.get_pairwise_dists(hd_verts_ioc, hd_verts_ioc, squared=True)
                    geom_idx = self.geovec_verts[hd_verts_ioc_idx]
                    hd_geo = self.geomask[geom_idx,:][:,geom_idx]
                    #hd_v2v[:, ~hd_geo] = float('inf')
                    inf_tensor = float('inf') * torch.ones((1,(~hd_geo).sum().item()), device=hd_v2v.device)
                    #hd_v2v[:, ~self.geomask] = inf_tensor
                    hd_v2v[:, ~hd_geo] = inf_tensor
                    hd_v2v_min, hd_v2v_min_idx = torch.min(hd_v2v, dim=1)

                    # add little offset to those vertices for in/ex computation
                    faces_ioc_idx = self.geovec[hd_verts_ioc_idx]
                    hd_verts_ioc_offset = hd_verts_ioc + \
                        0.001 * face_normals[faces_ioc_idx, :].unsqueeze(0)

                    # test if hd point is in- or outside
                    hd_exterior = self.get_hd_intersection_mask(
                        hd_verts_ioc_offset.detach(),
                        triangles.detach(),
                        faces_ioc_idx=faces_ioc_idx,
                        hd_verts_ioc_idx=hd_verts_ioc_idx,
                        test_segments=False,
                    )[0]

                    hd_close_faces = torch.vstack((faces_ioc_idx, faces_ioc_idx[hd_v2v_min_idx[0]]))
                    hd_verts_in_close_contact = hd_v2v_min < 0.005**2
                    hd_faces_in_contact = hd_close_faces[:, hd_verts_in_close_contact[0]]

                hd_v2v_min = torch.norm(hd_verts_ioc[0] - hd_verts_ioc[0, hd_v2v_min_idx, :], dim=2)[0]
                hd_v2v_mins += [hd_v2v_min]
                hd_faces_in_contacts += [hd_faces_in_contact]
                hd_exteriors += [hd_exterior]
                hd_points += [hd_verts_ioc_offset]
            else:
                hd_v2v_mins += [None]
                hd_exteriors += [None]
                hd_points += [None]
                hd_faces_in_contacts += [None]

        return hd_v2v_mins, hd_exteriors, hd_points, hd_faces_in_contacts



class SelfContactSmall(nn.Module):
    def __init__(self,
        essentials_folder,
        geothres=0.3,
        euclthres=0.02,
        model_type='smplx',
    ):
        super().__init__()

        self.model_type = model_type

        # contact thresholds
        self.euclthres = euclthres
        self.geothres = geothres

        # geodesic distance mask
        geodesics_path = osp.join(essentials_folder, 'geodesics', model_type, f'{model_type}_neutral_geodesic_dist.npy')
        geodesicdists = torch.Tensor(np.load(geodesics_path))
        geodistmask = geodesicdists >= self.geothres
        self.register_buffer('geomask', geodistmask)

        # create faces tensor
        faces_path =  osp.join(essentials_folder, 'models_utils', model_type, f'{model_type}_faces.npy')
        faces = np.load(faces_path)
        if type(faces) is not torch.Tensor:
            faces = torch.tensor(faces.astype(np.int64), dtype=torch.long)
        self.register_buffer('faces', faces)

    def get_pairwise_dists(self, verts1, verts2, squared=False):
        """
            compute pairwise distance between vertices
        """
        v2v = batch_pairwise_dist(verts1, verts2, squared=squared)

        return v2v
  
    def pairwise_selfcontact_for_verts(self, vertices):
            """
            Returns tensor of vertex pairs that are in contact. If you have a batch of vertices,
            the number of vertices returned per mesh can be different. To get verts in contact 
            for batch_index_x use:
            batch_x_verts_in_contact = contact[torch.where(in_contact_batch_idx == batch_index_x)[0], :]
            """

            # get pairwise distances of vertices
            v2v = self.get_pairwise_dists(vertices, vertices, squared=True)

            # find closes vertex in contact
            v2v[:, ~ self.geomask] = float('inf')
            v2v_min, v2v_min_index = torch.min(v2v, dim=1)
            in_contact_batch_idx, in_contact_idx1 = torch.where(v2v_min < self.euclthres**2)
            in_contact_idx2 = v2v_min_index[in_contact_batch_idx, in_contact_idx1]
            contact = torch.vstack((in_contact_idx1, in_contact_idx2)).view(-1,2,1)

            return in_contact_batch_idx, contact


    def verts_in_contact(self, vertices, return_idx=False):

            # get pairwise distances of vertices
            v2v = self.get_pairwise_dists(vertices, vertices, squared=True)

            # mask v2v with eucledean and geodesic dsitance
            euclmask = v2v < self.euclthres**2
            mask = euclmask * self.geomask

            # find closes vertex in contact
            in_contact = mask.sum(1) > 0

            if return_idx:
                in_contact = torch.where(in_contact)

            return in_contact
