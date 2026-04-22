import functools
import math
import os
import time


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from utils.graphics_utils import apply_rotation, batch_quaternion_multiply
from scene.hexplane import HexPlaneField


class Deformation(nn.Module):
    def __init__(self, D=8, W=256, input_ch=27, input_ch_time=9, grid_pe=0, skips=[], args=None):
        super(Deformation, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_time = input_ch_time
        self.skips = skips
        self.grid_pe = grid_pe
        self.no_grid = args.no_grid
        self.grid = HexPlaneField(args.bounds, args.kplanes_config, args.multires)
        
        self.args = args
        self.use_class_deformation = getattr(args, 'use_class_deformation', False)
        if self.args.static_mlp:
            self.static_mlp = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))
        
        self.ratio=0
        self.create_net()
        
    @property
    def get_aabb(self):
        return self.grid.get_aabb
    
    def set_aabb(self, xyz_max, xyz_min):
        print("Deformation Net Set aabb",xyz_max, xyz_min)
        self.grid.set_aabb(xyz_max, xyz_min)
        if self.args.empty_voxel:
            self.empty_voxel.set_aabb(xyz_max, xyz_min)
    
    def create_net(self):
        mlp_out_dim = 0
        if self.grid_pe !=0:
            grid_out_dim = self.grid.feat_dim+(self.grid.feat_dim)*2 
        else:
            grid_out_dim = self.grid.feat_dim
        if self.no_grid:
            self.feature_out = [nn.Linear(4,self.W)]
        else:
            self.feature_out = [nn.Linear(mlp_out_dim + grid_out_dim ,self.W)]
        
        for i in range(self.D-1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W,self.W))
        self.feature_out = nn.Sequential(*self.feature_out)
        # Elastic decoder heads (tissue/vessel - full deformation)
        self.pos_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3))
        self.scales_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3))
        self.rotations_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 4))
        self.opacity_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))
        self.shs_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 16*3))
        
        # Phase 2: Rigid decoder heads (tools - rotation + translation only)
        if self.use_class_deformation:
            self.rigid_pos_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3))
            self.rigid_rotations_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 4))

    def query_time(self, rays_pts_emb, scales_emb, rotations_emb, time_feature, time_emb):
        
        # only encode the pts and time
        if self.no_grid:
            hidden = torch.cat([rays_pts_emb[:,:3],time_emb[:,:1]],-1)
        else:
            # multi-scale feature [327680, 64]
            grid_feature = self.grid(rays_pts_emb[:,:3], time_emb[:,:1])
            if self.grid_pe > 1:
                grid_feature = poc_fre(grid_feature, self.grid_pe)
            hidden = torch.cat([grid_feature],-1) 
        hidden = self.feature_out(hidden)   
 
        return hidden
    @property
    def get_empty_ratio(self):
        return self.ratio
    
    def forward(self, rays_pts_emb, scales_emb=None, rotations_emb=None, opacity = None,shs_emb=None, time_feature=None, time_emb=None, semantic_labels=None):
        if time_emb is None:
            return self.forward_static(rays_pts_emb[:,:3])
        else:
            return self.forward_dynamic(rays_pts_emb, scales_emb, rotations_emb, opacity, shs_emb, time_feature, time_emb, semantic_labels)

    def forward_static(self, rays_pts_emb):
        grid_feature = self.grid(rays_pts_emb[:,:3])
        dx = self.static_mlp(grid_feature)
        return rays_pts_emb[:, :3] + dx
    
    def forward_dynamic(self, rays_pts_emb, scales_emb, rotations_emb, opacity_emb, shs_emb, time_feature, time_emb, semantic_labels=None):
        hidden = self.query_time(rays_pts_emb, scales_emb, rotations_emb, time_feature, time_emb)
        if self.args.static_mlp:
            mask = self.static_mlp(hidden)
        elif self.args.empty_voxel:
            mask = self.empty_voxel(rays_pts_emb[:,:3])
        else:
            mask = torch.ones_like(opacity_emb[:,0]).unsqueeze(-1)
        
        # Phase 2: Class-conditioned deformation routing
        if self.use_class_deformation and semantic_labels is not None:
            return self._forward_class_conditioned(
                hidden, mask, rays_pts_emb, scales_emb, rotations_emb, 
                opacity_emb, shs_emb, semantic_labels)
        
        # Original path (no class conditioning)
        if self.args.no_dx:
            pts = rays_pts_emb[:,:3]
        else:
            dx = self.pos_deform(hidden)
            pts = torch.zeros_like(rays_pts_emb[:,:3])
            pts = rays_pts_emb[:,:3]*mask + dx
        if self.args.no_ds :
            scales = scales_emb[:,:3]
        else:
            ds = self.scales_deform(hidden)
            scales = torch.zeros_like(scales_emb[:,:3])
            scales = scales_emb[:,:3]*mask + ds
            
        if self.args.no_dr :
            rotations = rotations_emb[:,:4]
        else:
            dr = self.rotations_deform(hidden)
            rotations = torch.zeros_like(rotations_emb[:,:4])
            if self.args.apply_rotation:
                rotations = batch_quaternion_multiply(rotations_emb, dr)
            else:
                rotations = rotations_emb[:,:4] + dr

        if self.args.no_do :
            opacity = opacity_emb[:,:1] 
        else:
            do = self.opacity_deform(hidden) 
          
            opacity = torch.zeros_like(opacity_emb[:,:1])
            opacity = opacity_emb[:,:1]*mask + do
        if self.args.no_dshs:
            shs = shs_emb
        else:
            dshs = self.shs_deform(hidden).reshape([shs_emb.shape[0],16,3])

            shs = torch.zeros_like(shs_emb)
            shs = shs_emb*mask.unsqueeze(-1) + dshs

        return pts, scales, rotations, opacity, shs
    
    def _forward_class_conditioned(self, hidden, mask, rays_pts_emb, scales_emb, 
                                    rotations_emb, opacity_emb, shs_emb, semantic_labels):
        """Route Gaussians through class-specific decoder heads.
        
        Class 0 (background): pass through unchanged
        Class 1 (tool): rigid decoder (position + rotation only)
        Class 2 (tissue): elastic decoder (full deformation)
        """
        N = hidden.shape[0]
        device = hidden.device
        
        # Initialize outputs as pass-through
        pts = rays_pts_emb[:,:3].clone()
        scales = scales_emb[:,:3].clone()
        rotations = rotations_emb[:,:4].clone()
        opacity = opacity_emb[:,:1].clone()
        shs = shs_emb.clone()
        
        # Tool mask (label == 1) -> rigid decoder
        tool_mask = (semantic_labels == 1)
        if tool_mask.any():
            h_tool = hidden[tool_mask]
            m_tool = mask[tool_mask] if mask.shape[0] == N else mask
            
            # Rigid position deformation
            if not self.args.no_dx:
                dx_rigid = self.rigid_pos_deform(h_tool)
                pts[tool_mask] = rays_pts_emb[tool_mask, :3] * m_tool + dx_rigid
            
            # Rigid rotation deformation
            if not self.args.no_dr:
                dr_rigid = self.rigid_rotations_deform(h_tool)
                if self.args.apply_rotation:
                    rotations[tool_mask] = batch_quaternion_multiply(
                        rotations_emb[tool_mask], dr_rigid)
                else:
                    rotations[tool_mask] = rotations_emb[tool_mask, :4] + dr_rigid
            # Tool: no scale, opacity, or SH deformation (rigid body)
        
        # Tissue/vessel mask (label == 2) -> elastic decoder
        tissue_mask = (semantic_labels == 2)
        if tissue_mask.any():
            h_tissue = hidden[tissue_mask]
            m_tissue = mask[tissue_mask] if mask.shape[0] == N else mask
            
            if not self.args.no_dx:
                dx_elastic = self.pos_deform(h_tissue)
                pts[tissue_mask] = rays_pts_emb[tissue_mask, :3] * m_tissue + dx_elastic
            
            if not self.args.no_ds:
                ds_elastic = self.scales_deform(h_tissue)
                scales[tissue_mask] = scales_emb[tissue_mask, :3] * m_tissue + ds_elastic
            
            if not self.args.no_dr:
                dr_elastic = self.rotations_deform(h_tissue)
                if self.args.apply_rotation:
                    rotations[tissue_mask] = batch_quaternion_multiply(
                        rotations_emb[tissue_mask], dr_elastic)
                else:
                    rotations[tissue_mask] = rotations_emb[tissue_mask, :4] + dr_elastic
            
            if not self.args.no_do:
                do_elastic = self.opacity_deform(h_tissue)
                opacity[tissue_mask] = opacity_emb[tissue_mask, :1] * m_tissue + do_elastic
            
            if not self.args.no_dshs:
                dshs_elastic = self.shs_deform(h_tissue).reshape([-1, 16, 3])
                shs[tissue_mask] = shs_emb[tissue_mask] * m_tissue.unsqueeze(-1) + dshs_elastic
        
        # Background (label == 0): already pass-through from initialization
        
        return pts, scales, rotations, opacity, shs
    
    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" not in name:
                parameter_list.append(param)
        return parameter_list
    def get_grid_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" in name:
                parameter_list.append(param)
        return parameter_list
    
class deform_network(nn.Module):
    def __init__(self, args) :
        super(deform_network, self).__init__()
        net_width = args.net_width
        timebase_pe = args.timebase_pe
        defor_depth= args.defor_depth
        posbase_pe= args.posebase_pe
        scale_rotation_pe = args.scale_rotation_pe
        opacity_pe = args.opacity_pe
        timenet_width = args.timenet_width
        timenet_output = args.timenet_output
        grid_pe = args.grid_pe
        times_ch = 2*timebase_pe+1
        self.timenet = nn.Sequential(
        nn.Linear(times_ch, timenet_width), nn.ReLU(),
        nn.Linear(timenet_width, timenet_output))
        self.deformation_net = Deformation(W=net_width, D=defor_depth, input_ch=(3)+(3*(posbase_pe))*2, grid_pe=grid_pe, input_ch_time=timenet_output, args=args)
        self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.register_buffer('rotation_scaling_poc', torch.FloatTensor([(2**i) for i in range(scale_rotation_pe)]))
        self.register_buffer('opacity_poc', torch.FloatTensor([(2**i) for i in range(opacity_pe)]))
        self.apply(initialize_weights)

    def forward(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None, semantic_labels=None):
        return self.forward_dynamic(point, scales, rotations, opacity, shs, times_sel, semantic_labels)
    @property
    def get_aabb(self):
        
        return self.deformation_net.get_aabb
    @property
    def get_empty_ratio(self):
        return self.deformation_net.get_empty_ratio
        
    def forward_static(self, points):
        points = self.deformation_net(points)
        return points
    
    def forward_dynamic(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None, semantic_labels=None):        
        point_emb = poc_fre(point, self.pos_poc)
        scales_emb = poc_fre(scales, self.rotation_scaling_poc)
        rotations_emb = poc_fre(rotations, self.rotation_scaling_poc)
        means3D, scales, rotations, opacity, shs = self.deformation_net(point_emb,
                                                scales_emb,
                                                rotations_emb,
                                                opacity,
                                                shs,
                                                None,
                                                times_sel,
                                                semantic_labels)
        return means3D, scales, rotations, opacity, shs
    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters() + list(self.timenet.parameters())
    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            init.zeros_(m.bias)
            
def poc_fre(input_data, poc_buf):

    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_emb = torch.cat([input_data, input_data_sin, input_data_cos], -1)
    return input_data_emb