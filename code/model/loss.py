from numpy.lib.function_base import diff
import torch
from torch import nn
from torch.nn import functional as F
from itertools import accumulate
import numpy as np
import os
import importlib

from utils.my_utils import carving_t, carving_t2, FeatExt, get_in_range, idx_cam2img, idx_world2cam, normalize_for_grid_sample
import model.conf as conf
if os.environ.get('IDR_USE_ENV', '0') == '1' and os.environ.get('IDR_CONF', '') != '':
    print('override conf: ', os.environ.get('IDR_CONF'))
    conf = importlib.import_module(os.environ.get('IDR_CONF'))

class IDRLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss(reduction='sum')

    def get_rgb_loss(self,rgb_values, rgb_gt, network_object_mask, object_mask):
        if (network_object_mask & object_mask).sum() == 0:
            return torch.tensor(0.0).cuda().float()

        rgb_values = rgb_values[network_object_mask & object_mask]
        rgb_gt = rgb_gt.reshape(-1, 3)[network_object_mask & object_mask]
        rgb_loss = self.l1_loss(rgb_values, rgb_gt) / float(object_mask.shape[0])
        return rgb_loss

    def get_eikonal_loss(self, grad_theta):
        if grad_theta.shape[0] == 0:
            return torch.tensor(0.0).cuda().float()

        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss
    
    def get_depth_loss(self, eikonal_points_hom, eikonal_output, depths, cams, size, center, far_thresh, far_att, near_thresh, near_att, smooth):
        eikonal_points_hom = eikonal_points_hom.detach()
        depths = depths.permute(1,0,2,3,4)
        cams = cams.permute(1,0,2,3,4)

        eikonal_points_hom[:,:,:3,0] = eikonal_points_hom[:,:,:3,0] / 2 * size.view(1,1,1) + center.view(1,1,3)
        if conf.use_invalid:  # treat out-of-mask depth as inf
            dist, occ, in_range = carving_t(eikonal_points_hom, depths, cams, out_thresh_perc=conf.out_thresh_perc)
        else:  # ignore out-of-mask depth
            dist, occ, in_range = carving_t2(eikonal_points_hom, depths, cams, out_thresh_perc=conf.out_thresh_perc)  # scale is applied in cams NOTE: hard code
        dist_r = (dist / size.view(1,1) * 2 + (-1.25) * (~in_range).to(torch.float32)).clamp(-1.25,1.25)
        # loss = nn.SmoothL1Loss()(eikonal_output, -dist_r)

        # single depth
        # not_inside = (dist_r < int_thresh)
        # inside_weight = not_inside + (~not_inside) * int_att
        far_mask = dist_r.abs() > far_thresh
        far_weight = far_mask * far_att + (~far_mask)
        near_mask = dist_r.abs() < near_thresh
        near_weight = near_mask * near_att + (~near_mask)
        if smooth is not None:
            loss = nn.SmoothL1Loss(reduction='none')(eikonal_output / smooth, -dist_r / smooth) * smooth
        else:
            loss = nn.L1Loss(reduction='none')(eikonal_output, -dist_r)
        loss = (loss * far_weight * near_weight * in_range).mean()

        return loss
    
    def get_feat_loss2(self, diff_surf_pts, uncerts, feat, cam, feat_src, src_cams, size, center, network_object_mask, object_mask):
        mask = network_object_mask & object_mask
        if (mask).sum() == 0:
            return torch.tensor(0.0).float().cuda()
        
        sample_mask = mask.view(feat.size()[0], -1)
        hit_nums = sample_mask.sum(-1)
        accu_nums = [0] + hit_nums.cumsum(0).tolist()
        slices = [slice(accu_nums[i], accu_nums[i+1]) for i in range(len(accu_nums)-1)]

        loss = []
        ## for each image in minibatch
        for view_i, slice_ in enumerate(slices):
            if slice_.start < slice_.stop:

                ## projection
                diff_surf_pts_slice = diff_surf_pts[slice_]
                pts_world = (diff_surf_pts_slice / 2 * size.view(1,1) + center.view(1,3)).view(1,-1,1,3,1)  # 1m131
                pts_world = torch.cat([pts_world, torch.ones_like(pts_world[...,-1:,:])], dim=-2)  # 1m141
                # rgb_pack = torch.cat([rgb[view_i:view_i+1], rgb_src[view_i]], dim=0)  # v3hw
                cam_pack = torch.cat([cam[view_i:view_i+1], src_cams[view_i]], dim=0)  # v244
                pts_img = idx_cam2img(idx_world2cam(pts_world, cam_pack), cam_pack)  # vm131

                ## gathering
                grid = pts_img[...,:2,0]  # vm12
                # feat2_pack = self.feat_ext(rgb_pack)[2]  # vchw # TODO: multi-scale feature
                feat2_pack = torch.cat([feat[view_i:view_i+1], feat_src[view_i]], dim=0)
                grid_n = normalize_for_grid_sample(feat2_pack, grid/2)
                grid_in_range = get_in_range(grid_n)
                valid_mask = (grid_in_range[:1,...] * grid_in_range[1:,...]).unsqueeze(1) > 0.5  # and
                gathered_feat = F.grid_sample(feat2_pack, grid_n, mode='bilinear', padding_mode='zeros', align_corners=False)  # vcm1

                ## calculation
                diff = gathered_feat[:1] - gathered_feat[1:]
                if uncerts is None:
                    gathered_norm = gathered_feat.norm(dim=1, keepdim=True)  # vcm1
                    diff_mask = diff.norm(dim=1, keepdim=True) < ((gathered_norm[:1,...] + gathered_norm[1:,...])/2*1)
                    print('feat loss mask', (valid_mask & diff_mask).sum().item(), '/', valid_mask.size()[0] * valid_mask.size()[2])
                    sample_loss = (diff * valid_mask * diff_mask).abs().mean()
                else:
                    uncert = uncerts[view_i].unsqueeze(1).unsqueeze(3)  # (v-1)1m1
                    print(f'uncert: {uncert.min():.4f}, {uncert.median():.4f}, {uncert.max():.4f}')
                    sample_loss = ((diff.abs() * (-uncert).exp() + 0.01 * uncert)*valid_mask).mean()
            else:
                sample_loss = torch.zeros(1).float().cuda()
            loss.append(sample_loss)
        loss = sum(loss) / len(loss)

        return loss
    
    def get_feat_loss_corr(self, diff_surf_pts, uncerts, feat, cam, feat_src, src_cams, size, center, network_object_mask, object_mask):
        mask = network_object_mask & object_mask
        if (mask).sum() == 0:
            return torch.tensor(0.0).float().cuda()
        
        sample_mask = mask.view(feat.size()[0], -1)
        hit_nums = sample_mask.sum(-1)
        accu_nums = [0] + hit_nums.cumsum(0).tolist()
        slices = [slice(accu_nums[i], accu_nums[i+1]) for i in range(len(accu_nums)-1)]

        loss = []
        ## for each image in minibatch
        for view_i, slice_ in enumerate(slices):
            if slice_.start < slice_.stop:
                
                ## projection
                diff_surf_pts_slice = diff_surf_pts[slice_]
                pts_world = (diff_surf_pts_slice / 2 * size.view(1,1) + center.view(1,3)).view(1,-1,1,3,1)  # 1m131
                pts_world = torch.cat([pts_world, torch.ones_like(pts_world[...,-1:,:])], dim=-2)  # 1m141
                # rgb_pack = torch.cat([rgb[view_i:view_i+1], rgb_src[view_i]], dim=0)  # v3hw
                cam_pack = torch.cat([cam[view_i:view_i+1], src_cams[view_i]], dim=0)  # v244
                pts_img = idx_cam2img(idx_world2cam(pts_world, cam_pack), cam_pack)  # vm131
                
                ## gathering
                grid = pts_img[...,:2,0]  # vm12
                # feat2_pack = self.feat_ext(rgb_pack)[2]  # vchw # TODO: multi-scale feature
                feat2_pack = torch.cat([feat[view_i:view_i+1], feat_src[view_i]], dim=0)
                grid_n = normalize_for_grid_sample(feat2_pack, grid/2)
                grid_in_range = get_in_range(grid_n)
                valid_mask = (grid_in_range[:1,...] * grid_in_range[1:,...]).unsqueeze(1) > 0.5  # and
                gathered_feat = F.grid_sample(feat2_pack, grid_n, mode='bilinear', padding_mode='zeros', align_corners=False)  # vcm1
                
                ## calculation
                gathered_norm = gathered_feat.norm(dim=1, keepdim=True)  # v1m1
                corr = (gathered_feat[:1] * gathered_feat[1:]).sum(dim=1, keepdim=True) \
                        / gathered_norm[:1].clamp(min=1e-9) / gathered_norm[1:].clamp(min=1e-9)  # (v-1)1m1
                corr_loss = (1 - corr).abs()
                if uncerts is None:
                    diff_mask = corr_loss < 0.5
                    print('feat loss mask', (valid_mask & diff_mask).sum().item(), '/', valid_mask.size()[0] * valid_mask.size()[2])
                    sample_loss = (corr_loss * valid_mask * diff_mask).mean()
                else:
                    uncert = uncerts[view_i].unsqueeze(1).unsqueeze(3)  # (v-1)1m1
                    print(f'uncert: {uncert.min():.4f}, {uncert.median():.4f}, {uncert.max():.4f}')
                    sample_loss = ((corr_loss * (-uncert).exp() + uncert)*valid_mask).mean()
            else:
                sample_loss = torch.zeros(1).float().cuda()
            loss.append(sample_loss)
        loss = sum(loss) / len(loss)

        return loss
    
    def get_surf_loss(self, surf_indicator_output, network_object_mask, object_mask_true):
        mask = network_object_mask & object_mask_true
        N = mask.sum()
        gt1 = torch.ones(N, dtype=surf_indicator_output.dtype, device=surf_indicator_output.device)
        gt0 = torch.zeros(surf_indicator_output.size()[0]-N, dtype=surf_indicator_output.dtype, device=surf_indicator_output.device)
        gt = torch.cat([gt1, gt0], dim=0)
        loss = nn.BCEWithLogitsLoss(reduction='mean')(surf_indicator_output, gt)
        return loss

    def forward(self, model_outputs, ground_truth, train_progress, n_img):
        rgb_gt = ground_truth['rgb'].cuda()
        network_object_mask = model_outputs['network_object_mask']
        object_mask = model_outputs['object_mask']

        ground_truth['size'] = ground_truth['size'][:1]
        ground_truth['center'] = ground_truth['center'][:1]

        if conf.enable_rgb:
            rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'], rgb_gt, network_object_mask, object_mask)
        else:
            rgb_loss = torch.zeros(1).float().cuda()
        
        eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])

        depth_loss = self.get_depth_loss(model_outputs['eikonal_points_hom'], model_outputs['eikonal_output'], ground_truth['depths'], ground_truth['depth_cams'], ground_truth['size'], ground_truth['center'], 
            far_thresh=conf.far_thresh, far_att=conf.far_att(train_progress), 
            near_thresh=conf.near_thresh, near_att=conf.near_att(train_progress), 
            smooth=conf.smooth(train_progress))
        
        if conf.phase[0] <= train_progress and conf.enable_feat:
            feat_loss = self.get_feat_loss_corr(model_outputs['diff_surf_pts'], model_outputs.get('uncerts'), *[ground_truth[attr] for attr in ['feat', 'cam', 'feat_src', 'src_cams', 'size', 'center']], network_object_mask, object_mask)
        else:
            feat_loss = torch.zeros(1).float().cuda()
        
        if conf.phase[0] <= train_progress:
            surf_loss = self.get_surf_loss(model_outputs['surf_indicator_output'], network_object_mask, model_outputs['object_mask_true'])
        else:
            surf_loss = torch.zeros(1).float().cuda()

        loss = rgb_loss * conf.rgb_weight(train_progress) + \
               eikonal_loss * conf.eikonal_weight + \
               surf_loss * conf.surf_weight + \
               feat_loss * conf.feat_weight(train_progress) + \
               depth_loss * conf.depth_weight(train_progress)

        return {
            'loss': loss,
            'rgb_loss': rgb_loss,
            'eikonal_loss': eikonal_loss,
            'depth_loss': depth_loss,
            'feat_loss': feat_loss,
            'surf_loss': surf_loss
        }
