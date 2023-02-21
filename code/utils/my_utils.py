import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import re
import sys
from typing import List, Union, Tuple
from collections import OrderedDict
try:
    import open3d as o3d
except (ModuleNotFoundError, OSError):
    pass


def image_net_center(img):
    stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    img = img.astype(np.float32)
    img /= 255.
    std = np.array([[stats['std'][::-1]]], dtype=np.float32)  # RGB to BGR
    mean = np.array([[stats['mean'][::-1]]], dtype=np.float32)
    return (img - mean) / (std + 0.00000001)


def image_net_center_inv(img):
    stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    std = np.array([[stats['std'][::-1]]], dtype=np.float32)  # RGB to BGR
    mean = np.array([[stats['mean'][::-1]]], dtype=np.float32)
    return ((img * std + mean)*255).astype(np.uint8)


def scale_camera(cam: Union[np.ndarray, torch.Tensor], scale: Union[Tuple, float]=1):
    """ resize input in order to produce sampled depth map """
    if type(scale) != tuple:
        scale = (scale, scale)
    if type(cam) == np.ndarray:
        new_cam = np.copy(cam)
        # focal:
        new_cam[1, 0, 0] = cam[1, 0, 0] * scale[0]
        new_cam[1, 1, 1] = cam[1, 1, 1] * scale[1]
        # principle point:
        new_cam[1, 0, 2] = cam[1, 0, 2] * scale[0]
        new_cam[1, 1, 2] = cam[1, 1, 2] * scale[1]
    elif type(cam) == torch.Tensor:
        new_cam = cam.clone()
        # focal:
        new_cam[..., 1, 0, 0] = cam[..., 1, 0, 0] * scale[0]
        new_cam[..., 1, 1, 1] = cam[..., 1, 1, 1] * scale[1]
        # principle point:
        new_cam[..., 1, 0, 2] = cam[..., 1, 0, 2] * scale[0]
        new_cam[..., 1, 1, 2] = cam[..., 1, 1, 2] * scale[1]
    # elif type(cam) == tf.Tensor:
    #     scale_tensor = np.ones((1, 2, 4, 4))
    #     scale_tensor[0, 1, 0, 0] = scale[0]
    #     scale_tensor[0, 1, 1, 1] = scale[1]
    #     scale_tensor[0, 1, 0, 2] = scale[0]
    #     scale_tensor[0, 1, 1, 2] = scale[1]
    #     new_cam = cam * scale_tensor
    else:
        raise TypeError
    return new_cam


def bin_op_reduce(lst, func):
    result = lst[0]
    for i in range(1, len(lst)):
        result = func(result, lst[i])
    return result


def get_pixel_grids(height, width, cuda=True):
    x_coord = (torch.arange(width, dtype=torch.float32) + 0.5).repeat(height, 1)
    y_coord = (torch.arange(height, dtype=torch.float32) + 0.5).repeat(width, 1).t()
    if cuda:
        x_coord = x_coord.cuda()
        y_coord = y_coord.cuda()
    ones = torch.ones_like(x_coord)
    indices_grid = torch.stack([x_coord, y_coord, ones], dim=-1).unsqueeze(-1)  # hw31
    return indices_grid


def idx_img2cam(idx_img_homo, depth, cam):  
    """nhw31, n1hw -> nhw41"""
    idx_cam = cam[:,1:2,:3,:3].unsqueeze(1).inverse() @ idx_img_homo  # nhw31
    idx_cam = idx_cam / (idx_cam[...,-1:,:]+1e-9) * depth.permute(0,2,3,1).unsqueeze(4)  # nhw31
    idx_cam_homo = torch.cat([idx_cam, torch.ones_like(idx_cam[...,-1:,:])], dim=-2)  # nhw41
    # FIXME: out-of-range is 0,0,0,1, will have valid coordinate in world
    return idx_cam_homo


def idx_cam2world(idx_cam_homo, cam):  
    """nhw41 -> nhw41"""
    idx_world_homo = cam[:,0:1,...].unsqueeze(1).inverse() @ idx_cam_homo  # nhw41
    idx_world_homo = idx_world_homo / (idx_world_homo[...,-1:,:]+1e-9)  # nhw41
    return idx_world_homo


def idx_world2cam(idx_world_homo, cam):  
    """nhw41 -> nhw41"""
    idx_cam_homo = cam[:,0:1,...].unsqueeze(1) @ idx_world_homo  # nhw41
    idx_cam_homo = idx_cam_homo / (idx_cam_homo[...,-1:,:]+1e-9)   # nhw41
    return idx_cam_homo


def idx_cam2img(idx_cam_homo, cam):  
    """nhw41 -> nhw31"""
    idx_cam = idx_cam_homo[...,:3,:] / (idx_cam_homo[...,3:4,:]+1e-9)  # nhw31
    idx_img_homo = cam[:,1:2,:3,:3].unsqueeze(1) @ idx_cam  # nhw31
    idx_img_homo = idx_img_homo / (idx_img_homo[...,-1:,:]+1e-9)
    return idx_img_homo


def project_img(src_img, dst_depth, src_cam, dst_cam, height=None, width=None):  
    """nchw, n1hw -> nchw, n1hw"""
    if height is None: height = src_img.size()[-2]
    if width is None: width = src_img.size()[-1]
    dst_idx_img_homo = get_pixel_grids(height, width, src_img.is_cuda).unsqueeze(0)  # nhw31
    dst_idx_cam_homo = idx_img2cam(dst_idx_img_homo, dst_depth, dst_cam)  # nhw41
    dst_idx_world_homo = idx_cam2world(dst_idx_cam_homo, dst_cam)  # nhw41
    dst2src_idx_cam_homo = idx_world2cam(dst_idx_world_homo, src_cam)  # nhw41
    dst2src_idx_img_homo = idx_cam2img(dst2src_idx_cam_homo, src_cam)  # nhw31
    warp_coord = dst2src_idx_img_homo[...,:2,0]  # nhw2
    warp_coord[..., 0] /= width
    warp_coord[..., 1] /= height
    warp_coord = (warp_coord*2-1).clamp(-1.1, 1.1)  # nhw2
    in_range = bin_op_reduce([-1<=warp_coord[...,0], warp_coord[...,0]<=1, -1<=warp_coord[...,1], warp_coord[...,1]<=1], torch.min).to(src_img.dtype).unsqueeze(1)  # n1hw
    warped_img = F.grid_sample(src_img, warp_coord, mode='bilinear', padding_mode='zeros', align_corners=False)
    return warped_img, in_range


def im2col(x, kernel_size, dim=2):  
    """nchw -> nc k**2 hw"""
    size = x.size()
    offsets = list(itertools.product([*range(kernel_size//2+1), *range(-(kernel_size//2), 0)], repeat=dim))
    # out = torch.cuda.FloatTensor(size[0], len(offsets), *size[2:]).zero_()
    out = torch.zeros((size[0], size[1], len(offsets), *size[2:]), dtype=x.dtype, device=x.device)
    for k, o in enumerate(offsets):
        out[[slice(size[0]), slice(size[1]), k] + [slice(max(0, i), min(size[2+d], size[2+d] + i)) for d, i in enumerate(o)]] = \
            x[[slice(size[0]), slice(size[1])] + [slice(max(0, -i), min(size[2+d], size[2+d] - i)) for d, i in enumerate(o)]]
    out = out.contiguous()
    return out


def normalize_for_grid_sample_nondiff(input_, grid):
    grid_ = grid.clone()
    for dim in range(grid_.size()[-1]):
        grid_[..., dim] /= input_.size()[-1-dim]
    grid_ = (grid_ * 2 - 1).clamp(-1.1, 1.1)
    return grid_


def normalize_for_grid_sample(input_, grid):
    size = torch.tensor(input_.size())[2:].flip(0).to(grid.dtype).to(grid.device).view(1,1,1,-1)  # 111N
    grid_n = grid / size
    grid_n = (grid_n * 2 - 1).clamp(-1.1, 1.1)
    return grid_n


def get_in_range(grid):  
    """after normalization, keepdim=False"""
    masks = []
    for dim in range(grid.size()[-1]):
        masks += [grid[..., dim]<=1, grid[..., dim]>=-1]
    in_range = bin_op_reduce(masks, torch.min).to(grid.dtype)
    return in_range


class RunningTopK:

    def __init__(self, k, max_keep, max=True, invalid_value=None):
        super(RunningTopK, self).__init__()
        if max_keep < k:
            raise ValueError('max_keep cannot be smaller than k.')
        self.k = k
        self.max_keep = max_keep
        self.max = max
        self.result = None
        self.update = []
        self.invalid_value = invalid_value
    
    def append(self, x):
        self.update.append(x.unsqueeze(-1))
        if (self.result is None and len(self.update) > self.max_keep) or (self.result is not None and len(self.update) > (self.max_keep - self.k)):
            self.calc()
    
    def calc(self):
        stack_list = [self.result] + self.update if self.result is not None else self.update
        self.result = torch.cat(stack_list, dim=-1).sort(dim=-1, descending=self.max)[0][...,:self.k]
        self.update = []
    
    def aggregate(self):
        if len(self.update) > 0:
            self.calc()
        if self.invalid_value is not None:
            valid_mask = self.result.abs() < self.invalid_value*.99
            valid_num = valid_mask.float().sum(-1)
            ret = (self.result * valid_mask).sum(-1) / (valid_num + 1e-9)
            ret = ret * (valid_num > 0.5) + self.invalid_value * (-1 if self.max else 1) * (valid_num < 0.5)
        else:
            ret = self.result.sum(dim=-1) / (self.k + 1e-9)
        return ret


def carving_t(points, depths, cams, out_thresh_perc, mask=False, image_in_cpu=False):
    """nm41, nv1hw, nv244 -> nm"""
    n, m, _, _ = points.size()
    _, v, _, h, w = depths.size() if not image_in_cpu else depths.shape
    MIN_FLOAT32 = 1e-30
    MAX_FLOAT32 = 1e+30 / v

    total_in_range = 0
    total_valid = 0
    total_inside = 0
    total_outside = 0
    total_dist_pos_agg = RunningTopK(1, 8, max=False, invalid_value=MAX_FLOAT32)  # NOTE: hardcode
    total_dist_neg_agg = RunningTopK(1, 8, max=True, invalid_value=MAX_FLOAT32)
    for view in range(v):
        cam = cams[:,view,...]  # n244
        depth = depths[:,view,...]  # n1hw
        if image_in_cpu: depth = torch.from_numpy(depth).float().cuda()

        if mask: depth *= MIN_FLOAT32

        cam_coord_grid_flat = idx_world2cam(points.unsqueeze(1), cam)  # n1m41
        point_depth = cam_coord_grid_flat[:,0,:,2,0]  # nm
        img_coord_grid_flat = idx_cam2img(cam_coord_grid_flat, cam)  # n1m31
        img_xy_grid_flat = img_coord_grid_flat[..., :2, 0]  # n1m2
        img_xy_grid_flat = normalize_for_grid_sample(depth, img_xy_grid_flat)  # n1m2
        in_range = get_in_range(img_xy_grid_flat)  # n1m
        feat_grid_flat = F.grid_sample(depth, img_xy_grid_flat, mode='nearest', padding_mode='zeros', align_corners=False)  # n11m
        gathered_depth = feat_grid_flat[:,0,0,:]  # nm

        in_range = in_range[:,0,:] > 0.5
        not_in_range = ~in_range
        valid = (gathered_depth > 0) & in_range
        invalid = in_range ^ valid
        inside = (point_depth > gathered_depth*0.99) & valid
        outside = valid ^ inside
        dist = (point_depth - gathered_depth) * valid

        total_in_range += in_range
        total_valid += valid
        total_inside += inside
        total_outside += outside
        total_dist_pos_agg.append( dist * inside + MAX_FLOAT32 * (~inside) )
        total_dist_neg_agg.append( dist * outside + (-MAX_FLOAT32) * (~outside) )
    
    total_dist_pos = total_dist_pos_agg.aggregate()
    total_dist_neg = total_dist_neg_agg.aggregate()

    total_outside_valid = total_valid - total_inside
    total_invalid = total_in_range - total_valid
    total_outside = total_in_range - total_inside
    # in_thresh_perc = in_thresh / v
    # inside_perc = total_inside / (total_in_range + 1e-9)
    outside_perc = (total_outside_valid + total_invalid * 0.5) / (total_in_range + 1e-9)  # NOTE: hard code
    # scene_in_range = torch.max(torch.zeros_like(total_in_range), total_in_range-1) / (total_in_range + 1e-9) > in_thresh_perc
    scene_in_range = total_in_range > 0
    scene_valid = total_valid > 0
    # scene_inside = (inside_perc >= in_thresh_perc) & scene_valid
    # scene_outside = scene_in_range ^ scene_inside
    scene_outside = (outside_perc > out_thresh_perc) & scene_in_range
    scene_inside = scene_in_range ^ scene_outside
    scene_outside_valid = scene_valid ^ scene_inside  # xor
    ave_dist = total_dist_pos * scene_inside + total_dist_neg * scene_outside# + (-MAX_FLOAT32) * (scene_in_range ^ scene_valid)
    return ave_dist, scene_inside, scene_in_range


def carving_t2(points, depths, cams, out_thresh_perc, mask=False, image_in_cpu=False):
    """nm41, nv1hw, nv244 -> nm"""
    n, m, _, _ = points.size()
    _, v, _, h, w = depths.size() if not image_in_cpu else depths.shape
    MIN_FLOAT32 = 1e-30
    MAX_FLOAT32 = 1e+30 / v

    total_in_range = 0
    total_valid = 0
    total_inside = 0
    total_outside = 0
    total_dist_pos_agg = RunningTopK(1, 8, max=False, invalid_value=MAX_FLOAT32)  # NOTE: hardcode
    total_dist_neg_agg = RunningTopK(1, 8, max=True, invalid_value=MAX_FLOAT32)
    for view in range(v):
        cam = cams[:,view,...]  # n244
        depth = depths[:,view,...]  # n1hw
        if image_in_cpu: depth = torch.from_numpy(depth).float().cuda()

        if mask: depth *= MIN_FLOAT32

        cam_coord_grid_flat = idx_world2cam(points.unsqueeze(1), cam)  # n1m41
        point_depth = cam_coord_grid_flat[:,0,:,2,0]  # nm
        # along_view_scale = 1 / (F.normalize(cam_coord_grid_flat[:,0,:,:3,0], dim=2)[...,2] + 1e-9)  # nm
        img_coord_grid_flat = idx_cam2img(cam_coord_grid_flat, cam)  # n1m31
        img_xy_grid_flat = img_coord_grid_flat[..., :2, 0]  # n1m2
        img_xy_grid_flat = normalize_for_grid_sample(depth, img_xy_grid_flat)  # n1m2
        in_range = get_in_range(img_xy_grid_flat)  # n1m
        feat_grid_flat = F.grid_sample(depth, img_xy_grid_flat, mode='nearest', padding_mode='zeros', align_corners=False)  # n11m
        gathered_depth = feat_grid_flat[:,0,0,:]  # nm

        in_range = in_range[:,0,:] > 0.5
        valid = (gathered_depth > 0) & in_range
        invalid = in_range ^ valid
        inside = (point_depth > gathered_depth*0.99) & valid
        outside = valid ^ inside
        dist = (point_depth - gathered_depth) * valid

        total_in_range += in_range
        total_valid += valid
        total_inside += inside
        total_outside += outside
        total_dist_pos_agg.append( dist * inside + MAX_FLOAT32 * (~inside) )
        total_dist_neg_agg.append( dist * outside + (-MAX_FLOAT32) * (~outside) )
    
    total_dist_pos = total_dist_pos_agg.aggregate()
    total_dist_neg = total_dist_neg_agg.aggregate()

    total_outside_valid = total_valid - total_inside
    total_invalid = total_in_range - total_valid
    total_outside = total_in_range - total_inside
    # in_thresh_perc = in_thresh / v
    # inside_perc = total_inside / (total_in_range + 1e-9)
    outside_perc = (total_outside_valid) / (total_valid + 1e-9)  # NOTE: hard code
    # scene_in_range = torch.max(torch.zeros_like(total_in_range), total_in_range-1) / (total_in_range + 1e-9) > in_thresh_perc
    scene_in_range = total_in_range > 0
    scene_valid = total_valid > 0
    # scene_inside = (inside_perc >= in_thresh_perc) & scene_valid
    # scene_outside = scene_in_range ^ scene_inside
    scene_outside = (outside_perc > out_thresh_perc) & scene_valid
    scene_inside = scene_valid ^ scene_outside
    scene_outside_valid = scene_valid ^ scene_inside  # xor
    ave_dist = total_dist_pos * scene_inside + total_dist_neg * scene_outside# + (-MAX_FLOAT32) * (scene_in_range ^ scene_valid)
    return ave_dist, scene_inside, scene_valid


def load_pair(file: str, min_views: int=None):
    with open(file) as f:
        lines = f.readlines()
    n_cam = int(lines[0])
    pairs = {}
    img_ids = []
    for i in range(1, 1+2*n_cam, 2):
        pair = []
        score = []
        img_id = lines[i].strip()
        pair_str = lines[i+1].strip().split(' ')
        n_pair = int(pair_str[0])
        if min_views is not None and n_pair < min_views: continue
        for j in range(1, 1+2*n_pair, 2):
            pair.append(pair_str[j])
            score.append(float(pair_str[j+1]))
        img_ids.append(img_id)
        pairs[img_id] = {'id': img_id, 'index': i//2, 'pair': pair, 'score': score}
    pairs['id_list'] = img_ids
    return pairs


def write_pair(file: str, pair):
    content = f"{len(pair['id_list'])}\n"
    for idx in pair['id_list']:
        content += f'{idx}\n'
        content += f"{' '.join([f'{src_idx} {src_score}' for src_idx, src_score in zip(pair[idx]['pair'], pair[idx]['score'])])}\n"
    with open(file, 'w') as f:
        f.write(content)


def load_cam(file: str, max_d, interval_scale=1, override=False):
    """ read camera txt file """
    cam = np.zeros((2, 4, 4))
    with open(file) as f:
        words = f.read().split()
    # read extrinsic
    for i in range(0, 4):
        for j in range(0, 4):
            extrinsic_index = 4 * i + j + 1
            cam[0][i][j] = words[extrinsic_index]

    # read intrinsic
    for i in range(0, 3):
        for j in range(0, 3):
            intrinsic_index = 3 * i + j + 18
            cam[1][i][j] = words[intrinsic_index]

    if len(words) == 29:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = max_d
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * (cam[1][3][2] - 1)
    elif len(words) == 30:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = words[29]
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * (cam[1][3][2] - 1)
    elif len(words) == 31:
        if override:
            cam[1][3][0] = words[27]
            cam[1][3][1] = (float(words[30]) - float(words[27])) / (max_d - 1)
            cam[1][3][2] = max_d
            cam[1][3][3] = words[30]
        else:
            cam[1][3][0] = words[27]
            cam[1][3][1] = float(words[28]) * interval_scale
            cam[1][3][2] = words[29]
            cam[1][3][3] = words[30]
    else:
        cam[1][3][0] = 0
        cam[1][3][1] = 0
        cam[1][3][2] = 0
        cam[1][3][3] = 0

    return cam


def cam_adjust_max_d(cam, max_d):
    cam = cam.copy()
    interval_scale = cam[1][3][2] / max_d
    cam[1][3][1] *= interval_scale
    cam[1][3][2] = max_d
    return cam


def write_cam(file: str, cam):
    content = f"""extrinsic
{cam[0][0][0]} {cam[0][0][1]} {cam[0][0][2]} {cam[0][0][3]}
{cam[0][1][0]} {cam[0][1][1]} {cam[0][1][2]} {cam[0][1][3]}
{cam[0][2][0]} {cam[0][2][1]} {cam[0][2][2]} {cam[0][2][3]}
{cam[0][3][0]} {cam[0][3][1]} {cam[0][3][2]} {cam[0][3][3]}

intrinsic
{cam[1][0][0]} {cam[1][0][1]} {cam[1][0][2]}
{cam[1][1][0]} {cam[1][1][1]} {cam[1][1][2]}
{cam[1][2][0]} {cam[1][2][1]} {cam[1][2][2]}

{cam[1][3][0]} {cam[1][3][1]} {cam[1][3][2]} {cam[1][3][3]}
"""
    with open(file, 'w') as f:
        f.write(content)


def load_pfm(file: str):
    color = None
    width = None
    height = None
    scale = None
    endian = None
    with open(file, 'rb') as f:
        header = f.readline().rstrip()
        if header == b'PF':
            color = True
        elif header == b'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')
        dim_match = re.match(br'^(\d+)\s(\d+)\s$', f.readline())
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')
        scale = float(f.readline().rstrip())
        if scale < 0:  # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>'  # big-endian
        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = data[::-1, ...]  # cv2.flip(data, 0)
    return data


def write_pfm(file: str, image, scale=1):
    with open(file, 'wb') as f:
        color = None

        if image.dtype.name != 'float32':
            raise Exception('Image dtype must be float32.')

        image = np.flipud(image)

        if len(image.shape) == 3 and image.shape[2] == 3: # color image
            color = True
        elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
            color = False
        else:
            raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

        f.write(b'PF\n' if color else b'Pf\n')
        f.write(b'%d %d\n' % (image.shape[1], image.shape[0]))

        endian = image.dtype.byteorder

        if endian == '<' or endian == '=' and sys.byteorder == 'little':
            scale = -scale

        f.write(b'%f\n' % scale)

        image.tofile(f)


class ListModule(nn.Module):
    def __init__(self, modules: Union[List, OrderedDict]):
        super(ListModule, self).__init__()
        if isinstance(modules, OrderedDict):
            iterable = modules.items()
        elif isinstance(modules, list):
            iterable = enumerate(modules)
        else:
            raise TypeError('modules should be OrderedDict or List.')
        for name, module in iterable:
            if not isinstance(module, nn.Module):
                module = ListModule(module)
            if not isinstance(name, str):
                name = str(name)
            self.add_module(name, module)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dim=2):
        super(BasicBlock, self).__init__()

        self.conv_fn = nn.Conv2d if dim == 2 else nn.Conv3d
        self.bn_fn = nn.BatchNorm2d if dim == 2 else nn.BatchNorm3d
        # self.bn_fn = nn.GroupNorm

        self.conv1 = self.conv3x3(inplanes, planes, stride)
        # nn.init.xavier_uniform_(self.conv1.weight)
        self.bn1 = self.bn_fn(planes)
        # nn.init.constant_(self.bn1.weight, 1)
        # nn.init.constant_(self.bn1.bias, 0)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = self.conv3x3(planes, planes)
        # nn.init.xavier_uniform_(self.conv2.weight)
        self.bn2 = self.bn_fn(planes)
        # nn.init.constant_(self.bn2.weight, 0)
        # nn.init.constant_(self.bn2.bias, 0)
        self.downsample = downsample
        self.stride = stride

    def conv1x1(self, in_planes, out_planes, stride=1):
        """1x1 convolution"""
        return self.conv_fn(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    def conv3x3(self, in_planes, out_planes, stride=1):
        """3x3 convolution with padding"""
        return self.conv_fn(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def _make_layer(inplanes, block, planes, blocks, stride=1, dim=2):
    downsample = None
    conv_fn = nn.Conv2d if dim==2 else nn.Conv3d
    bn_fn = nn.BatchNorm2d if dim==2 else nn.BatchNorm3d
    # bn_fn = nn.GroupNorm
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            conv_fn(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
            bn_fn(planes * block.expansion)
        )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample, dim=dim))
    inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(block(inplanes, planes, dim=dim))

    return nn.Sequential(*layers)


class UNet(nn.Module):

    def __init__(self, inplanes: int, enc: int, dec: int, initial_scale: int,
                 bottom_filters: List[int], filters: List[int], head_filters: List[int],
                 prefix: str, dim: int=2):
        super(UNet, self).__init__()

        conv_fn = nn.Conv2d if dim==2 else nn.Conv3d
        bn_fn = nn.BatchNorm2d if dim==2 else nn.BatchNorm3d
        # bn_fn = nn.GroupNorm
        deconv_fn = nn.ConvTranspose2d if dim==2 else nn.ConvTranspose3d
        current_scale = initial_scale
        idx = 0
        prev_f = inplanes

        self.bottom_blocks = OrderedDict()
        for f in bottom_filters:
            block = _make_layer(prev_f, BasicBlock, f, enc, 1 if idx==0 else 2, dim=dim)
            self.bottom_blocks[f'{prefix}{current_scale}_{idx}'] = block
            idx += 1
            current_scale *= 2
            prev_f = f
        self.bottom_blocks = ListModule(self.bottom_blocks)

        self.enc_blocks = OrderedDict()
        for f in filters:
            block = _make_layer(prev_f, BasicBlock, f, enc, 1 if idx == 0 else 2, dim=dim)
            self.enc_blocks[f'{prefix}{current_scale}_{idx}'] = block
            idx += 1
            current_scale *= 2
            prev_f = f
        self.enc_blocks = ListModule(self.enc_blocks)

        self.dec_blocks = OrderedDict()
        for f in filters[-2::-1]:
            block = [
                deconv_fn(prev_f, f, 3, 2, 1, 1, bias=False),
                conv_fn(2*f, f, 3, 1, 1, bias=False),
            ]
            if dec > 0:
                block.append(_make_layer(f, BasicBlock, f, dec, 1, dim=dim))
            # nn.init.xavier_uniform_(block[0].weight)
            # nn.init.xavier_uniform_(block[1].weight)
            self.dec_blocks[f'{prefix}{current_scale}_{idx}'] = block
            idx += 1
            current_scale //= 2
            prev_f = f
        self.dec_blocks = ListModule(self.dec_blocks)

        self.head_blocks = OrderedDict()
        for f in head_filters:
            block = [
                deconv_fn(prev_f, f, 3, 2, 1, 1, bias=False)
            ]
            if dec > 0:
                block.append(_make_layer(f, BasicBlock, f, dec, 1, dim=dim))
            block = nn.Sequential(*block)
            # nn.init.xavier_uniform_(block[0])
            self.head_blocks[f'{prefix}{current_scale}_{idx}'] = block
            idx += 1
            current_scale //= 2
            prev_f = f
        self.head_blocks = ListModule(self.head_blocks)

    def forward(self, x, multi_scale=1):
        for b in self.bottom_blocks:
            x = b(x)
        enc_out = []
        for b in self.enc_blocks:
            x = b(x)
            enc_out.append(x)
        dec_out = [x]
        for i, b in enumerate(self.dec_blocks):
            if len(b) == 3: deconv, post_concat, res = b
            elif len(b) == 2: deconv, post_concat = b
            x = deconv(x)
            x = torch.cat([x, enc_out[-2-i]], 1)
            x = post_concat(x)
            if len(b) == 3: x = res(x)
            dec_out.append(x)
        for b in self.head_blocks:
            x = b(x)
            dec_out.append(x)
        if multi_scale == 1: return x
        else: return dec_out[-multi_scale:]


class FeatExt(nn.Module):

    def __init__(self):
        super(FeatExt, self).__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2, 2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.unet = UNet(16, 2, 1, 2, [], [32, 64, 128], [], '2d', 2)
        self.final_conv_1 = nn.Conv2d(128, 32, 3, 1, 1, bias=False)
        self.final_conv_2 = nn.Conv2d(64, 32, 3, 1, 1, bias=False)
        self.final_conv_3 = nn.Conv2d(32, 32, 3, 1, 1, bias=False)

        feat_ext_dict = {k[16:]:v for k,v in torch.load('utils/vismvsnet.pt')['state_dict'].items() if k.startswith('module.feat_ext')}
        self.load_state_dict(feat_ext_dict)

    def forward(self, x):
        out = self.init_conv(x)
        out1, out2, out3 = self.unet(out, multi_scale=3)
        return self.final_conv_1(out1), self.final_conv_2(out2), self.final_conv_3(out3)


class GridSampleDiff(torch.autograd.Function):  # not used
    
    @staticmethod
    def forward(ctx, input_, grid):
        res = F.grid_sample(input_, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        ctx.save_for_backward(input_, grid, res)
        return res
    
    @staticmethod
    def backward(ctx, grad):
        input_, grid, res = ctx.saved_tensors
        grad_input, grad_grid = torch.autograd.grad(
            outputs = [res],
            inputs = [input_, grid],
            grad_outputs = [grad]
        )


def plot_pcd(pts):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.cpu().numpy())
    o3d.visualization.draw_geometries([pcd])
