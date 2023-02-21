import argparse
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from utils.my_utils import load_pfm, write_pfm, load_cam, load_pair, scale_camera
import open3d as o3d
import tqdm


def resize_single(img, width, height):
    h_o, w_o = img.shape[:2]
    ret = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR) if w_o != width or h_o != height else img
    return ret

def center_crop_single(img, width, height):
    h_o, w_o = img.shape[:2]
    start_w = (w_o - width)//2
    start_h = (h_o - height)//2
    finish_w = start_w + width
    finish_h = start_h + height
    ret = img[start_h:finish_h, start_w:finish_w] if w_o != width or h_o != height else img
    return ret


parser = argparse.ArgumentParser()

parser.add_argument('--data_root', type=str, default='eg/path/to/vismvsnet/output')
parser.add_argument('--range_source', type=str, choices=['range', 'pcd'], default='pcd')
parser.add_argument('--pthresh', type=str, default='.7,.7,0')
parser.add_argument('--prob_mask', action='store_true', default=False)
parser.add_argument('--resize', type=str, default='1920,1080')
parser.add_argument('--crop', type=str, default='1920,1072')
parser.add_argument('--ext_image_path', type=str, default='eg/path/to/image/{:08}.jpg')
parser.add_argument('--ext_image_from_one', action='store_true', default=False)
parser.add_argument('--show_range', action='store_true', default=False)

args = parser.parse_args()

pair = load_pair(f'{args.data_root}/pair.txt')
total_views = len(pair['id_list'])
resize_w, resize_h = [int(v) for v in args.resize.split(',')]
crop_w, crop_h = [int(v) for v in args.crop.split(',')]

cams = torch.stack([torch.from_numpy(load_cam(f'{args.data_root}/cam_{pair["id_list"][i].zfill(8)}_flow3.txt', 256, 1, override=True)).float() for i in range(total_views)], dim=0)
depths = torch.stack([torch.from_numpy(np.ascontiguousarray(load_pfm(f'{args.data_root}/{pair["id_list"][i].zfill(8)}_flow3.pfm'))).float() for i in range(total_views)], dim=0).unsqueeze(1)
probs = torch.stack([torch.stack([torch.from_numpy(np.ascontiguousarray(load_pfm(f'{args.data_root}/{pair["id_list"][i].zfill(8)}_flow{j+1}_prob.pfm'))).float() for j in range(3)], dim=0) for i in range(total_views)], dim=0).unsqueeze(2)

d_w, d_h = depths.size()[-1], depths.size()[-2]

pthresh = [float(v) for v in args.pthresh.split(',')]
if args.prob_mask:
    masks = ((probs > torch.from_numpy(np.array(pthresh)).float().view(1,3,1,1,1)).sum(1) > 2.9).float()
else:
    masks = torch.stack([torch.from_numpy(cv2.imread(f'{args.data_root}/{pair["id_list"][i].zfill(8)}_mask.png', cv2.IMREAD_GRAYSCALE)).float()/255 for i in range(total_views)], dim=0).unsqueeze(1)

depths *= masks

if args.range_source == 'range':
    corners_world_coord = []
    image_height = depths[0].size()[-2]
    image_width = depths[0].size()[-1]
    for i in range(len(depths)):
        frustum_corners = torch.from_numpy(np.array([
            [0, 0, 1],
            [image_height, 0, 1],
            [0, image_width, 1],
            [image_height, image_width, 1]
        ])).to(torch.float32)
        corner_cam_coord = cams[i][1:2,:3,:3].inverse() @ frustum_corners.unsqueeze(-1)  # 431
        near_cam_coord = corner_cam_coord * cams[i][1,3,0]
        far_cam_coord = corner_cam_coord * cams[i][1,3,3]
        corner_cam_coord = torch.cat([near_cam_coord, far_cam_coord], dim=0)  # 831
        corner_cam_coord_homo = torch.cat([corner_cam_coord, torch.ones_like(corner_cam_coord[:,-1:,:])], 1)  # 841
        corner_world_coord = (cams[i][0:1,:,:].inverse() @ corner_cam_coord_homo)[:,:3,0]  # 83
        corners_world_coord.append(corner_world_coord)
    corners_world_coord = torch.cat(corners_world_coord, 0)  # 8*v 3
    vert_min, _ = torch.min(corners_world_coord, dim=0)
    vert_max, _ = torch.max(corners_world_coord, dim=0)
    center = (vert_min + vert_max) / 2
    size = torch.max(vert_max - vert_min)
elif args.range_source == 'pcd':
    pcd = o3d.io.read_point_cloud(f'{args.data_root}/cut.ply')
    vert = torch.from_numpy(np.asarray(pcd.points)).float()
    vert_min, _ = torch.min(vert, dim=0)
    vert_max, _ = torch.max(vert, dim=0)
    center = (vert_min + vert_max) / 2
    size = torch.max(vert_max - vert_min) * 1.1
else:
    raise ValueError

save_dict = {}
masks_hd = (F.interpolate(masks, size=(crop_h, crop_w), mode='bilinear', align_corners=False) > 0.5).float()
for i in tqdm.tqdm(range(total_views), dynamic_ncols=True):
    img = cv2.imread(args.ext_image_path.format(int(pair["id_list"][i])+1 if args.ext_image_from_one else int(pair["id_list"][i])))
    img = resize_single(img, resize_w, resize_h)
    img = center_crop_single(img, crop_w, crop_h)
    cv2.imwrite(f'{args.data_root}/imfunc4/image_hd/{i:06}.png', img)

    cv2.imwrite(f'{args.data_root}/imfunc4/mask_hd/{i:03}.png', (masks_hd[i,0].numpy().astype(np.uint8)*255))
    
    write_pfm(f'{args.data_root}/imfunc4/depth/{i:03}.pfm', depths[i,0].numpy())

    cam = cams[i]
    cam = scale_camera(cam, (crop_w/d_w, crop_h/d_h))
    intm = cam[1]
    intm[3, :] = 0
    intm[:, 3] = 0
    intm[3,3] = 1
    extm = cam[0]
    wm = intm @ extm
    sm = np.eye(4).astype(np.float32)
    sm[:3,:3] *= size.item() / 2
    sm[:3, 3] = center.numpy()
    save_dict[f'world_mat_{i}'] = wm.numpy()
    save_dict[f'scale_mat_{i}'] = sm
np.savez(f'{args.data_root}/imfunc4/cameras_hd.npz', **save_dict)

if args.show_range:
    mesh_box = o3d.geometry.TriangleMesh.create_box(size,size,size).translate(np.expand_dims(center-size/2,1)).get_axis_aligned_bounding_box()
    if args.range_source == 'range':
        vis = o3d.geometry.PointCloud()
        vis.points = o3d.utility.Vector3dVector(corners_world_coord)
    elif args.range_source == 'pcd':
        vis = pcd
    else:
        raise ValueError
    o3d.visualization.draw_geometries([vis, mesh_box])