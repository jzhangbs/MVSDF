import os
import importlib
import torch
import numpy as np
import torch.nn.functional as F

import utils.general as utils
from utils import rend_util

from utils.my_utils import load_pair, load_pfm, load_cam, scale_camera, FeatExt
import model.conf as conf
if os.environ.get('IDR_USE_ENV', '0') == '1' and os.environ.get('IDR_CONF', '') != '':
    print('override conf: ', os.environ.get('IDR_CONF'))
    conf = importlib.import_module(os.environ.get('IDR_CONF'))

class SceneDataset(torch.utils.data.Dataset):
    """Dataset for a class of objects, where each datapoint is a SceneInstanceDataset."""

    def __init__(self,
                 data_dir,
                 train_cameras,
                 cam_file=None
                 ):

        self.instance_dir = data_dir
        self.only_cam = (os.environ.get('IDR_USE_ENV', '0') == '1' and os.environ.get('IDR_ONLY_CAM', '0') == '1')

        if self.only_cam:
            img_res = [int(v) for v in os.environ.get('IDR_SIZE').strip().split(',')]
            self.total_pixels = img_res[0] * img_res[1]
            self.img_res = img_res
            assert os.path.exists(self.instance_dir), "Data directory is empty"
            self.sampling_idx = None
            self.train_cameras = train_cameras
            uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
            uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
            self.uv = uv.reshape(2, -1).transpose(1, 0)

            self.cam_file = '{0}/cameras_hd.npz'.format(self.instance_dir)
            if cam_file is not None:
                self.cam_file = '{0}/{1}'.format(self.instance_dir, cam_file)

            camera_dict = np.load(self.cam_file)
            self.n_images = len(list(camera_dict.items())) // 2
            scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
            world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

            self.intrinsics_all = []
            self.pose_all = []
            for scale_mat, world_mat in zip(scale_mats, world_mats):
                P = world_mat @ scale_mat
                P = P[:3, :4]
                intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
                self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
                self.pose_all.append(torch.from_numpy(pose).float())
            self.object_masks = [torch.ones(self.total_pixels).bool() for _ in range(self.n_images)]
        else:
            assert os.path.exists(self.instance_dir), "Data directory is empty"

            self.sampling_idx = None
            self.train_cameras = train_cameras

            image_dir = '{0}/image_hd'.format(self.instance_dir)
            image_paths = sorted(utils.glob_imgs(image_dir))
            mask_dir = '{0}/mask_hd'.format(self.instance_dir)
            mask_paths = sorted(utils.glob_imgs(mask_dir))

            self.n_images = len(image_paths)

            self.cam_file = '{0}/cameras_hd.npz'.format(self.instance_dir)
            if cam_file is not None:
                self.cam_file = '{0}/{1}'.format(self.instance_dir, cam_file)

            camera_dict = np.load(self.cam_file)
            scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
            world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

            self.intrinsics_all = []
            self.pose_all = []
            for scale_mat, world_mat in zip(scale_mats, world_mats):
                P = world_mat @ scale_mat
                P = P[:3, :4]
                intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
                self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
                self.pose_all.append(torch.from_numpy(pose).float())

            self.rgb_images = []
            for path in image_paths:
                rgb = rend_util.load_rgb(path)
                input_res = rgb.shape[-2:]
                rgb = rgb.reshape(3, -1).transpose(1, 0)
                self.rgb_images.append(torch.from_numpy(rgb).float())
            
            self.img_res = input_res
            self.total_pixels = self.img_res[0] * self.img_res[1]

            self.object_masks = []
            for path in mask_paths:
                object_mask = rend_util.load_mask(path)
                object_mask = object_mask.reshape(-1)
                self.object_masks.append(torch.from_numpy(object_mask).bool())
            
            self.pair = load_pair(f'{self.instance_dir}/../pair.txt')
            self.num_src = 2  # NOTE: hard code
            self.depths = torch.stack(
                [torch.from_numpy(np.ascontiguousarray(
                    load_pfm(f'{self.instance_dir}/depth/{i:03}.pfm'))).to(torch.float32) 
                    for i in range(self.n_images)], dim=0).unsqueeze(1)
            self.depth_cams = torch.stack(
                [torch.from_numpy(
                    load_cam(f'{self.instance_dir}/../cam_{i:08}_flow3.txt', 256, 1)).to(torch.float32) 
                    for i in range(self.n_images)], dim=0)
            self.feat_img_scale = conf.feat_img_scale
            self.cams_hd = torch.stack(
                [scale_camera(self.depth_cams[i], self.feat_img_scale) for i in range(self.n_images)]  # NOTE: hard code
            )
            self.rgb_2xd = torch.stack([
                F.interpolate(
                    self.rgb_images[idx].permute(1,0).view(1, 3, *self.img_res), 
                    size=(self.depths.size()[-2]*self.feat_img_scale, self.depths.size()[-1]*self.feat_img_scale), mode='bilinear', align_corners=False)[0]
                if self.img_res[0] != self.depths.size()[-2]*self.feat_img_scale or self.img_res[1] != self.depths.size()[-1]*self.feat_img_scale else
                    self.rgb_images[idx].permute(1,0).view(3, *self.img_res)
                for idx in range(self.n_images)
            ], dim=0)  # v3hw

            mean = torch.tensor([0.485, 0.456, 0.406]).float()
            std = torch.tensor([0.229, 0.224, 0.225]).float()
            self.rgb_2xd = (self.rgb_2xd / 2 + 0.5 - mean.view(1,3,1,1)) / std.view(1,3,1,1)

            self.size = torch.from_numpy(scale_mats[0]).float()[0,0] * 2
            self.center = torch.from_numpy(scale_mats[0]).float()[:3,3]
            self.sel_depth_num = 1  # NOTE: hard code, fixed

            uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
            uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
            self.uv = uv.reshape(2, -1).transpose(1, 0)

            # calc feat
            feat_ext = FeatExt().cuda()
            feat_ext.eval()
            for p in feat_ext.parameters():
                p.requires_grad = False
            feats = []
            feat_eval_bs = 20
            for start_i in range(0, self.n_images, feat_eval_bs):
                eval_batch = self.rgb_2xd[start_i:start_i+feat_eval_bs]
                feat2 = feat_ext(eval_batch.cuda())[2].detach().cpu()
                feats.append(feat2)
            self.feats = torch.cat(feats, dim=0)

            if os.path.exists(f'{self.instance_dir}/pmask'):
                pmask_dir = f'{self.instance_dir}/pmask'
                print('find perfect mask dir:', pmask_dir)

                pmask_paths = sorted(utils.glob_imgs(pmask_dir))
                self.perfect_masks = []
                for path in pmask_paths:
                    perfect_mask = rend_util.load_mask(path)
                    perfect_mask = perfect_mask.reshape(-1)
                    self.perfect_masks.append(torch.from_numpy(perfect_mask).bool())

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        if self.only_cam:
            sample = {
                "object_mask": self.object_masks[idx],
                "uv": self.uv,
                "intrinsics": self.intrinsics_all[idx],
            }
            if self.sampling_idx is not None:
                sample["object_mask"] = self.object_masks[idx][self.sampling_idx]
                sample["uv"] = self.uv[self.sampling_idx, :]
            if not self.train_cameras:
                sample["pose"] = self.pose_all[idx]
            sample['perfect_mask'] = sample['object_mask']
            ground_truth = {}
        else:
            sample = {
                "object_mask": self.object_masks[idx],
                "uv": self.uv,
                "intrinsics": self.intrinsics_all[idx],
            }

            if hasattr(self, 'perfect_masks'):
                sample['perfect_mask'] = self.perfect_masks[idx]

            ground_truth = {
                "rgb": self.rgb_images[idx]
            }

            if self.sampling_idx is not None:
                ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
                sample["object_mask"] = self.object_masks[idx][self.sampling_idx]
                sample["uv"] = self.uv[self.sampling_idx, :]
                if hasattr(self, 'perfect_masks'):
                    sample["perfect_mask"] = self.perfect_masks[idx][self.sampling_idx]

            if not self.train_cameras:
                sample["pose"] = self.pose_all[idx]

            views = [i for i in range(self.n_images) if i != idx]
            sel_depth_idxs = np.sort(np.concatenate([np.random.choice(views, self.sel_depth_num-1, replace=False), [idx]]))
            ground_truth['depths'] = self.depths[sel_depth_idxs].cuda()
            ground_truth['depth_cams'] = self.depth_cams[sel_depth_idxs].cuda()
            ground_truth['size'] = self.size.cuda()
            ground_truth['center'] = self.center.cuda()

            # ground_truth["rgb_c"] = self.rgb_2xd[idx].cuda()
            ground_truth["feat"] = self.feats[idx].cuda()
            # ground_truth["rgb_src_c"] = torch.stack([self.rgb_2xd[int(self.pair[str(idx)]['pair'][i])] for i in range(self.num_src)]).cuda()
            ground_truth["feat_src"] = self.feats[ [int(self.pair[str(idx)]['pair'][i]) for i in range(self.num_src)] ].cuda()
            ground_truth["cam"] = self.cams_hd[idx].cuda()
            ground_truth["src_cams"] = self.cams_hd[ [int(self.pair[str(idx)]['pair'][i]) for i in range(self.num_src)] ].cuda()

            for attr in ['depths', 'depth_cams', 'size', 'center', 'cam', 'src_cams']:
                sample[attr] = ground_truth[attr]
            # for attr in ['size', 'center']:
            #     sample[attr] = ground_truth[attr]

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    def get_scale_mat(self):
        return np.load(self.cam_file)['scale_mat_0']

    def get_gt_pose(self, scaled=False):
        # Load gt pose without normalization to unit sphere
        camera_dict = np.load(self.cam_file)
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat
            if scaled:
                P = world_mat @ scale_mat
            P = P[:3, :4]
            _, pose = rend_util.load_K_Rt_from_P(None, P)
            pose_all.append(torch.from_numpy(pose).float())

        return torch.cat([p.float().unsqueeze(0) for p in pose_all], 0)

    def get_pose_init(self):
        # get noisy initializations obtained with the linear method
        cam_file = '{0}/cameras_linear_init.npz'.format(self.instance_dir)
        camera_dict = np.load(cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        init_pose = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            _, pose = rend_util.load_K_Rt_from_P(None, P)
            init_pose.append(pose)
        init_pose = torch.cat([torch.Tensor(pose).float().unsqueeze(0) for pose in init_pose], 0).cuda()
        init_quat = rend_util.rot_to_quat(init_pose[:, :3, :3])
        init_quat = torch.cat([init_quat, init_pose[:, :3, 3]], 1)

        return init_quat
