import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import importlib

from utils import rend_util
from model.embedder import *
from model.ray_tracing import RayTracing
from model.sample_network import SampleNetwork

from utils.my_utils import get_pixel_grids, idx_cam2world, idx_img2cam, idx_world2cam, plot_pcd
import model.conf as conf
if os.environ.get('IDR_USE_ENV', '0') == '1' and os.environ.get('IDR_CONF', '') != '':
    print('override conf: ', os.environ.get('IDR_CONF'))
    conf = importlib.import_module(os.environ.get('IDR_CONF'))

class ImplicitNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            d_in,
            d_out,
            dims,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0
    ):
        super().__init__()

        dims = [d_in] + dims + [d_out + 1 + feature_vector_size]

        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    outward = 1
                    torch.nn.init.normal_(lin.weight, mean=outward*np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -outward*bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)

    def forward(self, input, compute_grad=False):
        if self.embed_fn is not None:
            input = self.embed_fn(input)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        return x

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)[:,:1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)

class RenderingNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            mode,
            d_in,
            d_out,
            dims,
            weight_norm=True,
            multires_view=0
    ):
        super().__init__()

        self.mode = mode
        dims = [d_in + feature_vector_size] + dims + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, points, normals, view_dirs, feature_vectors):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        x = self.tanh(x)
        return x

class IDRNetwork(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.feature_vector_size = conf.get_int('feature_vector_size')
        self.implicit_network = ImplicitNetwork(self.feature_vector_size, **conf.get_config('implicit_network'))
        self.rendering_network = RenderingNetwork(self.feature_vector_size, **conf.get_config('rendering_network'))
        self.ray_tracer = RayTracing(**conf.get_config('ray_tracer'))
        self.sample_network = SampleNetwork()
        self.object_bounding_sphere = conf.get_float('ray_tracer.object_bounding_sphere')

    def forward(self, input, train_progress=None):

        # Parse model input
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]
        object_mask_true = input["object_mask"].reshape(-1)
        object_mask = object_mask_true if conf.use_mask else torch.ones_like(object_mask_true)

        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)

        batch_size, num_pixels, _ = ray_dirs.shape

        self.implicit_network.eval()
        with torch.no_grad():
            points, network_object_mask, dists = self.ray_tracer(sdf=lambda x: self.implicit_network(x)[:, 0],
                                                                 cam_loc=cam_loc,
                                                                 object_mask=object_mask,
                                                                 ray_directions=ray_dirs)
        self.implicit_network.train()

        points = (cam_loc.unsqueeze(1) + dists.reshape(batch_size, num_pixels, 1) * ray_dirs).reshape(-1, 3)

        sdf_output_full = self.implicit_network(points)  # M(1+1+256)
        sdf_output = sdf_output_full[:, :1]
        ray_dirs = ray_dirs.reshape(-1, 3)

        if self.training:
            surface_mask = network_object_mask & object_mask
            surface_points = points[surface_mask]
            surface_dists = dists[surface_mask].unsqueeze(-1)
            surface_ray_dirs = ray_dirs[surface_mask]
            surface_cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)[surface_mask]
            surface_output = sdf_output[surface_mask]
            N = surface_points.shape[0]

            # Sample points for the eikonal loss
            eik_bounding_box = self.object_bounding_sphere
            n_eik_points = batch_size * num_pixels // 2
            eikonal_points_rand = torch.empty(n_eik_points, 3).uniform_(-eik_bounding_box, eik_bounding_box).cuda()
            # eikonal_pixel_points = points.clone()
            # eikonal_pixel_points = eikonal_pixel_points.detach()
            eikonal_points = eikonal_points_rand#torch.cat([eikonal_points_rand, eikonal_pixel_points], 0)

            assert(train_progress is not None)

            # extra samples around depth surface
            if any([conf.d_use_dsurf_on(train_progress), conf.d_use_dsurf_jitter(train_progress), 
                    conf.eik_use_dsurf_on(train_progress), conf.eik_use_dsurf_jitter(train_progress)]):
                jitter_rad = 0.1  # NOTE: hard code
                n_dsurf_points = batch_size * num_pixels // 2
                depths = input['depths']  # nv1hw
                depth_cams = input['depth_cams']  # nv244
                center = input['center'][:1]  # 13
                size = input['size'][:1]  # 1
                depths_pack, depth_cams_pack = [arr.view(-1, *arr.size()[2:]) for arr in [depths, depth_cams]]  # N1hw, N244
                depth_surface_points_hom = idx_cam2world(idx_img2cam(
                    get_pixel_grids(*depths.size()[-2:]).unsqueeze(0), depths_pack, depth_cams_pack), depth_cams_pack)  # Nhw41
                depth_surface_points = depth_surface_points_hom[depths_pack[:,0]>0][:,:3,0]  # m3
                depth_surface_points_norm = (depth_surface_points - center) / size * 2  # m3
                depth_surface_jitter = depth_surface_points_norm + torch.rand_like(depth_surface_points_norm) * jitter_rad * 2 - jitter_rad  # m3
                ds_results = []
                for ds in [depth_surface_points_norm, depth_surface_jitter]:
                    inbound = (ds.abs() < eik_bounding_box).float().sum(-1) > 2.9
                    ds_inbound = ds[inbound]  # m3
                    sample_idx = np.sort(np.random.choice(ds_inbound.size()[0], n_dsurf_points, replace=False))
                    ds_sample = ds_inbound[sample_idx]  # m3
                    ds_results.append(ds_sample)
                dsurf_on_sample, dsurf_jitter_sample = ds_results
            else:
                n_dsurf_points = 0
                dsurf_on_sample = torch.zeros(0,3).float().cuda()
                dsurf_jitter_sample = torch.zeros(0,3).float().cuda()

            points_all = torch.cat([surface_points, eikonal_points, dsurf_on_sample, dsurf_jitter_sample], dim=0)

            # on demand sdf calculation
            output = self.implicit_network(points_all[N:])
            # surface_sdf_values = output[:N, 0:1].detach()  # from surface_points
            surface_sdf_values = surface_output.detach()  # from surface_points
            eikonal_output_list = []
            if conf.d_use_rt_surf(train_progress):
                eikonal_output_list.append( (surface_output, points_all[:N]) )
            if conf.d_use_eik(train_progress):
                eikonal_output_list.append( (output[:n_eik_points, :1], points_all[N:N+n_eik_points]) )
            if conf.d_use_dsurf_on(train_progress):
                eikonal_output_list.append( (output[n_eik_points:n_eik_points+n_dsurf_points, :1], points_all[N+n_eik_points:N+n_eik_points+n_dsurf_points]) )
            if conf.d_use_dsurf_jitter(train_progress):
                eikonal_output_list.append( (output[n_eik_points+n_dsurf_points:n_eik_points+2*n_dsurf_points, :1], points_all[N+n_eik_points+n_dsurf_points:N+n_eik_points+2*n_dsurf_points]) )
            eikonal_output = torch.cat([arr[0] for arr in eikonal_output_list], dim=0).view(1, -1)
            eikonal_points_hom = torch.cat([arr[1] for arr in eikonal_output_list], dim=0)
            eikonal_points_hom = torch.cat([eikonal_points_hom, torch.ones_like(eikonal_points_hom[:,-1:])], dim=-1).view(1, -1, 4, 1)

            surf_indicator_output = torch.cat([sdf_output_full[:,1][surface_mask & object_mask_true], output[:n_eik_points, 1]], dim=0)

            # on demand gradient calculation
            g = self.implicit_network.gradient(points_all)
            surface_points_grad = g[:N, 0, :].clone().detach()
            grad_theta_list = []
            if conf.eik_use_rt_surf(train_progress):
                grad_theta_list.append(g[:N, 0, :])
            if conf.eik_use_eik(train_progress):
                grad_theta_list.append(g[N:N+n_eik_points, 0, :])
            if conf.eik_use_dsurf_on(train_progress):
                grad_theta_list.append(g[N+n_eik_points:N+n_eik_points+n_dsurf_points, 0, :])
            if conf.eik_use_dsurf_jitter(train_progress):
                grad_theta_list.append(g[N+n_eik_points+n_dsurf_points:N+n_eik_points+2*n_dsurf_points, 0, :])
            grad_theta = torch.cat(grad_theta_list, dim=0)

            differentiable_surface_points = self.sample_network(surface_output,
                                                                surface_sdf_values,
                                                                surface_points_grad,
                                                                surface_dists,
                                                                surface_cam_loc,
                                                                surface_ray_dirs)

        else:
            surface_mask = network_object_mask
            differentiable_surface_points = points[surface_mask]
            grad_theta = None

        view = -ray_dirs[surface_mask]

        rgb_values = torch.ones_like(points).float().cuda()
        if differentiable_surface_points.shape[0] > 0:
            rgb_values[surface_mask] = self.get_rbg_value(differentiable_surface_points, view, train_progress)

        output = {
            'points': points,
            'diff_surf_pts': differentiable_surface_points,
            'rgb_values': rgb_values,
            'sdf_output': sdf_output,
            'network_object_mask': network_object_mask,
            'object_mask': object_mask,
            'object_mask_true': object_mask_true,
            'grad_theta': grad_theta
        }

        if self.training:
            output['eikonal_points_hom'] = eikonal_points_hom
            output['eikonal_output'] = eikonal_output
            output['surf_indicator_output'] = surf_indicator_output

        return output

    def get_rbg_value(self, points, view_dirs, train_progress):
        output = self.implicit_network(points)
        g = self.implicit_network.gradient(points)
        normals = g[:, 0, :]

        feature_vectors = output[:, 2:]

        if train_progress is not None and train_progress < conf.phase[0] or conf.disable_rgb_grad:
            points, normals, view_dirs = [
                arr.detach() for arr in [points, normals, view_dirs]
            ]

        rgb_vals = self.rendering_network(points, normals, view_dirs, feature_vectors)

        return rgb_vals
