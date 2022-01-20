feat_img_scale = 2

phase = (1/6, 1/2)

d_use_rt_surf =        lambda tp: True if tp < phase[0] else (True if tp < phase[1] else True)
d_use_eik =            lambda tp: True if tp < phase[0] else (True if tp < phase[1] else True)
d_use_dsurf_on =       lambda tp: True if tp < phase[0] else (False if tp < phase[1] else False)
d_use_dsurf_jitter =   lambda tp: True if tp < phase[0] else (False if tp < phase[1] else False)
eik_use_rt_surf =      lambda tp: True if tp < phase[0] else (True if tp < phase[1] else True)
eik_use_eik =          lambda tp: True if tp < phase[0] else (True if tp < phase[1] else True)
eik_use_dsurf_on =     lambda tp: True if tp < phase[0] else (False if tp < phase[1] else False)
eik_use_dsurf_jitter = lambda tp: True if tp < phase[0] else (False if tp < phase[1] else False)

disable_rgb_grad = False

use_invalid = False
use_mask = False
out_thresh_perc = 1/8
enable_feat = True
enable_rgb = True
far_thresh = 0.25
far_att = lambda tp: (1 if tp < phase[0] else (1 if tp < phase[1] else 1))
near_thresh = 0.1
near_att = lambda tp: (1 if tp < phase[0] else (0.1 if tp < phase[1] else 0.01))
smooth = lambda tp: None
rgb_weight = lambda tp: (0.5 if tp < phase[0] else (0.5 if tp < phase[1] else 0.5))
surf_weight = 0.01
feat_weight = lambda tp: (0 if tp < phase[0] else (0.1 if tp < phase[1] else 0.01))
depth_weight = lambda tp: (1 if tp < phase[0] else (1 if tp < phase[1] else 1))
eikonal_weight = 0.1

enable_grad_cap = True
grad_cap = lambda tp: 2 if tp < phase[1] else 0.5
