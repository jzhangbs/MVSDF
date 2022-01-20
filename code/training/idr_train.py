import os
from datetime import datetime
from pyhocon import ConfigFactory
import sys
import torch
import importlib

from datasets.scene_dataset import SceneDataset
from model.implicit_differentiable_renderer import IDRNetwork
from model.loss import IDRLoss
import utils.general as utils
import utils.plots as plt
import model.conf as conf
if os.environ.get('IDR_USE_ENV', '0') == '1' and os.environ.get('IDR_CONF', '') != '':
    print('override conf: ', os.environ.get('IDR_CONF'))
    conf = importlib.import_module(os.environ.get('IDR_CONF'))

class IDRTrainRunner():
    def __init__(self,**kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.data_dir = kwargs['data_dir']
        self.batch_size = kwargs['batch_size']
        self.nepochs = kwargs['nepochs']
        self.exps_folder_name = kwargs['exps_folder_name']
        self.GPU_INDEX = kwargs['gpu_index']
        self.train_cameras = kwargs['train_cameras']

        self.expname = self.conf.get_string('train.expname') + '_' + kwargs['expname']

        if kwargs['is_continue'] and kwargs['timestamp'] == 'latest':
            if os.path.exists(os.path.join('../',kwargs['exps_folder_name'],self.expname)):
                timestamps = os.listdir(os.path.join('../',kwargs['exps_folder_name'],self.expname))
                if (len(timestamps)) == 0:
                    is_continue = False
                    timestamp = None
                else:
                    timestamp = sorted(timestamps)[-1]
                    is_continue = True
            else:
                is_continue = False
                timestamp = None
        else:
            timestamp = kwargs['timestamp']
            is_continue = kwargs['is_continue']

        utils.mkdir_ifnotexists(os.path.join('../',self.exps_folder_name))
        self.expdir = os.path.join('../', self.exps_folder_name, self.expname)
        utils.mkdir_ifnotexists(self.expdir)
        self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))

        self.plots_dir = os.path.join(self.expdir, self.timestamp, 'plots')
        utils.mkdir_ifnotexists(self.plots_dir)

        # create checkpoints dirs
        self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)
        self.model_params_subdir = "ModelParameters"
        self.optimizer_params_subdir = "OptimizerParameters"
        self.scheduler_params_subdir = "SchedulerParameters"

        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.scheduler_params_subdir))

        if self.train_cameras:
            self.optimizer_cam_params_subdir = "OptimizerCamParameters"
            self.cam_params_subdir = "CamParameters"

            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_cam_params_subdir))
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.cam_params_subdir))

        # os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.expdir, self.timestamp, 'runconf.conf')))

        if (not self.GPU_INDEX == 'ignore'):
            os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(self.GPU_INDEX)

        print('shell command : {0}'.format(' '.join(sys.argv)))

        print('Loading data ...')

        dataset_conf = self.conf.get_config('dataset')

        self.train_dataset = SceneDataset(self.data_dir, self.train_cameras, **dataset_conf)

        print('Finish loading data ...')

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            drop_last=True,
                                                            collate_fn=self.train_dataset.collate_fn
                                                            )
        self.plot_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                           batch_size=self.conf.get_int('plot.plot_nimgs'),
                                                           shuffle=True,
                                                           collate_fn=self.train_dataset.collate_fn
                                                           )

        self.model = IDRNetwork(conf=self.conf.get_config('model'))
        if torch.cuda.is_available():
            self.model.cuda()

        self.loss = IDRLoss(**self.conf.get_config('loss'))

        self.num_pixels = self.conf.get_int('train.num_pixels')
        self.lr = self.conf.get_float('train.learning_rate')
        self.lr *= self.batch_size
        print('batch size scaled lr:', self.lr)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.sched_milestones = self.conf.get_list('train.sched_milestones', default=[])
        self.sched_milestones = [v.split('/') for v in self.sched_milestones]
        self.sched_milestones = [int(v[0]) / int(v[1]) for v in self.sched_milestones]
        self.sched_milestones = [int(self.nepochs * float(v)) for v in self.sched_milestones]
        self.sched_factor = self.conf.get_float('train.sched_factor', default=0.0)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.sched_milestones, gamma=self.sched_factor)

        # settings for camera optimization
        if self.train_cameras:
            num_images = len(self.train_dataset)
            self.pose_vecs = torch.nn.Embedding(num_images, 7, sparse=True).cuda()
            self.pose_vecs.weight.data.copy_(self.train_dataset.get_pose_init())

            self.optimizer_cam = torch.optim.SparseAdam(self.pose_vecs.parameters(), self.conf.get_float('train.learning_rate_cam'))

        self.start_epoch = 0
        if is_continue:
            old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints')

            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
            self.model.load_state_dict(saved_model_state["model_state_dict"])
            self.start_epoch = saved_model_state['epoch']

            data = torch.load(
                os.path.join(old_checkpnts_dir, 'OptimizerParameters', str(kwargs['checkpoint']) + ".pth"))
            self.optimizer.load_state_dict(data["optimizer_state_dict"])

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.scheduler.load_state_dict(data["scheduler_state_dict"])

            if self.train_cameras:
                data = torch.load(
                    os.path.join(old_checkpnts_dir, self.optimizer_cam_params_subdir, str(kwargs['checkpoint']) + ".pth"))
                self.optimizer_cam.load_state_dict(data["optimizer_cam_state_dict"])

                data = torch.load(
                    os.path.join(old_checkpnts_dir, self.cam_params_subdir, str(kwargs['checkpoint']) + ".pth"))
                self.pose_vecs.load_state_dict(data["pose_vecs_state_dict"])

        self.total_pixels = self.train_dataset.total_pixels
        self.img_res = self.train_dataset.img_res
        self.n_batches = len(self.train_dataloader)
        self.plot_freq = self.conf.get_string('train.plot_freq')
        self.plot_freq = self.plot_freq.split('/')
        self.plot_freq = int(self.plot_freq[0]) / int(self.plot_freq[1])
        self.plot_freq = int(self.plot_freq * self.nepochs)
        self.plot_conf = self.conf.get_config('plot')

    def save_checkpoints(self, epoch):
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, "latest.pth"))

        if self.train_cameras:
            torch.save(
                {"epoch": epoch, "optimizer_cam_state_dict": self.optimizer_cam.state_dict()},
                os.path.join(self.checkpoints_path, self.optimizer_cam_params_subdir, str(epoch) + ".pth"))
            torch.save(
                {"epoch": epoch, "optimizer_cam_state_dict": self.optimizer_cam.state_dict()},
                os.path.join(self.checkpoints_path, self.optimizer_cam_params_subdir, "latest.pth"))

            torch.save(
                {"epoch": epoch, "pose_vecs_state_dict": self.pose_vecs.state_dict()},
                os.path.join(self.checkpoints_path, self.cam_params_subdir, str(epoch) + ".pth"))
            torch.save(
                {"epoch": epoch, "pose_vecs_state_dict": self.pose_vecs.state_dict()},
                os.path.join(self.checkpoints_path, self.cam_params_subdir, "latest.pth"))
    

    def plot_epoch(self, epoch, full=False):
        self.model.eval()
        if self.train_cameras:
            self.pose_vecs.eval()

        if full:
            self.train_dataset.change_sampling_idx(-1)
            indices, model_input, ground_truth = next(iter(self.plot_dataloader))

            model_input["intrinsics"] = model_input["intrinsics"].cuda()
            model_input["uv"] = model_input["uv"].cuda()
            model_input["object_mask"] = model_input["object_mask"].cuda()

            if self.train_cameras:
                pose_input = self.pose_vecs(indices.cuda())
                model_input['pose'] = pose_input
            else:
                model_input['pose'] = model_input['pose'].cuda()

            split = utils.split_input(model_input, self.total_pixels)
            res = []
            for s in split:
                out = self.model(s)
                res.append({
                    'points': out['points'].detach(),
                    'rgb_values': out['rgb_values'].detach(),
                    'network_object_mask': out['network_object_mask'].detach(),
                    'object_mask': out['object_mask'].detach()
                })

            batch_size = ground_truth['rgb'].shape[0]
            model_outputs = utils.merge_output(res, self.total_pixels, batch_size)

            plt.plot(self.model,
                        indices,
                        model_outputs,
                        model_input['pose'],
                        ground_truth['rgb'],
                        self.plots_dir,
                        epoch,
                        self.img_res,
                        **self.plot_conf
                        )
        else:
            plt.get_surface_trace(path=self.plots_dir, epoch=epoch, 
                sdf=lambda x: self.model.implicit_network(x)[:, 0], resolution=self.plot_conf['resolution'])

        self.model.train()
        if self.train_cameras:
            self.pose_vecs.train()

    def train_epoch(self, epoch):
        self.train_dataset.change_sampling_idx(self.num_pixels)
        for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):
            train_progress = epoch/self.nepochs

            model_input["intrinsics"] = model_input["intrinsics"].cuda()
            model_input["uv"] = model_input["uv"].cuda()
            model_input["object_mask"] = model_input["object_mask"].cuda()

            if self.train_cameras:
                pose_input = self.pose_vecs(indices.cuda())
                model_input['pose'] = pose_input
            else:
                model_input['pose'] = model_input['pose'].cuda()

            model_outputs = self.model(model_input, train_progress)
            loss_output = self.loss(model_outputs, ground_truth, train_progress, len(self.train_dataloader))

            loss = loss_output['loss']
            
            # for ln, p in [('depth_loss', 0), ('feat_loss', phase[0]), ('rgb_loss', 1e-5), ('surf_loss', phase[0])]:
            #     if train_progress >= p:
            #         result = torch.autograd.grad(
            #             outputs = [loss_output[ln]],
            #             inputs = self.model.implicit_network.parameters(),
            #             retain_graph=True
            #         )
            #         grad_norm = torch.cat([p.flatten() for p in result]).norm()
            #         print(ln, grad_norm.item())

            self.optimizer.zero_grad()
            if self.train_cameras:
                self.optimizer_cam.zero_grad()

            loss.backward()

            all_norm = torch.cat([p.grad.flatten() for p in self.model.parameters() if p.grad is not None]).norm()
            print('grad norm:', all_norm.item())
            if conf.phase[0] <= train_progress and conf.enable_grad_cap:
                grad_cap = conf.grad_cap(train_progress)
                print('grad cap:', grad_cap)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_cap)

            # if all_norm.item() > 100:
            #     print('large grad, skip')
            #     self.optimizer.zero_grad()
            #     if self.train_cameras:
            #         self.optimizer_cam.zero_grad()

            self.optimizer.step()
            if self.train_cameras:
                self.optimizer_cam.step()

            print(f"{self.expname} [{epoch}/{self.nepochs}] ({data_index}/{self.n_batches}):",
            f"loss = {loss.item():.4f},",
            f"rgb_loss = {loss_output['rgb_loss'].item():.4f},",
            f"eikonal_loss = {loss_output['eikonal_loss'].item():.4f},",
            f"feat_loss = {loss_output['feat_loss'].item():.4f},",
            f"depth_loss = {loss_output['depth_loss'].item():.4f},",
            f"surf_loss = {loss_output['surf_loss'].item():.4f},",
            f"lr = {self.scheduler.get_lr()[0]}")

        self.scheduler.step()

    def run(self):
        print("training...")

        for epoch in range(self.start_epoch, self.nepochs + 1):
            
            self.train_epoch(epoch)

            if epoch % self.plot_freq == 0 and epoch != 0:
                self.save_checkpoints(epoch)

            if epoch % self.plot_freq == 0 and epoch != 0:
                self.plot_epoch(epoch, full=(epoch % (self.plot_freq*4) == 0) )
