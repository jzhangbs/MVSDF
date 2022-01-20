import sys
sys.path.append('../code')
import argparse
import GPUtil

from training.idr_train import IDRTrainRunner

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='fill_in_data_dir')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
    parser.add_argument('--nepoch', type=int, default=1800, help='number of epochs to train for')
    parser.add_argument('--conf', type=str, default='./confs/mvsdf_dtu.conf')
    parser.add_argument('--expname', type=str, default='test')
    parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto]')
    parser.add_argument('--is_continue', default=False, action="store_true", help='If set, indicates continuing from a previous run.')
    parser.add_argument('--timestamp', default='latest', type=str, help='The timestamp of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--checkpoint', default='latest',type=str,help='The checkpoint epoch number of the run to be used in case of continuing from a previous run.')
    # parser.add_argument('--train_cameras', default=False, action="store_true", help='If set, optimizing also camera location.')

    opt = parser.parse_args()

    if opt.gpu == "auto":
        deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[])
        gpu = deviceIDs[0]
    else:
        gpu = opt.gpu

    trainrunner = IDRTrainRunner(conf=opt.conf,
                                 data_dir=opt.data_dir,
                                 batch_size=opt.batch_size,
                                 nepochs=opt.nepoch,
                                 expname=opt.expname,
                                 gpu_index=gpu,
                                 exps_folder_name='exps',
                                 is_continue=opt.is_continue,
                                 timestamp=opt.timestamp,
                                 checkpoint=opt.checkpoint,
                                 train_cameras=False
                                 )

    trainrunner.run()
