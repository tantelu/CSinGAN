import argparse
from torch.utils.data import DataLoader
import torch
from torch.nn import functional as F
import torch.nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import os
from models.generator import Generator
from models.discriminator import Discriminator
from train import *
from validation import *
from utils import *
from datasets.datasetgetter import get_dataset

parser = argparse.ArgumentParser(description='PyTorch Simultaneous Training')
parser.add_argument('--data_dir', default='../data/', help='path to dataset')
parser.add_argument('--dataset', default='PHOTO', help='type of dataset', choices=['PHOTO'])
parser.add_argument('--gantype', default='zerogp', help='type of GAN loss', choices=['wgangp', 'zerogp', 'lsgan'])
parser.add_argument('--model_name', type=str, default='SinGAN', help='model name')
parser.add_argument('--workers', default=0, type=int, help='number of data loading workers (default: 8)')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Total batch size - e.g) num_gpus = 2 , batch_size = 128 then, effectively, 64')
parser.add_argument('--val_batch', default=1, type=int)
parser.add_argument('--img_size_max', default=250, type=int, help='Input image size')
parser.add_argument('--img_size_min', default=25, type=int, help='Input image size')
parser.add_argument('--total_iter', default=500, type=int, help='total num of iter')
parser.add_argument('--decay_lr', default=500, type=int, help='learning rate change iter times')
parser.add_argument('--img_to_use', default=-999, type=int, help='Index of the input image to use < 6287')
parser.add_argument('--load_model', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: None)')
parser.add_argument('--validation', dest='validation', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--test', dest='test', action='store_true',
                    help='test model on validation set')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--gpu', default='1', type=str,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--port', default='8888', type=str)

def run30UnconditionGaussian():
    condi_parser = argparse.ArgumentParser()
    condi_parser.add_argument('--model_file', default='D:/Data/Paper/SinGAN/网络模型/NetModel/logs-gaussian-3kernel-zerogp/gen.pkl')
    condi_parser.add_argument('--save_folder', default='D:/Data/Paper/SinGAN/图/高斯场与非平稳河道/')
    condi_parser.add_argument('--batch_size', default=1)
    condi_parser.add_argument('--img_ch', default=1)
    condi_args=condi_parser.parse_args()
    condi_args.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    condi_args.size_list=[25,33,44,60,80,108,144,192,250]
    condi_args.num_scale = len(condi_args.size_list)-1
    rmse_list=np.loadtxt(os.path.join(os.path.dirname(condi_args.model_file),'rmse_list.txt'))
    condi_args.generator=torch.load(condi_args.model_file)
    condi_args.generator.eval()
    if condi_args.save_folder is None:
        return
    for gen_nums in range(30):
        z_list = [F.pad(rmse_list[z_idx] * torch.randn(condi_args.batch_size, condi_args.img_ch, condi_args.size_list[z_idx],
                        condi_args.size_list[z_idx]).to(condi_args.device),
                        [5, 5, 5, 5], value=0) for z_idx in range(condi_args.num_scale+1)]
        x_fake_list = condi_args.generator(z_list)
        file_names=["gaussian_{0}.gslib".format(gen_nums+1)]
        imgs = x_fake_list[-1].detach().cpu().numpy()
        for i in range(imgs.shape[0]):
            img=imgs[i]
            if file_names is not None:
                name=os.path.join(condi_args.save_folder,file_names[i]) 
            else:
                name=os.path.join(condi_args.save_folder,"{0}_{1}.gslib".format('gaussian',i+1))
            helper.WriteGslibFile(img.squeeze(),name)
if __name__ == '__main__':
    args = parser.parse_args()
    train_dataset, _ = get_dataset(args.dataset, args)
    run30UnconditionGaussian()