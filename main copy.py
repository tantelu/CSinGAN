import argparse
import warnings
from datetime import datetime
from glob import glob
from shutil import copyfile
from datasets.datasetgetter import get_dataset
from datasets.cub200 import pil_loader
import torch
from torch import autograd
from torch.nn import functional as F
import torch.nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from models.generator import Generator
from models.discriminator import Discriminator

from train import *
from validation import *

from utils import *

parser = argparse.ArgumentParser(description='PyTorch Simultaneous Training')
parser.add_argument('--data_dir', default='../data/', help='path to dataset')
parser.add_argument('--dataset', default='PHOTO', help='type of dataset', choices=['PHOTO'])
parser.add_argument('--gantype', default='zerogp', help='type of GAN loss', choices=['wgangp', 'zerogp', 'lsgan'])
parser.add_argument('--model_name', type=str, default='SinGAN', help='model name')
parser.add_argument('--workers', default=0, type=int, help='number of data loading workers (default: 8)')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Total batch size - e.g) num_gpus = 2 , batch_size = 128 then, effectively, 64')
parser.add_argument('--val_batch', default=1, type=int)
parser.add_argument('--img_size_max', default=120, type=int, help='Input image size')
parser.add_argument('--img_size_min', default=25, type=int, help='Input image size')
parser.add_argument('--total_iter', default=500, type=int, help='total num of iter')
parser.add_argument('--decay_lr', default=500, type=int, help='learning rate change iter times')
parser.add_argument('--img_to_use', default=1, type=int, help='Index of the input image to use < 6287')
parser.add_argument('--load_model', default='SinGAN_2022-10-19_21-00-46', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: None)')
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


def main():
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()

    if args.load_model is None:
        args.model_name = '{}_{}'.format(args.model_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    else:
        args.model_name = args.load_model

    makedirs('./logs')
    makedirs('./results')

    args.log_dir = os.path.join('./logs', args.model_name)
    args.res_dir = os.path.join('./results', args.model_name)

    makedirs(args.log_dir)
    makedirs(os.path.join(args.log_dir, 'codes'))
    makedirs(os.path.join(args.log_dir, 'codes', 'models'))
    makedirs(args.res_dir)

    if args.load_model is None:
        pyfiles = glob("./*.py")
        modelfiles = glob('./models/*.py')
        for py in pyfiles:
            copyfile(py, os.path.join(args.log_dir, 'codes') + "/" + py)
        for py in modelfiles:
            copyfile(py, os.path.join(args.log_dir, 'codes', py[2:]))

    formatted_print('Total Number of GPUs:', ngpus_per_node)
    formatted_print('Total Number of Workers:', args.workers)
    formatted_print('Batch Size:', args.batch_size)
    formatted_print('Max image Size:', args.img_size_max)
    formatted_print('Min image Size:', args.img_size_min)
    formatted_print('Log DIR:', args.log_dir)
    formatted_print('Result DIR:', args.res_dir)
    formatted_print('GAN TYPE:', args.gantype)

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
        if len(args.gpu) == 1:
            args.gpu = 0
        else:
            args.gpu = gpu

    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:'+args.port,
                                world_size=args.world_size, rank=args.rank)

    ################
    # Define model #
    ################
    # 4/3 : scale factor in the paper
    scale_factor = 4/3
    tmp_scale = args.img_size_max / args.img_size_min
    args.num_scale = int(np.round(np.log(tmp_scale) / np.log(scale_factor)))
    args.size_list = [int(args.img_size_min * scale_factor**i) for i in range(args.num_scale + 1)]
    
    discriminator = Discriminator()
    generator = Generator(args.img_size_min, args.num_scale, scale_factor)

    networks = [discriminator, generator]

    if args.distributed:
        if args.gpu is not None:
            print('Distributed to', args.gpu)
            torch.cuda.set_device(args.gpu)
            networks = [x.cuda(args.gpu) for x in networks]
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            networks = [torch.nn.parallel.DistributedDataParallel(x, device_ids=[args.gpu], output_device=args.gpu) for x in networks]
        else:
            networks = [x.cuda() for x in networks]
            networks = [torch.nn.parallel.DistributedDataParallel(x) for x in networks]

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        networks = [x.cuda(args.gpu) for x in networks]
    else:
        # device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        networks = [x.cuda(args.gpu) for x in networks]

    discriminator, generator, = networks

    ######################
    # Loss and Optimizer #
    ######################
    if args.distributed:
        d_opt = torch.optim.Adam(discriminator.module.sub_discriminators[0].parameters(), 5e-4, (0.5, 0.999))
        g_opt = torch.optim.Adam(generator.module.sub_generators[0].parameters(), 5e-4, (0.5, 0.999))
    else:
        d_opt = torch.optim.Adam(discriminator.sub_discriminators[0].parameters(), 5e-4, (0.5, 0.999))
        g_opt = torch.optim.Adam(generator.sub_generators[0].parameters(), 5e-4, (0.5, 0.999))

    ##############
    # Load model #
    ##############
    args.stage = 0
    if args.load_model is not None:
        check_load = open(os.path.join(args.log_dir, "checkpoint.txt"), 'r')
        to_restore = check_load.readlines()[-1].strip()
        load_file = os.path.join(args.log_dir, to_restore)
        if os.path.isfile(load_file):
            print("=> loading checkpoint '{}'".format(load_file))
            checkpoint = torch.load(load_file, map_location='cpu')
            for _ in range(int(checkpoint['stage'])):
                generator.progress()
                discriminator.progress()
            networks = [discriminator, generator]
            if args.distributed:
                if args.gpu is not None:
                    print('Distributed to', args.gpu)
                    torch.cuda.set_device(args.gpu)
                    networks = [x.cuda(args.gpu) for x in networks]
                    args.batch_size = int(args.batch_size / ngpus_per_node)
                    args.workers = int(args.workers / ngpus_per_node)
                    networks = [
                        torch.nn.parallel.DistributedDataParallel(x, device_ids=[args.gpu], output_device=args.gpu) for
                        x in networks]
                else:
                    networks = [x.cuda() for x in networks]
                    networks = [torch.nn.parallel.DistributedDataParallel(x) for x in networks]

            elif args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                networks = [x.cuda(args.gpu) for x in networks]
            else:
                networks = [torch.nn.DataParallel(x).cuda() for x in networks]

            discriminator, generator, = networks

            args.stage = checkpoint['stage']
            args.img_to_use = checkpoint['img_to_use']
            discriminator.load_state_dict(checkpoint['D_state_dict'])
            generator.load_state_dict(checkpoint['G_state_dict'])
            d_opt.load_state_dict(checkpoint['d_optimizer'])
            g_opt.load_state_dict(checkpoint['g_optimizer'])
            print("=> loaded checkpoint '{}' (stage {})"
                  .format(load_file, checkpoint['stage']))
        else:
            print("=> no checkpoint found at '{}'".format(args.log_dir))

    cudnn.benchmark = True

    ###########
    # Dataset #
    ###########
    train_dataset, valid = get_dataset(args.dataset, args)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    ######################
    # Validate and Train #
    ######################
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=args.batch_size,
                                               shuffle=(train_sampler is None), num_workers=args.workers,
                                               pin_memory=True, sampler=train_sampler)
    z_fix_list = [F.pad(torch.randn(args.batch_size, 3, args.size_list[0], args.size_list[0]), [5, 5, 5, 5], value=0)]
    zero_list = [F.pad(torch.zeros(args.batch_size, 3, args.size_list[zeros_idx], args.size_list[zeros_idx]),
                       [5, 5, 5, 5], value=0) for zeros_idx in range(1, args.num_scale + 1)]
    z_rec = z_fix_list + zero_list

    x_in = next(iter(valid_loader))
    x_in = x_in.cuda(args.gpu, non_blocking=True)
    x_org = x_in
    
    x_in_list = []
    for xidx in range(0, args.stage):
        x_tmp = F.interpolate(x_org, (args.size_list[xidx], args.size_list[xidx]), mode='bilinear', align_corners=True)
        x_in_list.append(x_tmp)

    for z_idx in range(len(z_rec)):
        z_rec[z_idx] = z_rec[z_idx].cuda(args.gpu, non_blocking=True)


    final_stage=args.num_scale
    
    out_size=int(np.round(args.size_list[final_stage]*2))

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
    transform = transforms.Compose([transforms.Resize((out_size, out_size)),
                                             transforms.ToTensor(),
                                             normalize])
    img = F.interpolate(x_org, (out_size, out_size), mode='bilinear', align_corners=True)
    vutils.save_image(img.detach().cpu(), os.path.join(args.res_dir, 'Final_stage_Input_{}.png'.format(final_stage)),
                        nrow=1, normalize=True)
    G = networks[1]
    G.eval()

    with torch.no_grad():
        G.current_scale -= 1
        x_rec_list = G(z_rec)

        # calculate rmse for each scale
        rmse_list = [1.0]
        for rmseidx in range(1, args.stage):
            rmse = torch.sqrt(F.mse_loss(x_rec_list[rmseidx], x_in_list[rmseidx]))
            rmse /= 100.0
            rmse_list.append(rmse)
        if len(rmse_list) > 1:
            rmse_list[-1] = 0.0

        vutils.save_image(x_rec_list[-1].detach().cpu(), os.path.join(args.res_dir, 'REC_{}.png'.format(args.stage)),
                          nrow=1, normalize=True)

        out=img
        for i_sup in range(10):
            noisz = torch.randn(args.batch_size, 3, out_size, out_size).cuda(args.gpu, non_blocking=True)
            input=0.04*noisz+out
            x_rec_list = G.SuperResolution(input,final_stage)
            vutils.save_image(x_rec_list[-1].detach().cpu(), os.path.join(args.res_dir, 'output-epol{}.png'.format(i_sup)),
                            nrow=1, normalize=True)
            out=x_rec_list
    return

if __name__ == '__main__':
    main()
