from __future__ import print_function
import argparse
import os
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

from data_loader import get_loader
from utils import plot_feats, read_binary_file
from models import define_netD, define_netG


def train(netD, netG, data_loader, opt):
    
    real_label = 1
    fake_label = 0

    # cost criterion
    #criterion = nn.BCELoss() # normal gan 
    criterion = nn.MSELoss() # lsgan
    # get the device
    device = torch.device("cuda:0" if opt.cuda else "cpu")
    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    for epoch in range(opt.niter):
        for i, data in enumerate(data_loader, 0):
            # discard the data if the size less than opt.mgcDim frames
            if data[0].size(-1) <= opt.mgcDim:
                continue
            real_data, pred_data = data

            #################################
            # (1) Updata D network: maximize log(D(x)) + log(1 - D(G(z)))
            #################################
            # train with real 
            # clear the gradient buffers
            netD.zero_grad()
            real_cpu = real_data.to(device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label, device=device)
            output = netD(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(real_data.size(), device=device)
            pred_cpu = pred_data.to(device)
            fake = netG(noise, pred_cpu)
            # add the residual to the tts predicted data 
            fake = fake + pred_cpu
            label.fill_(fake_label)
            # crop the tensor to fixed size
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            # update the discriminator on mini batch
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ############################    
            netG.zero_grad()
            label.fill_(real_label) # fake labels are real for generator cost
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                %(epoch, opt.niter, i, len(data_loader), 
            errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # fake = netG(noise, pred_data)
            # fake = fake + pred_data
            # fake = fake.data.cpu().numpy()
            # fake = fake.reshape(opt.mgcDim, -1)
            # fake = fake[:,rand_int:rand_int+60]
                
            # pred = pred_data.data.cpu().numpy()
            # pred = pred.reshape(opt.mgcDim, -1)
            # pred = pred[:,rand_int:rand_int+60]
                
            # real = real_data.cpu().numpy()
            # real = real.reshape(opt.mgcDim, -1)
            # real = real[:,rand_int:rand_int+60]
            # plot_feats(real, pred, fake,  epoch, i, opt.outf)
            
            # batch_data = []
            
            del errD_fake, errD_real, errG, real_cpu , pred_cpu
            del noise, fake, output, errD
            torch.cuda.empty_cache()

        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' %(opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' %(opt.outf, epoch))

def test(netG, opt):
    assert opt.netG != ''
    test_dir = opt.testdata_dir
    for f in os.listdir(test_dir):
        fname, ext = os.path.splitext(f)
        if ext == '.cmp':
            print(fname)
            cmp_file = os.path.join(test_dir, f)
            ac_data = read_binary_file(cmp_file, dim=47)
            ac_data = torch.FloatTensor(ac_data)
            noise = torch.FloatTensor(ac_data.size(0), nz)
            if opt.cuda:
                ac_data, noise = ac_data.cuda(), noise.cuda()
            ac_data = Variable(ac_data)
            noise = Variable(noise)
            noise.data.normal_(0, 1)
            generated_pulses = netG(noise, ac_data)
            generated_pulses = generated_pulses.data.cpu().numpy()
            generated_pulses = generated_pulses.reshape(ac_data.size(0), -1)
            out_file = os.path.join(test_dir, fname + '.pls')
            with open(out_file, 'wb') as fid:
                generated_pulses.tofile(fid)    


def parse_args():
    # parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--voiceName', required=True, help='nick | jenny ')
    parser.add_argument('--mode', required=True, type=str, help='train | test')
    parser.add_argument('--xFilesList', required=True, help='path to input files list')
    parser.add_argument('--yFilesList', required=True, help='path to output files list')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--mgcDim', type=int, default=60, help='mel-cepstrum dimension')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0001')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--testdata_dir', type=str, help='path to test data')
    opt = parser.parse_args()
    return opt

def main():
    opt = parse_args()
    print(opt)
    # if manual seed is not provide then pick one randomly
    if opt.manualSeed is None:
        opt.manualSeed  = random.randint(1, 10000)
    print('Random Seed: ', opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    # check wether cuda is available
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)
    cudnn.enabled = True
    cudnn.benchmark = False 

    # prepare the output directories
    try:
        os.makedirs(opt.outf)
        os.makedirs(os.path.join(opt.outf, 'figures'))
    except OSError:
        pass

    # prepare the data loader
    x_files_list_file = opt.xFilesList
    y_files_list_file = opt.yFilesList
    in_dim = opt.mgcDim
    out_dim = opt.mgcDim

    with open(x_files_list_file, 'r') as fid:
        x_files_list = [l.strip() for l in fid.readlines()]

    with open(y_files_list_file, 'r') as fid:
        y_files_list = [l.strip() for l in fid.readlines()]
    
    data_loader = get_loader(x_files_list, y_files_list, 
                            in_dim, out_dim, opt.batchSize, False, 0)
    
    # get the device
    device = torch.device("cuda:0" if opt.cuda else "cpu")
    # define the generator 
    netG = define_netG(in_ch=2,device=device)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    # define the discriminator
    netD = define_netD(device=device)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    if opt.mode == 'train':
        train(netD, netG, data_loader, opt)
    elif opt.mode == 'test':
        test(netG, opt)
    else:
        print('Mode must be either train or test only')

if __name__ == "__main__":
    main()
