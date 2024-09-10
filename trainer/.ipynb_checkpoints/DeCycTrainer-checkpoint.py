#!/usr/bin/python3

import argparse
import itertools
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from .utils import LambdaLR, Logger, ReplayBuffer
from .utils import weights_init_normal
from .utils import DecoupleAdaptive
from .datasets import ImageDataset, ValDataset
from Model.CycleGan import *
from .utils import Resize, ToTensor, smooothing_loss
from .utils import Logger, write_loss_log
from .reg import Reg
from torchvision.transforms import RandomAffine, ToPILImage
from .transformer import Transformer_2D
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import numpy as np
import cv2
import random
from gralpruning import Masking, CosineDecay, LinearDecay


class DeCyc_Trainer():
    def __init__(self, args):
        super().__init__()
        self.args = args
        ## def networks
        self.netG_A2B = Generator(args.input_nc, args.output_nc).cuda()
        # print(self.netG_A2B)
        self.netD_B = Discriminator(args.input_nc).cuda()
        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=args.lr, betas=(0.5, 0.999))

        if args.regist:
            self.R_A = Reg(args.size, args.size, args.input_nc, args.input_nc).cuda()
            self.spatial_transform = Transformer_2D().cuda()
            self.optimizer_R_A = torch.optim.Adam(self.R_A.parameters(), lr=args.lr, betas=(0.5, 0.999))
        if args.bidirect:
            self.netG_B2A = Generator(args.input_nc, args.output_nc).cuda()
            self.netD_A = Discriminator(args.input_nc).cuda()
            self.optimizer_G_A2B = torch.optim.Adam(self.netG_A2B.parameters(),
                                                    lr=args.lr, betas=(0.5, 0.999))
            self.optimizer_G_B2A = torch.optim.Adam(self.netG_B2A.parameters(),
                                                    lr=args.lr, betas=(0.5, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=args.lr, betas=(0.5, 0.999))

        else:
            self.optimizer_G = torch.optim.Adam(self.netG_A2B.parameters(), lr=args.lr, betas=(0.5, 0.999))

        # Lossess
        self.MSE_loss = torch.nn.MSELoss()
        self.L1_loss = torch.nn.L1Loss()

        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if args.cuda else torch.Tensor
        self.input_A = Tensor(args.batchSize, args.input_nc, args.size, args.size)
        self.input_B = Tensor(args.batchSize, args.output_nc, args.size, args.size)
        self.target_real = Variable(Tensor(1, 1).fill_(1.0), requires_grad=False)
        self.target_fake = Variable(Tensor(1, 1).fill_(0.0), requires_grad=False)

        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

        # Dataset loader
        level = args.noise_level  # set noise level

        transforms_1 = [ToPILImage(),
                        RandomAffine(degrees=level, translate=[0.02 * level, 0.02 * level],
                                     scale=[1 - 0.02 * level, 1 + 0.02 * level], fill=-1),
                        ToTensor(),
                        Resize(size_tuple=(args.size, args.size))]

        transforms_2 = [ToPILImage(),
                        ToTensor(),
                        Resize(size_tuple=(args.size, args.size))]

        train_dataset = ImageDataset(args.dataroot, level, transforms_1=transforms_1, transforms_2=transforms_2,
                                     unaligned=False, )
        subset = torch.utils.data.Subset(train_dataset, np.arange(int(len(train_dataset) * self.args.data_ratio)))

        self.dataloader = DataLoader(subset, batch_size=args.batchSize, shuffle=True, num_workers=args.n_cpu)

        val_transforms = [ToTensor(),
                          Resize(size_tuple=(args.size, args.size))]

        self.val_data = DataLoader(ValDataset(args.val_dataroot, transforms_=val_transforms, unaligned=False),
                                   batch_size=args.batchSize, shuffle=False, num_workers=args.n_cpu)

        # Loss plot
        self.logger = Logger(args.n_epochs, len(self.dataloader),
                             args.decouple, args.regist, args.model_name)

        self.eva_flag = {0.1: 0,
                         1.0: 0}

        self.decouple = args.decouple
        # self.decouple_p = args.decouple_p
        self.decouple_target = args.decouple_target
        self.decouple_length = args.n_epochs * len(self.dataloader) / args.length_factor
        self.decouple_every = args.decouple_every

        # self.device = "cuda"

        # self.decouple_adaptive_G = DecoupleAdaptive(self.decouple_target, self.decouple_length, self.decouple_every)
        # self.decouple_adaptive_D = DecoupleAdaptive(self.decouple_target, self.decouple_length, self.decouple_every)
        # self.decouple_rate_G = 0.0
        # self.decouple_rate_D = 0.0
        # self.r_t_stat = 0.0
        # self.loss_mean = 0.0
        # self.update_count = 0
        self.SR_loss = torch.tensor(0.0)
        self.SM_loss = torch.tensor(0.0)
        self.extended_loss = torch.tensor(0.0)
        self.ex_adv_loss = torch.tensor(0.0)

        self.mask_D_A = None
        self.mask_D_B = None
        self.mask_G_A2B = None
        self.mask_G_B2A = None

        if self.args.sparse:
            decay = CosineDecay(self.args.prune_rate, len(self.dataloader) * self.args.n_epochs)
            self.mask_G_A2B = Masking(self.optimizer_G_A2B, prune_rate=self.args.prune_rate, death_mode=self.args.prune,
                                      prune_rate_decay=decay, growth_mode=self.args.growth,
                                      redistribution_mode=self.args.redistribution, args=args,
                                      train_loader=self.dataloader)
            self.mask_G_A2B.add_module(self.netG_A2B, sparse_init=self.args.sparse_init)

            self.mask_G_B2A = Masking(self.optimizer_G_B2A, prune_rate=self.args.prune_rate, death_mode=self.args.prune,
                                      prune_rate_decay=decay, growth_mode=self.args.growth,
                                      redistribution_mode=self.args.redistribution, args=args,
                                      train_loader=self.dataloader)
            self.mask_G_B2A.add_module(self.netG_B2A, sparse_init=self.args.sparse_init)

    def train(self):
        ###### Training ######
        print('Training begins')
        for epoch in range(self.args.epoch + 1, self.args.n_epochs + 1):
            if epoch == self.args.init_grow_epoch and self.args.sparse:
                if self.args.regrow:
                    self.mask_G_A2B.pruning_stage = False
                    self.mask_G_B2A.pruning_stage = False
                    print('Turn pruning stage to regrow stage!!')
            if epoch == self.args.final_grow_epoch:
                if self.args.turn_lr_fine_tune:
                    for params in self.optimizer_G_A2B.param_groups:
                        params['lr'] = self.args.lr * 0.1
                    for params in self.optimizer_G_B2A.param_groups:
                        params['lr'] = self.args.lr * 0.1
            for i, batch in enumerate(self.dataloader):
                # Set model input
                real_A = Variable(self.input_A.copy_(batch['A']))
                real_B = Variable(self.input_B.copy_(batch['B']))
                if self.args.bidirect:  # C dir
                    if self.args.regist:
                        self.optimizer_R_A.zero_grad()
                    self.optimizer_G_A2B.zero_grad()
                    self.optimizer_G_B2A.zero_grad()
                    # GAN loss
                    fake_B = self.netG_A2B(real_A)
                    pred_fake = self.netD_B(fake_B)
                    loss_GAN_A2B = self.args.Adv_lamda * self.MSE_loss(pred_fake, self.target_real)

                    fake_A = self.netG_B2A(real_B)
                    pred_fake = self.netD_A(fake_A)
                    loss_GAN_B2A = self.args.Adv_lamda * self.MSE_loss(pred_fake, self.target_real)

                    # Cycle loss
                    recovered_A = self.netG_B2A(fake_B)
                    loss_cycle_ABA = self.args.Cyc_lamda * self.L1_loss(recovered_A, real_A)

                    recovered_B = self.netG_A2B(fake_A)
                    loss_cycle_BAB = self.args.Cyc_lamda * self.L1_loss(recovered_B, real_B)

                    # Total loss
                    loss_Total = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB

                    if self.args.regist:
                        Trans = self.R_A(fake_B, real_B)
                        SysRegist_A2B = self.spatial_transform(fake_B, Trans)
                        self.SR_loss = self.args.Corr_lamda * self.L1_loss(SysRegist_A2B, real_B)  ###SR
                        self.SM_loss = self.args.Smooth_lamda * smooothing_loss(Trans)

                        loss_Total += self.SR_loss + self.SM_loss

                    if self.args.extended_loss:
                        extended_A_B = self.netG_A2B(recovered_A)
                        extended_A_A = self.netG_B2A(extended_A_B)
                        # extended_A_B_detach = self.netG_A2B(recovered_A.detach())
                        # pred_fake = self.netD_B(extended_A_B_detach)
                        # loss_GAN_A2B_extended = self.args.Adv_lamda * self.MSE_loss(pred_fake, self.target_real)

                        loss_ex_cycle_A_1 = self.L1_loss(recovered_A.detach(), extended_A_A)
                        # loss_ex_cycle_A_2 = self.L1_loss(real_A, extended_A_A)
                        loss_ex_cycle_A_3 = self.L1_loss(fake_B.detach(), extended_A_B)

                        # extended_B_A = self.netG_B2A(recovered_B)
                        # extended_B_B = self.netG_A2B(extended_B_A)
                        # # extended_B_A_detach = self.netG_A2B(recovered_B.detach())
                        # # pred_fake = self.netD_B(extended_B_A_detach)
                        # # loss_GAN_B2A_extended = self.args.Adv_lamda * self.MSE_loss(pred_fake, self.target_real)
                        # #
                        # loss_ex_cycle_B_1 = self.L1_loss(recovered_B.detach(), extended_B_B)
                        # loss_ex_cycle_B_2 = self.L1_loss(real_B, extended_B_B)
                        # loss_ex_cycle_B_3 = self.L1_loss(fake_A.detach(), extended_B_A)

                        # self.ex_adv_loss = loss_GAN_B2A_extended + loss_GAN_A2B_extended

                        self.extended_loss = self.args.EL_lamda * \
                                             (loss_ex_cycle_A_1 + loss_ex_cycle_A_3)

                        # loss_Total += self.ex_adv_loss
                        loss_Total += self.extended_loss

                    loss_Total.backward()

                    if self.mask_G_A2B is not None:
                        self.mask_G_A2B.step()
                        self.mask_G_B2A.step()
                    else:
                        self.optimizer_G_A2B.step()
                        self.optimizer_G_B2A.step()

                    if self.args.regist:
                        self.optimizer_R_A.step()

                    ###### Discriminator A ######
                    self.optimizer_D_A.zero_grad()
                    # Real loss
                    pred_real = self.netD_A(real_A)
                    loss_D_real = self.args.Adv_lamda * self.MSE_loss(pred_real, self.target_real)
                    # Fake loss
                    fake_A = self.fake_A_buffer.push_and_pop(fake_A)
                    pred_fake = self.netD_A(fake_A.detach())
                    loss_D_fake = self.args.Adv_lamda * self.MSE_loss(pred_fake, self.target_fake)

                    # Total loss
                    loss_D_A = (loss_D_real + loss_D_fake)
                    loss_D_A.backward()

                    self.optimizer_D_A.step()
                    ###################################

                    ###### Discriminator B ######
                    self.optimizer_D_B.zero_grad()

                    # Real loss
                    pred_real = self.netD_B(real_B)
                    loss_D_real = self.args.Adv_lamda * self.MSE_loss(pred_real, self.target_real)

                    # Fake loss
                    fake_B = self.fake_B_buffer.push_and_pop(fake_B)
                    pred_fake = self.netD_B(fake_B.detach())
                    loss_D_fake = self.args.Adv_lamda * self.MSE_loss(pred_fake, self.target_fake)

                    # Total loss
                    loss_D_B = (loss_D_real + loss_D_fake)
                    loss_D_B.backward()

                    self.optimizer_D_B.step()
                    ###################################

                else:  # s dir :NC
                    if self.args.regist:  # NC+R
                        self.optimizer_R_A.zero_grad()
                        self.optimizer_G.zero_grad()
                        #### regist sys loss
                        fake_B = self.netG_A2B(real_A)
                        Trans = self.R_A(fake_B, real_B)
                        SysRegist_A2B = self.spatial_transform(fake_B, Trans)
                        self.SR_loss = self.args.Corr_lamda * self.L1_loss(SysRegist_A2B, real_B)  ###SR
                        pred_fake0 = self.netD_B(fake_B)
                        adv_loss = self.args.Adv_lamda * self.MSE_loss(pred_fake0, self.target_real)
                        ####smooth loss
                        self.SM_loss = self.args.Smooth_lamda * smooothing_loss(Trans)
                        toal_loss = self.SM_loss + adv_loss + self.SR_loss
                        toal_loss.backward()
                        self.optimizer_R_A.step()
                        self.optimizer_G.step()
                        self.optimizer_D_B.zero_grad()
                        with torch.no_grad():
                            fake_B = self.netG_A2B(real_A)
                        pred_fake0 = self.netD_B(fake_B)
                        pred_real = self.netD_B(real_B)
                        loss_D_B = self.args.Adv_lamda * self.MSE_loss(pred_fake0, self.target_fake) + \
                                   self.args.Adv_lamda * self.MSE_loss(pred_real, self.target_real)

                        loss_D_B.backward()
                        self.optimizer_D_B.step()



                    else:  # only NC
                        self.optimizer_G.zero_grad()
                        fake_B = self.netG_A2B(real_A)
                        #### GAN aligin loss
                        pred_fake = self.netD_B(fake_B)
                        adv_loss = self.args.Adv_lamda * self.MSE_loss(pred_fake, self.target_real)
                        adv_loss.backward()
                        self.optimizer_G.step()
                        ###### Discriminator B ######
                        self.optimizer_D_B.zero_grad()
                        # Real loss
                        pred_real = self.netD_B(real_B)
                        loss_D_real = self.args.Adv_lamda * self.MSE_loss(pred_real, self.target_real)
                        # Fake loss
                        fake_B = self.fake_B_buffer.push_and_pop(fake_B)
                        pred_fake = self.netD_B(fake_B.detach())
                        loss_D_fake = self.args.Adv_lamda * self.MSE_loss(pred_fake, self.target_fake)
                        # Total loss
                        loss_D_B = (loss_D_real + loss_D_fake)
                        loss_D_B.backward()
                        self.optimizer_D_B.step()
                        ###################################

                loss_dict = {'loss_D_A': loss_D_A, 'loss_D_B': loss_D_B
                             }
                if self.args.regist:
                    loss_dict['SR_loss'] = self.SR_loss
                    loss_dict['SM_loss'] = self.SM_loss
                if self.args.extended_loss:
                    loss_dict['ex_loss'] = self.extended_loss
                    # loss_dict['ex_adv_loss'] = self.ex_adv_loss
                self.logger.log(loss_dict, self.args.log_root)

                # write_loss_log(loss_dict,
                #                self.args.save_root + 'result')

            # torch.save(self.netG_A2B.state_dict(),  + 'netG_A2B.pth')
            # torch.save(self.netG_B2A.state_dict(), self.args.save_root + 'netG_B2A.pth')
            # torch.save(self.netD_A.state_dict(), self.args.save_root + 'netD_A.pth')
            # torch.save(self.netD_B.state_dict(), self.args.save_root + 'netD_B.pth')
            # torch.save(self.R_A.state_dict(), self.args.save_root + 'Regist.pth')
            # torch.save(netD_A.state_dict(), 'output/netD_A_3D.pth')
            # torch.save(netD_B.state_dict(), 'output/netD_B_3D.pth')

            # #############val###############
            # with torch.no_grad():
            #     MAE = 0
            #     num = 0
            #     for i, batch in enumerate(self.val_data):
            #         real_A = Variable(self.input_A.copy_(batchA))
            #         real_B = Variable(self.input_B.copy_(batchB)).detach().cpu().numpy().squeeze()
            #         fake_B = self.netG_A2B(real_A).detach().cpu().numpy().squeeze()
            #         mae = self.MAE(fake_B, real_B)
            #         MAE += mae
            #         num += 1
            #     val_mae = MAE / num
            #     print('Val MAE:', val_mae)

            if epoch % self.args.eva_epoch == 0 and epoch >= self.eva_flag[self.args.data_ratio]:
                self.netG_A2B.eval()
                with torch.no_grad():
                    MAE = 0
                    PSNR = 0
                    SSIM = 0
                    num = 0
                    for i, batch in enumerate(self.val_data):
                        real_A = Variable(self.input_A.copy_(batch['A']))
                        real_B = Variable(self.input_B.copy_(batch['B'])).detach().cpu().numpy().squeeze()

                        fake_B = self.netG_A2B(real_A)
                        fake_B = fake_B.detach().cpu().numpy().squeeze()
                        mae = self.MAE(fake_B, real_B)
                        psnr = self.PSNR(fake_B, real_B)
                        ssim = compare_ssim(fake_B, real_B)
                        MAE += mae
                        PSNR += psnr
                        SSIM += ssim
                        num += 1
                    test_mae = MAE / num
                    test_psnr = PSNR / num
                    test_ssim = SSIM / num
                    print("Test MAE: %.5f, Test PSNR: %.5f, Test SSIM: %.5f" % (test_mae, test_psnr, test_ssim))
                self.netG_A2B.train()

                # torch.save(
                #     {
                #         "netG_A2B": self.netG_A2B.state_dict(),
                #         "netG_B2A": self.netG_B2A.state_dict(),
                #         "R_A": self.R_A.state_dict(),
                #         # "netD_A": self.netD_A.state_dict(),
                #         # "netD_B": self.netD_B.state_dict()
                #     },
                #     '%s/%s.pt' % (self.args.save_root + 'checkpoint', str(epoch).zfill(3)),
                # )

                with open('./%s/score_record.txt' % self.args.log_root, 'a') as f:

                    f.write("epoch: %02d" % epoch + ": ")
                    f.write("Test MAE: %.5f, Test PSNR: %.5f, Test SSIM: %.5f" %
                            (test_mae, test_psnr, test_ssim) + '\n')

    def test(self, ):
        self.netG_A2B.load_state_dict(torch.load(self.args.save_root + 'checkpoint' + '/350.pt')['netG_A2B'])
        # self.R_A.load_state_dict(torch.load(self.args.save_root + 'Regist.pth'))
        # with torch.no_grad():
        #     MAE = 0
        #     PSNR = 0
        #     SSIM = 0
        #     num = 0
        #     for i, batch in enumerate(self.val_data):
        #         real_A = Variable(self.input_A.copy_(batch['A']))
        #         real_B = Variable(self.input_B.copy_(batch['B'])).detach().cpu().numpy().squeeze()
        #
        #         fake_B = self.netG_A2B(real_A)
        #         fake_B = fake_B.detach().cpu().numpy().squeeze()
        #         mae = self.MAE(fake_B, real_B)
        #         psnr = self.PSNR(fake_B, real_B)
        #         ssim = compare_ssim(fake_B, real_B)
        #         MAE += mae
        #         PSNR += psnr
        #         SSIM += ssim
        #         num += 1
        #     print('MAE:', MAE / num)
        #     print('PSNR:', PSNR / num)
        #     print('SSIM:', SSIM / num)

    def PSNR(self, fake, real):
        x, y = np.where(real != -1)  # Exclude background
        mse = np.mean(((fake[x][y] + 1) / 2. - (real[x][y] + 1) / 2.) ** 2)
        if mse < 1.0e-10:
            return 100
        else:
            PIXEL_MAX = 1
            return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

    def MAE(self, fake, real):
        x, y = np.where(real != -1)  # Exclude background
        mae = np.abs(fake[x, y] - real[x, y]).mean()
        return mae / 2  # from (-1,1) normaliz  to (0,1)

    def save_deformation(self, defms, root):
        heatmapshow = None
        defms_ = defms.data.cpu().float().numpy()
        dir_x = defms_[0]
        dir_y = defms_[1]
        x_max, x_min = dir_x.max(), dir_x.min()
        y_max, y_min = dir_y.max(), dir_y.min()
        dir_x = ((dir_x - x_min) / (x_max - x_min)) * 255
        dir_y = ((dir_y - y_min) / (y_max - y_min)) * 255
        tans_x = cv2.normalize(dir_x, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        tans_x[tans_x <= 150] = 0
        tans_x = cv2.applyColorMap(tans_x, cv2.COLORMAP_JET)
        tans_y = cv2.normalize(dir_y, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        tans_y[tans_y <= 150] = 0
        tans_y = cv2.applyColorMap(tans_y, cv2.COLORMAP_JET)
        gradxy = cv2.addWeighted(tans_x, 0.5, tans_y, 0.5, 0)

        cv2.imwrite(root, gradxy)
