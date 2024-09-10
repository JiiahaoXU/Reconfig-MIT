#!/usr/bin/python3

import argparse
import os
from trainer import Nice_Trainer, P2p_Trainer, Munit_Trainer, Unit_Trainer, Reconfig_MIT_Trainer
import yaml
import gralpruning
import warnings

warnings.filterwarnings("ignore")


# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--model', type=str, default='DeCycleGan')

    parser.add_argument('--bidirect', action='store_true')
    parser.add_argument('--regist', action='store_true')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--decouple', action='store_true')
    parser.add_argument('--r_ccl', action='store_true')
    parser.add_argument('--few_shot', action='store_true')

    parser.add_argument('--noise_level', type=int, default=0)
    parser.add_argument('--Adv_lamda', type=int, default=1)
    parser.add_argument('--Cyc_lamda', type=int, default=10)
    parser.add_argument('--EL_lamda', type=int, default=10)
    parser.add_argument('--Corr_lamda', type=int, default=20)
    parser.add_argument('--Smooth_lamda', type=int, default=10)
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--n_epochs', type=int, default=80)
    parser.add_argument('--batchSize', type=int, default=1)
    parser.add_argument('--eva_epoch', type=int, default=10)
    parser.add_argument('--data_ratio', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=0.0001)

    parser.add_argument('--decay_epoch', type=int, default=20)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--input_nc', type=int, default=1)
    parser.add_argument('--output_nc', type=int, default=1)
    parser.add_argument('--n_cpu', type=int, default=4)

    parser.add_argument('--decouple_target', type=float, default=0.4)
    parser.add_argument('--decouple_every', type=int, default=4)
    parser.add_argument('--length_factor', type=int, default=3)

    parser.add_argument('--warmup_epoch', type=int, default=-1, help='The pruning rate / death rate.')
    parser.add_argument('--final-grow-epoch', type=int, default=-1, help='The pruning rate / death rate.')
    gralpruning.add_sparse_args(parser)

    args = parser.parse_args()
    # print(args.bidirect, args.regist)
    args.init_grow_epoch = args.warmup_epoch + 1
    if args.final_grow_epoch == -1:
        args.final_grow_epoch = args.warmup_epoch * 2 + 1

    args.model_name = 'Cyc'

    if args.few_shot:
        args.dataroot = './data/train2D_100shot/'
        args.model_name += '_fewshot'
    else:
        args.dataroot = './data/train2D/'

    args.val_dataroot = './data/val2D/'

    args.model_name += '_DR(%.1f)_NL(%d)' % (args.data_ratio, args.noise_level)
    if args.regist:
        args.model_name += '_Reg'
    if args.r_ccl:
        args.model_name += '_EL'
    if args.reconfig_mit:
        args.model_name += '_Ini(%.2f)_Fin(%.2f)_Wr(%d)_Fg(%d)' % (args.init_density, args.final_density, args.warmup_epoch, args.final_grow_epoch)


    args.save_root = './results/%s/' % args.model_name
    args.log_root = args.save_root + 'log'
    args.image_save = './results/%s/img/' % args.model_name

    print('init begins')

    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)
    if not os.path.exists(args.save_root + 'checkpoint'):
        os.makedirs(args.save_root + 'checkpoint')
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    # if args.model == 'CycleGan':
    #     trainer = Cyc_Trainer(args)
    elif args.model == 'Munit':
        trainer = Munit_Trainer(args)
    elif args.model == 'Unit':
        trainer = Unit_Trainer(args)
    elif args.model == 'NiceGAN':
        trainer = Nice_Trainer(args)
    # elif args.model == 'U-gat':
    #     trainer = Ugat_Trainer(args)
    elif args.model == 'P2p':
        trainer = P2p_Trainer(args)
    elif args.model == 'Reconfig_MIT':
        trainer = Reconfig_MIT_Trainer(args)

    print('init finished')

    trainer.train()


###################################
if __name__ == '__main__':
    main()
