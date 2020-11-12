import os
import time
import torch
import argparse
from utils import mkdirs


class BaseOptions:
    def __init__(self):
        self.isTrain = False
        self.parser = argparse.ArgumentParser(prog='Snakes in the Hood',
                                              description='A very deep application that classifies Snakes')
        self.parser = self.initialized(self.parser)

    def initialized(self, parser):
        """
        :param parser:
        :return:
        """
        # basic parameters
        parser.add_argument('--job-dir', dest="checkpoints_dir", type=str, default='./checkpoints',
                            help='models are saved here')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--name', type=str, default='uwie',
                            help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--name_time', action='store_true', help='Add Timestamp after name')
        parser.add_argument('--dataroot', required=True, type=str,
                            help="path to images (should have sub folders trainA, trainB, valA, valB, etc)")
        parser.add_argument('--no_gpu', action='store_true', help='Use only CPU')

        # model parameters
        parser.add_argument('--input_nc', default=3, type=int,
                            help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--output_n', type=int, default=3, help='# of output classes')
        parser.add_argument('--nf', type=int, default=64, help='# of filters in the last conv layer')
        parser.add_argument('--n_blocks', type=int, default=6, help='no of Resnet Blocks in Classifier')
        parser.add_argument('--norm', type=str, default='instance',
                            help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--padding_type', type=str, default='reflect',
                            help='the name of padding layer: reflect | replicate | zero')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')

        # dataset parameters
        parser.add_argument('--max_dataset_size', default=float('inf'), type=float,
                            help='Maximum number of samples allowed per dataset. If the dataset directory contains '
                                 'more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--preprocess', type=str, default='resize_and_crop',
                            help='scaling and cropping of images at load time [resize_and_crop | crop | none]')
        parser.add_argument('--no_flip', action='store_true',
                            help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
        parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--num_threads', default=0, type=int, help='# threads for loading data')

        return parser

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        opt = self.parser.parse_args()
        opt.isTrain = self.isTrain

        # Add Time stamp in the name
        if self.isTrain and opt.name_time:
            opt.name = opt.name + str(int(time.time()))

        # Setup Continue training, if continue training, change the epoch_count with the value of ct
        if self.isTrain and opt.ct > 0:
            opt.epoch_count = opt.ct

        # set gpu ids
        if not opt.no_gpu and torch.cuda.is_available():
            device_ids = list(range(torch.cuda.device_count()))
            gpus = len(device_ids)
            print(f'{gpus} no of GPUs detected. Using GPU: {str(device_ids)}')
            torch.cuda.set_device(device_ids[0])
        else:
            device_ids = []
            print('No GPU. switching to CPU')
        opt.gpu_ids = device_ids

        self.print_options(opt)

        return opt
