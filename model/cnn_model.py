import os
import torch
import itertools
from torch import nn, optim
from .networks import CnnEncoder, LinearDecoder


class CnnModel:
    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device(f'cuda:{self.gpu_ids[0]}') if self.gpu_ids else torch.device('cpu')

        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.encoder_out = opt.nf * (2 ** opt.res_blocks3x3) * 2

        # Models
        self.cnn_encoder = CnnEncoder(opt.nf, opt.res_blocks3x3, use_dropout=not opt.no_dropout)
        self.linear_decoder = LinearDecoder(self.encoder_out, opt.output_n)

        # Loss
        self.criterion = nn.NLLLoss()

        # Optimizer
        self.optimizer = optim.Adam(itertools.chain(self.cnn_encoder.parameters(), self.linear_decoder.parameters()),
                                    lr=opt.lr)

        print(self.cnn_encoder)
        print(self.linear_decoder)
