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
        self.cnn_encoder = CnnEncoder(opt.nf, opt.res_blocks3x3, use_dropout=not opt.no_dropout).to(self.device)
        self.linear_decoder = LinearDecoder(self.encoder_out, opt.output_n).to(self.device)

        # Loss
        self.criterion = nn.NLLLoss()

        # Optimizer
        self.optimizer = optim.Adam(itertools.chain(self.cnn_encoder.parameters(), self.linear_decoder.parameters()),
                                    lr=opt.lr)

        print(self.cnn_encoder)
        print(self.linear_decoder)

        self.image = None
        self.label_original = None
        self.label_pred = None

    def feed_input(self, x: dict):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        :param x: include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        self.image = x['image'].to(self.device)
        self.label_original = x['label'].to(self.device)

    def optimize_parameters(self):
        """
        Optimizes parameters
        """
        # Forward
        self.forward()
        self.optimizer.zero_grad()
        self.criterion(self.label_pred, self.label_original)
        self.optimizer.step()

    def forward(self):
        """Run forward pass
        Called by both functions <optimize_parameters> and <test>
        """
        feature_vec = self.cnn_encoder(self.image)
        self.label_pred = self.linear_decoder(feature_vec)
