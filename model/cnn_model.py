import os
import torch
import itertools
from torch import nn, optim
from torch.nn.parallel import DataParallel
from .networks import CnnEncoder, LinearDecoder
from utils import f1_loss


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

        if self.gpu_ids:
            self.cnn_encoder = DataParallel(self.cnn_encoder, self.gpu_ids)
            self.linear_decoder = DataParallel(self.linear_decoder, self.gpu_ids)

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
        self.image_id = None

        self.training_loss = 0
        self.test_loss = 0
        self.f1_scores = 0

    def feed_input(self, x: dict):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        :param x: include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        self.image = x['image'].to(self.device)
        self.label_original = x['label'].to(self.device)
        self.image_id = x['image_id']

    def optimize_parameters(self):
        """
        Optimizes parameters
        """
        # Forward
        self.forward()
        self.optimizer.zero_grad()
        # print(self.label_pred)
        # print(self.label_original)
        loss = self.criterion(self.label_pred, self.label_original)
        loss.backward()
        train_loss = loss.item()
        self.training_loss += train_loss
        # print("Train Loss", train_loss)
        self.optimizer.step()

    def forward(self):
        """Run forward pass
        Called by both functions <optimize_parameters> and <test>
        """
        feature_vec = self.cnn_encoder(self.image)
        self.label_pred = self.linear_decoder(feature_vec)

    def train(self):
        self.cnn_encoder.train()
        self.linear_decoder.train()

    def eval(self):
        self.cnn_encoder.eval()
        self.linear_decoder.eval()

    def get_last_avg_train_loss(self):
        loss = self.training_loss / self.opt.print_freq
        self.training_loss = 0
        return loss

    def _single_test(self, data):
        self.feed_input(data)
        self.forward()
        test_loss = self.criterion(self.label_pred, self.label_original).item()
        # print("Test Loss", test_loss)
        self.test_loss += test_loss
        self.f1_scores += f1_loss(self.label_original, self.label_pred).item()

    def run_test_on_training(self, testloader) -> tuple:
        with torch.no_grad():
            no = 0
            for i, data in enumerate(testloader):
                self._single_test(data)
                no = i + 1
        # print("Length of testloader", no)
        test_loss = self.test_loss / no
        f1_score = self.f1_scores / no
        self.test_loss = 0
        self.f1_scores = 0

        return test_loss, f1_score

    def save_networks(self, epoch: str) -> None:
        """Save models
        :param epoch: Current Epoch (prefix for the name)
        """
        self.save_network(self.cnn_encoder, 'cnn_encoder', epoch)
        self.save_network(self.linear_decoder, 'linear_decoder', epoch)

    def save_network(self, net, net_name, epoch):
        save_filename = f'{epoch}_net_{net_name}.pt'
        save_path = os.path.join(self.save_dir, save_filename)

        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(net.module.cpu().state_dict(), save_path)
            net.cuda(self.gpu_ids[0])
        else:
            torch.save(net.cpu().state_dict(), save_path)

    def save_optimizer(self, epoch):
        save_path = os.path.join(self.save_dir, f"{epoch}_optimizer.pt")

        torch.save(self.optimizer.state_dict(), save_path)

    def load_networks(self, model_name: str, load_optim: bool = False):
        self._load_object('cnn_encoder', f"{model_name}_net_cnn_encoder.pt")
        self._load_object('linear_decoder', f"{model_name}_net_linear_decoder.pt")

        if load_optim:
            self._load_object('optimizer', f"{model_name}_optimizer.pt")

    def _load_object(self, object_name:str, model_name:str):
        state_dict = torch.load(model_name, map_location=self.device)

        net = getattr(self, object_name)
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        net.load_state_dict(state_dict)
