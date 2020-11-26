import os
import torch
import itertools
from torch import nn, optim
from .networks import LinearDecoder
from torch.nn.parallel import DataParallel
from utils import f1_loss
from torchvision.models import resnet50


class ResnetModel:
    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device(f'cuda:{self.gpu_ids[0]}') if self.gpu_ids else torch.device('cpu')

        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

        self.image = None
        self.label_original = None
        self.label_pred = None
        self.image_id = None

        self.training_loss = 0
        self.test_loss = 0
        self.f1_scores = 0
        self.pretrained = opt.pretrained

        self.encoder_out = opt.nf * (2 ** opt.res_blocks3x3) * 2

        # Models

        resnet = resnet50(pretrained=self.pretrained)
        if self.pretrained:
            for param in resnet.parameters():
                param.requires_grad_(False)

        modules = list(resnet.children())[:-1]

        self.resnet_encoder = nn.Sequential(*modules).to(self.device)
        self.linear_decoder = LinearDecoder(resnet.fc.in_features, opt.output_n).to(self.device)

        if self.gpu_ids:
            self.resnet_encoder = DataParallel(self.resnet_encoder, self.gpu_ids)
            self.linear_decoder = DataParallel(self.linear_decoder, self.gpu_ids)

        if self.isTrain:
            # Loss
            self.criterion = nn.CrossEntropyLoss()

            # Optimizer
            self.optimizer = optim.Adam(
                itertools.chain(self.resnet_encoder.parameters(), self.linear_decoder.parameters()),
                lr=opt.lr)
            # Continue Training
            if self.opt.ct > 0:
                print(f"Continue training from {self.opt.ct}")
                self.load_networks(self.opt.ct, load_optim=True)

        print(self.resnet_encoder)
        print(self.linear_decoder)

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
        feature_vec = self.resnet_encoder(self.image)
        self.label_pred = self.linear_decoder(feature_vec)

    def train(self):
        self.resnet_encoder.train()
        self.linear_decoder.train()

    def eval(self):
        self.resnet_encoder.eval()
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

    def get_inference(self) -> dict:
        _, predicted = torch.max(self.label_pred, dim=1)
        return {'output': predicted.cpu().numpy(), 'image_id': self.image_id}

    def save_networks(self, epoch: str) -> None:
        """Save models
        :param epoch: Current Epoch (prefix for the name)
        """
        if not self.pretrained:
            self.save_network(self.resnet_encoder, 'resnet_encoder', epoch)
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
        if not self.pretrained:
            self._load_object('resnet_encoder', f"{model_name}_net_resnet_encoder.pt")

        self._load_object('linear_decoder', f"{model_name}_net_linear_decoder.pt")

        if load_optim:
            self._load_object('optimizer', f"{model_name}_optimizer.pt")

    def _load_object(self, object_name: str, model_name: str):
        path = os.path.join(self.save_dir, model_name)
        state_dict = torch.load(path, map_location=self.device)

        net = getattr(self, object_name)
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        net.load_state_dict(state_dict)
        print(f"Loading [{object_name}] from {path}")
