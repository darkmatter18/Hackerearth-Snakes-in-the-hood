import itertools

from torch import nn, optim
from torch.nn.parallel import DataParallel
from torchvision.models import resnet50

from .BaseModel import BaseModel
from .networks import LinearDecoder
from .resnet import ResNet


class ResnetModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.pretrained = opt.pretrained

        self.encoder_out = opt.nf * (2 ** opt.res_blocks3x3) * 2

        # Models
        if self.pretrained:
            resnet = resnet50(pretrained=self.pretrained)
            if self.pretrained:
                for param in resnet.parameters():
                    param.requires_grad_(False)

            modules = list(resnet.children())[:-1]

            self.resnet_encoder = nn.Sequential(*modules).to(self.device)
            self.linear_decoder = LinearDecoder(resnet.fc.in_features, opt.output_n).to(self.device)

        else:
            self.resnet_encoder = ResNet([3, 4, 6, 3], use_dropout=not opt.no_dropout).to(self.device)
            self.linear_decoder = LinearDecoder(512 * 4, opt.output_n).to(self.device)

        if self.gpu_ids:
            self.resnet_encoder = DataParallel(self.resnet_encoder, self.gpu_ids)
            self.linear_decoder = DataParallel(self.linear_decoder, self.gpu_ids)

        self.model_names = ['resnet_encoder', 'linear_decoder']

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
