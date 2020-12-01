from .BaseModel import BaseModel

from torch import nn, optim
from torch.nn import DataParallel
from torchvision.models import resnet50


class ResnetV2Model(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)

        self.model = resnet50(pretrained=True)
        num_filters = self.model.fc.in_features
        self.model.fc = nn.Linear(num_filters, opt.output_n)

        self.model = self.model.to(self.device)
        if self.gpu_ids:
            self.model = DataParallel(self.model, self.gpu_ids)

        self.model_names = ['model']

        if self.isTrain:
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001, betas=(0.5, 0.999))
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

            # Continue Training
            if self.opt.ct > 0:
                print(f"Continue training from {self.opt.ct}")
                self.load_networks(self.opt.ct, load_optim=True)

        print(self.model)

    def forward(self):
        self.label_pred = self.model(self.image)

    def feed_input(self, x: dict):
        self.image = x['image'].to(self.device)
        self.label_original = x['label'].to(self.device)
        self.image_id = x['image_id']

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.forward()
        loss = self.criterion(self.label_pred, self.label_original)
        loss.backward()
        self.optimizer.step()
        self.training_loss += loss.item()
