import os
from abc import ABC, abstractmethod

import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader


class BaseModel(ABC):
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

        self.model_names = []
        self.optimizer = None
        self.criterion = None
        self.scheduler = None

        self.training_loss = 0.0
        self.test_loss = 0.0
        self.f1_scores = 0.0

    @abstractmethod
    def feed_input(self, x: dict):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        :param x: include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        pass

    @abstractmethod
    def optimize_parameters(self):
        """
        Optimizes parameters
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass
        Called by both functions <optimize_parameters> and <test>
        """
        pass

    def update_learning_rate(self):
        old_lr = self.optimizer.param_groups[0]['lr']
        if self.scheduler:
            self.scheduler.step()
        else:
            print("No LR scheduler found!")
        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def train(self):
        """
        Set the Models in the train mode
        """
        for model in self.model_names:
            net = getattr(self, model)
            net.train()

    def eval(self) -> None:
        """
        Set the Models in the eval mode
        """
        for model in self.model_names:
            net = getattr(self, model)
            net.eval()

    def get_last_avg_train_loss(self) -> float:
        """
        Get the Training loss for the last batch
        :return: Training loss for last batch
        """
        loss = self.training_loss / self.opt.print_freq
        self.training_loss = 0
        return loss

    def _single_test(self, data: dict) -> None:
        """
        Single input testing pipeline
        :param data: The single batch of data
        """
        self.feed_input(data)
        self.forward()
        test_loss = self.criterion(self.label_pred, self.label_original).item()
        # print("Test Loss", test_loss)
        self.test_loss += test_loss
        self.f1_scores += f1_score(self.label_original.cpu().numpy(), self.label_pred.argmax(dim=1).cpu().numpy(),
                                   average='weighted')

    def run_test_on_training(self, testloader: DataLoader) -> tuple:
        """
        Run the full Testing on the time of training
        :param testloader: The testloader, having test data
        :return: tuple of (test_loss, f1_score*100)
        """
        with torch.no_grad():
            no = 0
            for i, data in enumerate(testloader):
                self._single_test(data)
                no = i + 1
        # print("Length of testloader", no)
        test_loss = self.test_loss / no
        f1_ = 100 * self.f1_scores / no
        self.test_loss = 0
        self.f1_scores = 0

        return test_loss, f1_

    def get_inference(self) -> dict:
        """
        Get the inference Values of each batch of data
        :return: {output: The output index, image_id: id of the image, label_orig: original label}
        """
        return {'output': self.label_pred.argmax(dim=1).cpu().numpy(), 'image_id': self.image_id,
                'label_orig': self.label_original.cpu().numpy()}

    def save_networks(self, epoch: str) -> None:
        """Save models
        :param epoch: Current Epoch (prefix for the name)
        """
        for model_name in self.model_names:
            net = getattr(self, model_name)
            self._save_network(net, model_name, epoch)

    def _save_network(self, net, net_name, epoch) -> None:
        save_filename = f'{epoch}_net_{net_name}.pt'
        save_path = os.path.join(self.save_dir, save_filename)

        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(net.module.cpu().state_dict(), save_path)
            net.cuda(self.gpu_ids[0])
        else:
            torch.save(net.cpu().state_dict(), save_path)

    def save_optimizer_scheduler(self, epoch: str) -> None:
        """
        Saved Optimizer and LR scheduler
        :param epoch: initials for the name
        """
        optimizer_save_path = os.path.join(self.save_dir, f"{epoch}_optimizer.pt")
        scheduler_save_path = os.path.join(self.save_dir, f"{epoch}_scheduler.pt")

        torch.save(self.optimizer.state_dict(), optimizer_save_path)

        if self.scheduler:
            torch.save(self.scheduler.state_dict(), scheduler_save_path)

    def load_networks(self, model_name: str, load_optim: bool = False, load_scheduler: bool = False) -> None:
        """
        Load models, optimizer and scheduler
        :param model_name: initials for the name of model, optimizer and scheduler
        :param load_optim: Load optimizer or not
        :param load_scheduler: Load scheduler or not
        """
        for model in self.model_names:
            self._load_object(model, f"{model_name}_net_{model}.pt")

        if load_optim:
            self._load_object('optimizer', f"{model_name}_optimizer.pt")

        if load_scheduler and self.scheduler:
            self._load_object('scheduler', f"{model_name}_scheduler.pt")

    def _load_object(self, object_name: str, model_name: str) -> None:
        path = os.path.join(self.save_dir, model_name)
        state_dict = torch.load(path, map_location=self.device)

        net = getattr(self, object_name)
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        net.load_state_dict(state_dict)
        print(f"Loading [{object_name}] from {path}")
