import os
import time
import pickle


class TrainStats:
    def __init__(self, opt):
        # create a logging file to store training losses
        self.log_loss_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.log_file_name = f'loss_log_{str(int(time.time()))}.txt'
        self.loss_file_name = f'loss_stats_{str(int(time.time()))}.pkl'
        self.losses = {'train_loss': [], 'test_loss': [], 'f1_score': []}

        lss = os.path.join(self.log_loss_dir, self.log_file_name)

        with open(lss, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def print_current_losses(self, epoch: int, iters: int, train_loss: float, test_loss: float, f1_score: float,
                             t_comp: float, t_data: float):
        """print current losses on console; also save the losses to the disk

        :param f1_score:
        :param test_loss:
        :param train_loss:
        :param epoch: current epoch
        :param iters: current training iteration during this epoch (reset to 0 at the end of every epoch)
        :param t_comp: computational time per data point (normalized by batch_size)
        :param t_data: data loading time per data point (normalized by batch_size)
        """
        message = f'(epoch: {epoch}, iters: {iters}, time: {t_comp:.3f}, data: {t_data:.3f}) training_loss: ' \
                  f'{train_loss}, test_loss: {test_loss}, f1_score: {f1_score}'

        self.losses['train_loss'].append(train_loss)
        self.losses['test_loss'].append(test_loss)
        self.losses['f1_score'].append(f1_score)

        lss = os.path.join(self.log_loss_dir, self.log_file_name)
        liss = os.path.join(self.log_loss_dir, self.loss_file_name)

        print("[STATS]", message)  # print the message
        with open(lss, "a") as log_file:
            log_file.write('%s\n' % message)

        with open(liss, 'wb') as f:
            pickle.dump(self.losses, f)
