import time
from data import create_dataset
from model.cnn_model import CnnModel
from options.TrainOptions import TrainOptions


def main():
    opt = TrainOptions().parse()
    trainloader, testloader = create_dataset(opt)

    model = CnnModel(opt)

    total_iters = 0                # the total number of training iterations
    for epoch in range(opt.epoch_count, opt.n_epochs + 1):
        epoch_start_time = time.time()
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        for i, data in enumerate(trainloader):
            model.train()
            model.feed_input(data)

            iter_start_time = time.time()
            model.optimize_parameters()

            total_iters += 1
            epoch_iter += 1

            model.eval()

            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - epoch_start_time
                t_comp = (time.time() - iter_start_time) / opt.batch_size


if __name__ == '__main__':
    main()
