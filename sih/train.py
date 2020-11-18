import time
from data import create_dataset
from model.cnn_model import CnnModel
from options.TrainOptions import TrainOptions
from utils.TestStats import TrainStats


def main():
    opt = TrainOptions().parse()
    trainloader, testloader = create_dataset(opt)

    model = CnnModel(opt)
    stats = TrainStats(opt)

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
                train_loss = model.get_last_avg_train_loss()
                test_loss, f1_score = model.run_test_on_training(testloader)
                stats.print_current_losses(epoch, epoch_iter, train_loss, test_loss, f1_score, t_comp, t_data)

            if total_iters % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)
                model.save_optimizer(save_suffix)

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_optimizer('latest')
            model.save_networks(str(epoch))
            model.save_optimizer(str(epoch))

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.n_epochs, time.time() - epoch_start_time))

    print("End of Training")


if __name__ == '__main__':
    main()
