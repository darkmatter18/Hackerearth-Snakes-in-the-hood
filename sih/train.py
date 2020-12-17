import time
from data import create_dataset
from model import create_model
from options.TrainOptions import TrainOptions
from utils.TestStats import TrainStats


def main():
    opt = TrainOptions().parse()
    trainloader, testloader = create_dataset(opt)

    model = create_model(opt)
    stats = TrainStats(opt)

    total_iters = 0                # the total number of training iterations
    for epoch in range(opt.epoch_count, opt.n_epochs + 1):
        epoch_start_time = time.time()
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        
        model.train()
        print(f"Training {epoch}/{opt.n_epochs}")
        for i, data in enumerate(trainloader):
            
            model.feed_input(data)
            model.optimize_parameters()

            total_iters += 1
            epoch_iter += 1
            
        training_end_time = time.time()
        model.eval()
            
        t_data = training_end_time - epoch_start_time #Training Time
        t_comp =  t_data / opt.batch_size  #Single input time
        
        train_loss = model.get_last_train_loss() / len(trainloader.dataset)
        
        print(f"Evaluating {epoch}/{opt.n_epochs}")
        test_loss, f1_score = model.evaluate_test(testloader)
        stats.print_current_losses(epoch, epoch_iter, train_loss, test_loss, f1_score, t_comp, t_data)


        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_optimizer_scheduler('latest')
            model.save_networks(str(epoch))
            model.save_optimizer_scheduler(str(epoch))

        model.update_learning_rate(test_loss)
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.n_epochs, time.time() - epoch_start_time))

    print("End of Training")


if __name__ == '__main__':
    main()
