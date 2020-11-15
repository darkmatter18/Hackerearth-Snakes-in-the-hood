from options.TrainOptions import TrainOptions
from data import create_dataset


def main():
    opt = TrainOptions().parse()
    trainloader, testloader = create_dataset(opt)

    print(next(iter(trainloader))['image'].shape)


if __name__ == '__main__':
    main()
