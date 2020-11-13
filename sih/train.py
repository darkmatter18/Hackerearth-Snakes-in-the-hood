from options.TrainOptions import TrainOptions
from data import create_dataset


def main():
    opt = TrainOptions().parse()
    create_dataset(opt)


if __name__ == '__main__':
    main()
