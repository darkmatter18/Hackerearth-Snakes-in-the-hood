from .BaseOptions import BaseOptions


class TrainOptions(BaseOptions):
    def __init__(self):
        super().__init__()
        self.isTrain = True

    def initialized(self, parser):
        parser = BaseOptions.initialized(self, parser)

        # Training Stats params
        parser.add_argument('--print_freq', type=int, default=10,
                            help='frequency of showing training results on console')

        # network saving and loading parameters
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--train_ratio', type=float, default=0.8, help="The ratio for train test split")
        parser.add_argument('--ct', type=int, default=0, help='Adding continue training. '
                                                              'The value is the epoch no, which the model will start '
                                                              'training from, and loads the model from.')
        parser.add_argument('--epoch_count', type=int, default=1,
                            help='the starting epoch count, '
                                 'we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>,')
        parser.add_argument('--save_latest_freq', type=int, default=600, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=4,
                            help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')

        # Training Parameters
        parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')

        return parser
