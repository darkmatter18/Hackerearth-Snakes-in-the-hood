from .BaseOptions import BaseOptions


class TestOptions(BaseOptions):
    def __init__(self):
        super().__init__()
        self.isTrain = False

    def initialized(self, parser):
        parser = BaseOptions.initialized(self, parser)

        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--load_model', type=str, default='latest', help="name of the models to load")
        parser.add_argument('--write', action="store_true", help="Write to the CSV file")
        parser.add_argument('--test_file_name', type=str, default="test.csv", help="the name of the file, write into")
        return parser
