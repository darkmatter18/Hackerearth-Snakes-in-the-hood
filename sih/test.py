import torch
from options.TestOptions import TestOptions
from data import create_test_dataset
from model import create_model
from utils.TestStore import TestStore


def main():
    opt = TestOptions().parse()
    opt.num_threads = 0  # test code only supports num_threads = 1

    # Dataset
    dataset = create_test_dataset(opt)

    test_store = TestStore(opt)

    # setup Gan
    model = create_model(opt)
    model.load_networks(opt.load_model)
    model.eval()

    with torch.no_grad():
        for i, data in enumerate(dataset):
            model.feed_input(data)
            model.forward()
            batch_out = model.get_inference()
            test_store.load_test_data(batch_out['image_id'], batch_out['output'], batch_out['label_orig'])
            print(f"Written batch {i} of {len(dataset)//opt.batch_size}")

    if opt.write:
        test_store.write()


if __name__ == '__main__':
    main()
