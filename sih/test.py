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
    for load_model_prefix in range(opt.start_load, opt.end_load, opt.difference):
        model.load_networks(load_model_prefix)
        model.eval()

        with torch.no_grad():
            for i, data in enumerate(dataset):
                model.feed_input(data)
                model.forward()
                batch_out = model.get_inference()
                test_store.load_test_data(batch_out['image_id'], batch_out['output'], batch_out['label_orig'])
                print(f"Written batch {i} of {len(dataset)//opt.batch_size}")

        if opt.write:
            test_store.write(load_model_prefix)


if __name__ == '__main__':
    main()
