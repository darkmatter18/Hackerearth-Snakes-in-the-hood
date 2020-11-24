# Train data:

1. **20201124-1258PM**
    - Model: `resnet50 pretrained`
    - Train Script: `python -m sih.train --dataroot ./dataset --model resnet --pretrained --batch_size 256 --num_threads 4 --n_epochs 200 --print_freq 5 --name resnet --name_time --load_size 128 --crop_size 128`
    - Test Script: `python -m sih.test --dataroot ./dataset --name resnet1606200038 --model resnet --pretrained --batch_size 64 --write`
    - Epoch: `67`
    - F1_score: 21.39777
2. **20201124-0234PM**
    - Model: `resnet50`
    - Train Script: `python -m sih.train --dataroot ./dataset --model resnet --batch_size 256 --num_threads 4 --n_epochs 200 --print_freq 5 --name resnet --name_time --load_size 128 --crop_size 128`
    - Test Script: `python -m sih.test --dataroot ./dataset --name resnet1606200038 --model resnet --pretrained --batch_size 64 --write`
    - Epoch: `80`
    - F1_score: 2.9