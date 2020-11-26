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

3. **20201125-0730PM**
    - Model: `resnet50 pretrained`
    - Train Script: `python -m sih.train --dataroot ./dataset --model resnet --pretrained --batch_size 256 --num_threads 4 --n_epochs 200 --print_freq 5 --name resnet --name_time --load_size 128 --crop_size 128`
    - Test Script: `python -m sih.test --dataroot ./dataset --name resnet1606200038 --model resnet --pretrained --batch_size 64 --write`
    - Epoch: `48`
    - F1_score: `21.70925`
    
4. **20201125-0850PM**
    - Model: `resnet50 pretrained`
    - Optimizer: `SGD  0.0001`
    - Train Script: `python -m sih.train --dataroot ./dataset --model resnet --pretrained --batch_size 512 --num_threads
     4 --n_epochs 400 --print_freq 5 --name resnet --name_time --load_size 128 --crop_size 128 --lr 0.0001 --save_latest_freq 400`
    - Test Script: `python -m sih.test --dataroot ./dataset --name resnet1606314894 --model resnet --pretrained 
    --load_model 104 --batch_size 64 --write`
    - Epoch: `104`
    - F1_score: 4.72493

5. **20201125-0914PM**
    - Model: `resnet50 pretrained`
    - Optimizer: `Adam 0.0002`
    - Train Script: `python -m sih.train --dataroot ./dataset --model resnet --pretrained --batch_size 512 --num_threads 
    4 --n_epochs 400 --print_freq 5 --name resnet --name_time --load_size 128 --crop_size 128 --lr 0.0001 --save_latest_freq 40`
    - Test Script: `python -m sih.test --dataroot ./dataset --name resnet1606317737 --model resnet --pretrained 
    --load_model 60 --batch_size 64 --write`
    - Epoch: `60`
    - F1_score: `19.07395`
    
6. **20201125-0914PM**
    - Model: `resnet50 pretrained`
    - Optimizer: `Adam 0.0001`
    - Train Script: `python -m sih.train --dataroot ./dataset --model resnet --pretrained --batch_size 512 --num_threads 
    4 --n_epochs 400 --print_freq 5 --name resnet --name_time --load_size 128 --crop_size 128 --lr 0.0001 --save_latest_freq 40`
    - Test Script: `python -m sih.test --dataroot ./dataset --name resnet1606317737 --model resnet --pretrained 
    --load_model 80 --batch_size 64 --write`
    - Epoch: `80`
    - F1_score: `20.51618`
    
7. **20201125-0920PM**
    - Model: `resnet50 pretrained`
    - Optimizer: `Adam 0.0001`
    - Train Script: `python -m sih.train --dataroot ./dataset --model resnet --pretrained --batch_size 512 --num_threads 
    4 --n_epochs 400 --print_freq 5 --name resnet --name_time --load_size 128 --crop_size 128 --lr 0.0001 --save_latest_freq 40`
    - Test Script: `python -m sih.test --dataroot ./dataset --name resnet1606317737 --model resnet --pretrained 
    --load_model 100 --batch_size 64 --write`
    - Epoch: `100`
    - F1_score: `20.04543`
    
8. **20201125-0936PM**
    - Model: `resnet50 pretrained`
    - Optimizer: `Adam 0.0001`
    - Train Script: `python -m sih.train --dataroot ./dataset --model resnet --pretrained --batch_size 512 --num_threads 
    4 --n_epochs 400 --print_freq 5 --name resnet --name_time --load_size 128 --crop_size 128 --lr 0.0001 --save_latest_freq 40`
    - Test Script: `python -m sih.test --dataroot ./dataset --name resnet1606317737 --model resnet --pretrained 
    --load_model 100 --batch_size 64 --write`
    - Epoch: `100`
    - F1_score: `20.04543`
    
    
9. **20201125-0936PM**
    - Model: `resnet50 pretrained`
    - Optimizer: `Adam 0.0001`
    - Train Script: `python -m sih.train --dataroot ./dataset --model resnet --pretrained --batch_size 512 --num_threads 
    4 --n_epochs 400 --print_freq 5 --name resnet --name_time --load_size 128 --crop_size 128 --lr 0.0001 --save_latest_freq 40`
    - Test Script: `python -m sih.test --dataroot ./dataset --name resnet1606317737 --model resnet --pretrained 
    --load_model 180 --batch_size 64 --write`
    - Epoch: `180`
    - F1_score: `21.42162`
    
10. **20201125-0936PM**
    - Model: `resnet50 pretrained`
    - Optimizer: `Adam 0.0001`
    - Train Script: `python -m sih.train --dataroot ./dataset --model resnet --pretrained --batch_size 512 --num_threads 
    4 --n_epochs 400 --print_freq 5 --name resnet --name_time --load_size 128 --crop_size 128 --lr 0.0001 --save_latest_freq 40`
    - Test Script: `python -m sih.test --dataroot ./dataset --name resnet1606317737 --model resnet --pretrained 
    --load_model 232 --batch_size 64 --write`
    - Epoch: `232`
    - F1_score: `21.11259`

10. **20201125-1038PM**
    - Model: `resnet50 pretrained`
    - Optimizer: `Adam 0.0001`
    - Train Script: `python -m sih.train --dataroot ./dataset --model resnet --pretrained --batch_size 512 --num_threads 
    4 --n_epochs 400 --print_freq 5 --name resnet --name_time --load_size 128 --crop_size 128 --lr 0.0001 --save_latest_freq 40`
    - Test Script: `python -m sih.test --dataroot ./dataset --name resnet1606317737 --model resnet --pretrained 
    --load_model 300 --batch_size 64 --write`
    - Epoch: `300`
    - F1_score: `21.70110`
    
10. **20201125-1038PM**
    - Model: `resnet50 pretrained`
    - Optimizer: `Adam 0.0001`
    - Train Script: `python -m sih.train --dataroot ./dataset --model resnet --pretrained --batch_size 512 --num_threads 
    4 --n_epochs 400 --print_freq 5 --name resnet --name_time --load_size 128 --crop_size 128 --lr 0.0001 --save_latest_freq 40`
    - Test Script: `python -m sih.test --dataroot ./dataset --name resnet1606324976 --model resnet --pretrained 
    --load_model 320 --batch_size 64 --write`
    - Epoch: `320`
    - F1_score: `22.05227`