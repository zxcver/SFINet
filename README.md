# SFINet

## Introduction
We release the code on "SFINet: Shuffle–and–Fusion Interaction Networks for Wind Power Forecasting".  
We mainly reference the code with [SCINet](https://github.com/cure-lab/SCINet).

## Install
please note your cuda version and reference [get-started](https://pytorch.org/get-started/locally/) while install pytorch.
```
cd SFINet
conda create -n sfinet python=3.8
conda activate sfinet
pip install torch torchvision torchaudio 
pip install -r requirements.txt
```

## Data Preparation
```
./
└── datasets/
    ├── ETT-data
    │   ├── ETTh1.csv
    │   ├── ETTh2.csv
    │   └── ETTm1.csv
    └── windpower
        ├── wpm1.csv
        └── wpm2.csv
```
We follow the same settings of [SCINet](https://github.com/cure-lab/SCINet) for ETTH1, ETTH2, ETTM1 datasets.
Since the privacy implications of the data set are being considered, we cannot release WP datasets.

## Training
### ETTh1
#### Multivariate, out 24
```
    python run_ETTh.py --data ETTh1 --gpu 0 --patience 80 --root_path datasets/ETT-data \
                        --data_path ETTh1.csv --features M  --train_epochs 100 --seq_len 48 \
                        --label_len 24 --pred_len 24 --lr 0.02 --batch_size 512 --hidden_size 2 --levels 4  
```

#### Multivariate, out 48
```
    python run_ETTh.py --data ETTh1 --gpu 0 --patience 80 --root_path datasets/ETT-data \
                        --data_path ETTh1.csv --features M  --train_epochs 100 --seq_len 96 \
                        --label_len 48 --pred_len 48 --lr 0.01 --batch_size 96 --hidden_size 1 --levels 5
```

#### Multivariate, out 168
```
    python run_ETTh.py --data ETTh1 --gpu 0 --patience 80 --root_path datasets/ETT-data \
                        --data_path ETTh1.csv --features M  --train_epochs 100 --seq_len 336 \
                        --label_len 168 --pred_len 168 --lr 0.005 --batch_size 128 --hidden_size 1 --levels 3  
```

#### Multivariate, out 336
```
    python run_ETTh.py --data ETTh1 --gpu 0 --patience 80 --root_path datasets/ETT-data \
                        --data_path ETTh1.csv --features M  --train_epochs 100 --seq_len 672 \
                        --label_len 336 --pred_len 336 --lr 0.0001 --batch_size 256 --hidden_size 1 --levels 1  
```

#### Multivariate, out 720
```
    python run_ETTh.py --data ETTh1 --gpu 0 --patience 80 --root_path datasets/ETT-data \
                        --data_path ETTh1.csv --features M  --train_epochs 100 --seq_len 1440 \
                        --label_len 720 --pred_len 720  --lr 0.0001 --batch_size 256 --hidden_size 1 --levels 1
```

#### Univariate, out 24
```
    python run_ETTh.py --data ETTh1 --gpu 0 --patience 80 --root_path datasets/ETT-data \
                        --data_path ETTh1.csv --features S --train_epochs 100 --seq_len 48 \
                        --label_len 24 --pred_len 24 --lr 0.02 --batch_size 256 --hidden_size 4 --levels 4 
```

#### Univariate, out 48
```
    python run_ETTh.py --data ETTh1 --gpu 0 --patience 80 --root_path datasets/ETT-data \
                        --data_path ETTh1.csv --features S --train_epochs 100 --seq_len 96 \
                        --label_len 48 --pred_len 48 --lr 0.05 --batch_size 512 --hidden_size 1 --levels 1
```

#### Univariate, out 168
```
    python run_ETTh.py --data ETTh1 --gpu 0 --patience 80 --root_path datasets/ETT-data \
                        --data_path ETTh1.csv --features S --train_epochs 100 --seq_len 336 \
                        --label_len 168 --pred_len 168 --lr 0.001 --batch_size 128 --hidden_size 4 --levels 3
```

#### Univariate, out 336
```
    python run_ETTh.py --data ETTh1 --gpu 0 --patience 80 --root_path datasets/ETT-data \
                        --data_path ETTh1.csv --features S --train_epochs 100 --seq_len 672 \
                        --label_len 336 --pred_len 336 --lr 0.001 --batch_size 64 --hidden_size 2 --levels 1
```

#### Univariate, out 720
```
    python run_ETTh.py --data ETTh1 --gpu 0 --patience 80 --root_path datasets/ETT-data \
                        --data_path ETTh1.csv --features S --train_epochs 100 --seq_len 1440 \
                        --label_len 720 --pred_len 720 --lr 0.001 --batch_size 512 --hidden_size 1 --levels 2
```

### ETTh2
#### Multivariate, out 24
```
    python run_ETTh.py --data ETTh1 --gpu 0 --patience 80 --root_path datasets/ETT-data \
                        --data_path ETTh2.csv --features M  --train_epochs 100 --seq_len 48 \
                        --label_len 24 --pred_len 24 --lr 0.01 --batch_size 128 --hidden_size 1 --levels 4  
```

#### Multivariate, out 48
```
    python run_ETTh.py --data ETTh1 --gpu 0 --patience 80 --root_path datasets/ETT-data \
                        --data_path ETTh2.csv --features M  --train_epochs 100 --seq_len 96 \
                        --label_len 48 --pred_len 48 --lr 0.007 --batch_size 96 --hidden_size 2 --levels 3
```

#### Multivariate, out 168
```
    python run_ETTh.py --data ETTh1 --gpu 0 --patience 80 --root_path datasets/ETT-data \
                        --data_path ETTh2.csv --features M  --train_epochs 100 --seq_len 336 \
                        --label_len 168 --pred_len 168 --lr 0.005 --batch_size 128 --hidden_size 1 --levels 4  
```

#### Multivariate, out 336
```
    python run_ETTh.py --data ETTh1 --gpu 0 --patience 80 --root_path datasets/ETT-data \
                        --data_path ETTh2.csv --features M  --train_epochs 100 --seq_len 672 \
                        --label_len 336 --pred_len 336 --lr 0.01 --batch_size 128 --hidden_size 1 --levels 2  
```

#### Multivariate, out 720
```
    python run_ETTh.py --data ETTh1 --gpu 0 --patience 80 --root_path datasets/ETT-data \
                        --data_path ETTh2.csv --features M  --train_epochs 100 --seq_len 1440 \
                        --label_len 720 --pred_len 720  --lr 0.005 --batch_size 512 --hidden_size 1 --levels 1
```

#### Univariate, out 24
```
    python run_ETTh.py --data ETTh1 --gpu 0 --patience 80 --root_path datasets/ETT-data \
                        --data_path ETTh2.csv --features S --train_epochs 100 --seq_len 48 \
                        --label_len 24 --pred_len 24 --lr 0.05 --batch_size 512 --hidden_size 2 --levels 1 
```

#### Univariate, out 48
```
    python run_ETTh.py --data ETTh1 --gpu 0 --patience 80 --root_path datasets/ETT-data \
                        --data_path ETTh2.csv --features S --train_epochs 100 --seq_len 96 \
                        --label_len 48 --pred_len 48 --lr 0.05 --batch_size 256 --hidden_size 4 --levels 1
```
#### Univariate, out 168
```
    python run_ETTh.py --data ETTh1 --gpu 0 --patience 80 --root_path datasets/ETT-data \
                        --data_path ETTh2.csv --features S --train_epochs 100 --seq_len 336 \
                        --label_len 168 --pred_len 168 --lr 0.002 --batch_size 512 --hidden_size 2 --levels 2
```

#### Univariate, out 336
```
    python run_ETTh.py --data ETTh1 --gpu 0 --patience 80 --root_path datasets/ETT-data \
                        --data_path ETTh2.csv --features S --train_epochs 100 --seq_len 672 \
                        --label_len 336 --pred_len 336 --lr 0.02 --batch_size 512 --hidden_size 1 --levels 2
```

#### Univariate, out 720
```
    python run_ETTh.py --data ETTh1 --gpu 0 --patience 80 --root_path datasets/ETT-data \
                        --data_path ETTh2.csv --features S --train_epochs 100 --seq_len 1440 \
                        --label_len 720 --pred_len 720 --lr 0.02 --batch_size 512 --hidden_size 1 --levels 3
```

### ETTm1
#### Multivariate, out 24
```
    python run_ETTh.py --data ETTh1 --gpu 0 --patience 80 --root_path datasets/ETT-data \
                        --data_path ETTm1.csv --features M  --train_epochs 100 --seq_len 48 \
                        --label_len 24 --pred_len 24 --lr 0.001 --batch_size 96 --hidden_size 4 --levels 1
```
#### Multivariate, out 48
```
    python run_ETTh.py --data ETTh1 --gpu 0 --patience 80 --root_path datasets/ETT-data \
                        --data_path ETTm1.csv --features M  --train_epochs 100 --seq_len 96 \
                        --label_len 48 --pred_len 48 --lr 0.001 --batch_size 96 --hidden_size 1 --levels 4  
```
#### Multivariate, out 96
```
    python run_ETTh.py --data ETTh1 --gpu 0 --patience 80 --root_path datasets/ETT-data \
                        --data_path ETTm1.csv --features M  --train_epochs 100 --seq_len 48 \
                        --label_len 96 --pred_len 96 --lr 0.001 --batch_size 96 --hidden_size 4 --levels 3
```
#### Multivariate, out 288
```
    python run_ETTh.py --data ETTh1 --gpu 0 --patience 80 --root_path datasets/ETT-data \
                        --data_path ETTm1.csv --features M  --train_epochs 100 --seq_len 672 \
                        --label_len 288 --pred_len 288 --lr 0.0001 --batch_size 96 --hidden_size 1 --levels 1  
```
#### Multivariate, out 672
```
    python run_ETTh.py --data ETTh1 --gpu 0 --patience 80 --root_path datasets/ETT-data \
                        --data_path ETTm1.csv --features M  --train_epochs 100 --seq_len 672 \
                        --label_len 672 --pred_len 672 --lr 0.00005 --batch_size 32 --hidden_size 1 --levels 1
```
#### Univariate, out 24
```
    python run_ETTh.py --data ETTh1 --gpu 0 --patience 80 --root_path datasets/ETT-data \
                        --data_path ETTm1.csv --features S  --train_epochs 100 --seq_len 96 \
                        --label_len 24 --pred_len 24 --lr 0.001 --batch_size 16 --hidden_size 4 --levels 2
```
#### Univariate, out 48
```
    python run_ETTh.py --data ETTh1 --gpu 0 --patience 80 --root_path datasets/ETT-data \
                        --data_path ETTm1.csv --features S  --train_epochs 100 --seq_len 96 \
                        --label_len 48 --pred_len 48 --lr 0.002 --batch_size 96 --hidden_size 2 --levels 2
```
#### Univariate, out 96
```
    python run_ETTh.py --data ETTh1 --gpu 0 --patience 80 --root_path datasets/ETT-data \
                        --data_path ETTm1.csv --features S  --train_epochs 100 --seq_len 384 \
                        --label_len 96 --pred_len 96 --lr 0.01 --batch_size 96 --hidden_size 1 --levels 1
```
#### Univariate, out 288
```
    python run_ETTh.py --data ETTh1 --gpu 0 --patience 80 --root_path datasets/ETT-data \
                        --data_path ETTm1.csv --features S  --train_epochs 100 --seq_len 672 \
                        --label_len 288 --pred_len 288 --lr 0.001 --batch_size 96 --hidden_size 1 --levels 1
```
#### Univariate, out 672
```
    python run_ETTh.py --data ETTh1 --gpu 0 --patience 80 --root_path datasets/ETT-data \
                        --data_path ETTm1.csv --features S  --train_epochs 100 --seq_len 672 \
                        --label_len 672 --pred_len 672 --lr 0.0005 --batch_size 96 --hidden_size 4 --levels 4
```

## Reference
[SCINet](https://github.com/cure-lab/SCINet)


