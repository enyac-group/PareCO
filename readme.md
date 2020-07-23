# Code for PareCO

## Dependencies

- PyTorch==1.5.0
- torchvision==0.6.0
- BoTorch==0.2.5
- timm==0.1.26

`pip install -r requirements.txt`


## Code Base Structure

`train.py`: Implementation of our main algorithm. Including options to train PareCO and Slim.

`eval.py`: Evaluate the performance of a trained network by sampling widths and re-calibrate batchnorm statistics.

`search.py`: Post-training search with MOBO-RS.

`model/`: For ResNets used in CIFAR experiments

`pruner`: For parsing the depnedencies used for heterogeneous width-multipliers

`utils/`: Some helper functions including data loaders

## Example commands
We have prepared the pre-trained models for MobileNetV2 (hyperparams from [53]), which are the models for Table 2 and Figure 7a. To evaluate this pre-trained models, directly run the **evalute** commands listed below.

### MobileNetV2 Slim
**Training Slim MobileNetV2 (Slim hyperparam)**

    python -u train.py --name mobilenetv2_100_imagenet_slim_b1024_e250_lr0.125_ls0.1_lower0.42_slimhyper --datapath ${IMAGENET_DATA} --dataset torchvision.datasets.ImageFolder --network mobilenetv2_100 --reinit --resource_type flops --pruner FilterPrunerMBNetV3 --epochs 250 --warmup 5 --baselr 0.125 --tau 1 --wd 4e-5 --label_smoothing 0.1 --batch_size 1024 --param_level 1 --upper_flops 1 --lower_channel 0.42 --num_sampled_arch 2 --print_freq 100 --slim --scheduler linear_decay --slim_dataaug


**Evaluate Slim MobileNetV2 (Slim hyperparam)**

    python -u eval.py --name mobilenetv2_100_imagenet_slim_b1024_e250_lr0.125_ls0.1_lower0.42_slimhyper --datapath ${IMAGENET_DATA} --dataset torchvision.datasets.ImageFolder --network mobilenetv2_100 --resource_type flops --pruner FilterPrunerMBNetV3 --batch_size 1024 --lower_channel 0.42 --uniform --slim_dataaug


### MobileNetV2 PareCO
**Training PareCO MobileNetV2 (Slim hyperparam)**

    python -u train.py --name mobilenetv2_100_imagenet_pareco_b1024_e250_lr0.125_ls0.1_lower0.42_slimhyper --datapath ${IMAGENET_DATA} --dataset torchvision.datasets.ImageFolder --network mobilenetv2_100 --reinit --resource_type flops --pruner FilterPrunerMBNetV3 --epochs 250 --warmup 5 --baselr 0.125 --tau 625 --wd 4e-5 --label_smoothing 0.1 --batch_size 1024 --param_level 3 --upper_flops 1 --lower_channel 0.42 --num_sampled_arch 2 --baseline -3 --pas --print_freq 100 --prior_points 20 --scheduler linear_decay --slim_dataaug

**Evaluate PareCO MobileNetV2 (Slim hyperparam)**

    python -u eval.py --name mobilenetv2_100_imagenet_pareco_b1024_e250_lr0.125_ls0.1_lower0.42_slimhyper --datapath ${IMAGENET_DATA} --dataset torchvision.datasets.ImageFolder --network mobilenetv2_100 --resource_type flops --pruner FilterPrunerMBNetV3 --batch_size 1024 --lower_channel 0.42 --slim_dataaug



### MobileNetV2 TwoStage
**Training TwoStage MobileNetV2 (Slim hyperparam)**

    python -u train.py --name mobilenetv2_100_imagenet_twostage_b1024_e250_lr0.125_ls0.1_lower0.42_slimhyper --datapath ${IMAGENET_DATA} --dataset torchvision.datasets.ImageFolder --network mobilenetv2_100 --reinit --resource_type flops --pruner FilterPrunerMBNetV3 --epochs 250 --warmup 5 --baselr 0.125 --tau 1 --wd 4e-5 --label_smoothing 0.1 --batch_size 1024 --param_level 3 --upper_flops 1 --lower_channel 0.42 --num_sampled_arch 2 --print_freq 100 --slim --scheduler linear_decay --slim_dataaug

**Searching TwoStage MobileNetV2 (Slim hyperparam)**

    python -u search.py --name mobilenetv2_100_imagenet_twostage_b1024_e250_lr0.125_ls0.1_lower0.42_slimhyper_postsearch --datapath ${IMAGENET_DATA} --dataset torchvision.datasets.ImageFolder --network mobilenetv2_100 --model ./ckpt/mobilenetv2_100_imagenet_twostage_b1024_e250_lr0.125_ls0.1_lower0.42_slimhyper.pt --resource_type flops --pruner FilterPrunerMBNetV3 --batch_size 1024 --param_level 3 --upper_flops 1 --lower_channel 0.42 --print_freq 100 --slim_dataaug --reinit --pas


**Evaluate TwoStage MobileNetV2 (Slim hyperparam)**

    python -u eval.py --name mobilenetv2_100_imagenet_twostage_b1024_e250_lr0.125_ls0.1_lower0.42_slimhyper_postsearch --datapath ${IMAGENET_DATA} --dataset torchvision.datasets.ImageFolder --network mobilenetv2_100 --resource_type flops --pruner FilterPrunerMBNetV3 --batch_size 1024 --lower_channel 0.42 --slim_dataaug