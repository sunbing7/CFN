Code for Paper "Defending against Universal Adversarial Patches by Clipping Feature Norms".


## Usage
ImageNet:

Limitted by the size of the committed file, we cannot provide the finetuned models and the ImageNet dataset currently. You can construct the dataset as the format of torchvision.datasets. 

Run the following code and adjust the statistics of the BatchNorm layer.

CUDA_VISIBLE_DEVICES=x python train_models.py --dist-url \'tcp://127.0.0.1:6008\' --dist-backend \'nccl\' --multiprocessing-distributed --world-size 1 --rank 0 /your/path/to/imagenet --model_type 0(or 1 or 2) --clp 1.0(or 1.1)
--model_type: 0 is using ResNet-50, 1 is using Inception-V3, 2 is using MobileNetV2
--clp: the clipping parameter \alpha

Then run the following code and get the result of our method in Table 2.

CUDA_VISIBLE_DEVICES=x python patchattack_models.py --advp 0(or 1) --clp 1.0(or 1.1) --model_type 0(or 1 or 2) --data /your/path/to/imagenet/validation
--model_type: 0 is using ResNet-50, 1 is using Inception-V3, 2 is using MobileNet-V2
--clp: the clipping parameter \alpha
--advp: 0 is using Adversarial Patch attack, 1 is using LaVAN attack



## Requirements
- Python 3.6.0
- Pytorch 1.4.0
- Torchvision 0.5.0
