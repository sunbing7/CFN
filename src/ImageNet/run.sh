#python train_models.py /root/autodl-tmp/sunbing/workspace/uap/data/imagenet/validation_folder/ --model_type=0 --clp=1.0 --evaluate
#python train_models.py --pretrained true --multiprocessing-distributed false --rank 0 --data ../../../data/imagenet/validation/ --model_type 0 --clp 1.0)

#comparison
#python patchattack_models_danse.py --clp=0--model_type=0 --data=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/validation/ --uap_path=/root/autodl-tmp/sunbing/workspace/uap/CFN/src/ImageNet/uap/ --target=174

for TARGET_CLASS in {755,743,804,700,922,174,547,369}
do
    python patchattack_models_danse.py --clp=0--model_type=0 --data=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/validation/ --uap_path=/root/autodl-tmp/sunbing/workspace/uap/CFN/src/ImageNet/uap/ --target=$TARGET_CLASS
    python patchattack_models_danse.py --clp=1.0--model_type=0 --data=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/validation/ --uap_path=/root/autodl-tmp/sunbing/workspace/uap/CFN/src/ImageNet/uap/ --target=$TARGET_CLASS
    python patchattack_models_danse.py --clp=1.1--model_type=0 --data=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/validation/ --uap_path=/root/autodl-tmp/sunbing/workspace/uap/CFN/src/ImageNet/uap/ --target=$TARGET_CLASS
done