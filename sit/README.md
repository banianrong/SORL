# SIT-XL/2

## 环境配置

```
# Create environment (Python 3.10+)
conda create -n irepa python=3.10 -y && conda activate irepa

# Install dependencies
pip install -r environment/requirements.txt
```

## 数据预处理

```
cd ldm/preprocessing

# create preprocessed image
python dataset_tools.py convert --source=imagenet_path \
    --dest=preprocess_image_path --resolution=256x256 --transform=center-crop-dhariwal

# transform ckpt to safetensor (maybe need to pip install pytorch-lightning first)
python convert_vae_pt_to_diffusers.py \
    --vae_pt_path ckpt_path \
    --config_path v1-inference.yaml \
    --dump_path vae_safetensor_path

# create latent
python dataset_tools.py encode --source=preprocess_image_path \
    --dest=latent_image_path \
    --model-url vae_safetensor_path

# transform npy to arrow
python convert_npy_to_arrow.py --images_dir preprocess_image_path \
	--features_dir latent_image_path --target_path data_path

# get latent stats(scale, bias)
python extract.py --images_dir preprocess_image_path \
--features_dir latent_image_path \
--target_path latent_stats_path
```

- `imagenet_path`：imagenet的训练集位置（注意这里imagenet的目录结构要是分类的，如下所示）
    ```
    # imagenet dir 
    - n02098749
        - image1
        - image2
    - n02439483
        - image3
        - image4
    ...
    ```
- `preprocess_image_path`：存放经过裁剪缩放后的图像的位置
- `ckpt_path`：所需要进行转换的一阶段的权重路径
- `vae_safetensor_path`：所需要存放的safetensor文件夹路径
- `latent_image_path`：存放预处理的图像隐空间向量的文件夹路径
- `data_path`：最终处理整合好的数据存放路径
- `latent_stats_path`：存放latent_stats.pt的路径，e.g. `--target_path ./latent_stats.pt`

## 训练指令

```
accelerate launch --num_processes 8 train.py --config configs/sorl.yaml \
  --model="SiT-XL/2" \
  --encoder-depth=4 \
  --data-dir=data_path \
  --exp-name="sorl" \
  --vae-pt ckpt_path \
  --latents-pt latent_stats_path
```

- `num_processes`：指定gpu数量
- `data-dir`：填写上面最终生成的预处理后的数据集路径`data_path`
- `vae-pt`：填写一阶段ckpt的路径
- `latents-pt`：填写上面生成的`latent_stats.pt`的路径