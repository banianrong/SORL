DATA_PATH=$1
EXP_NAME=$2
CKPT_PATH=$3
LATENT_STATS_PATH=$4

accelerate launch --num_processes 8 train.py --config configs/sorl.yaml \
    --model="SiT-XL/2" \
    --encoder-depth=4 \
    --data-dir=$DATA_PATH \
    --exp-name=$EXP_NAME \
    --vae-pt=$CKPT_PATH \
    --latents-pt=$LATENT_STATS_PATH \
    --learning-rate=1e-4 

