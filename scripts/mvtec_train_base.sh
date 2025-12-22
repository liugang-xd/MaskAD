
OUTPUT_DIR='output_dir/'
DATA_PATH=''

 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port 19161 train.py \
        --data_path ${DATA_PATH} --output_dir ${OUTPUT_DIR}  \
        --model vit_base_patch16_224_8k_vocab  \
        --batch_size 100 --lr 1.5e-3 --warmup_epochs 10 --epochs 600 \
        --clip_grad 3.0 --drop_path 0.1 --layer_scale_init_value 0.1 \
        --imagenet_default_mean_and_std