clear
echo "-----------------------Benchmark start------------------------"
accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc_mix.yaml \
         ./benchmark/volvo_bert_base_accelerate.py \
        --n_epochs 3 --warmup 50 \
        --lr 1e-4 --wd 0.01 \
        --optimizer adan \
        --log_file_name volvo_adan_fused_lr1e-4_wd1e-2_wm50_ep3_mixed_ls \
        --fused_optimizer True \
        --batch_size 16 \
        --seed 38 \
        --module_type 1 \
echo "-----------------------LightSeq Adan done------------------------"

# adamw:
#     lr=1e-5, wd=0.01, warmup=160
#     betas=(0.9, 0.999), eps=1e-8
# adan:
#     lr=5e-5, wd=0.005, warmup=320
#     betas=(0.98, 0.92, 0.99), eps=1e-8
# adam:
#     lr=1e-5, wd=0.01, warmup=160
#     betas=(0.9, 0.999), eps=1e-8