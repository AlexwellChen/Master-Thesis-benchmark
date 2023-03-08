clear
echo "-----------------------Benchmark start------------------------"
ncu -o ./ncu_log/imdb_acc90_adan_fp32 --target-processes all -k "adan_cuda_kernel" --nvtx --set full\
        accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc.yaml \
         ./benchmark/imdb_bert_base_accelerate.py \
        --n_epochs 2 --warmup 50 \
        --lr 1e-4 --wd 0.01 \
        --optimizer adan \
        --log_file_name imdb_adan_fused_lr1e-4_wd1e-2_wm50_ep2_acc90_mix \
        --target_val_acc 0.90 \
        --fused_optimizer True \
        --foreach False
echo "--------------------------fp32 done--------------------------"
ncu -o ./ncu_log/imdb_acc90_adan_fp16 --target-processes all -k "adan_cuda_kernel" --nvtx --set full\
        accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc_mix.yaml \
         ./benchmark/imdb_bert_base_accelerate.py \
        --n_epochs 2 --warmup 50 \
        --lr 1e-4 --wd 0.01 \
        --optimizer adan \
        --log_file_name imdb_adan_fused_lr1e-4_wd1e-2_wm50_ep2_acc90_mix \
        --target_val_acc 0.90 \
        --fused_optimizer True \
        --foreach False
echo "-----------------------fp16 done------------------------"
# Plot the results
# python ./benchmark/plot_loss_accuracy.py IMDB_acc92
# adamw:
#     lr=1e-5, wd=0.01, warmup=160
#     betas=(0.9, 0.999), eps=1e-8
# adan:
#     lr=5e-5, wd=0.005, warmup=320
#     betas=(0.98, 0.92, 0.99), eps=1e-8
# adam:
#     lr=1e-5, wd=0.01, warmup=160
#     betas=(0.9, 0.999), eps=1e-8