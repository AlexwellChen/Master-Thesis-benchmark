clear
echo "-----------------------Benchmark start------------------------"
python ./benchmark/imdb_bert_base.py \
        --n_epochs 2 --warmup 160 \
        --lr 1e-5 --wd 0.01 \
        --optimizer adam \
        --log_file_name imdb_adam_lr1e-5_wd1e-2_wm160_ep2_acc93 \
        --target_val_acc 0.93 \
        --seed 57
echo "--------------------------adam done--------------------------"
python ./benchmark/imdb_bert_base.py \
        --n_epochs 2 --warmup 50 \
        --lr 1e-4 --wd 0.01 \
        --optimizer adan \
        --log_file_name imdb_adan_lr1e-4_wd1e-2_wm50_ep2_acc93 \
        --target_val_acc 0.93 \
        --seed 57
echo "--------------------------adan done--------------------------"
python ./benchmark/imdb_bert_base.py \
        --n_epochs 2 --warmup 50 \
        --lr 1e-4 --wd 0.01 \
        --optimizer adan \
        --log_file_name imdb_adan_fused_lr1e-4_wd1e-2_wm50_ep2_acc93 \
        --target_val_acc 0.93 \
        --fused_optimizer True \
        --seed 57
echo "-----------------------fused adan done------------------------"
accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc_mix.yaml \
         ./benchmark/imdb_bert_base_accelerate.py \
        --n_epochs 2 --warmup 50 \
        --lr 1e-4 --wd 0.01 \
        --optimizer adan \
        --log_file_name imdb_adan_fused_lr1e-4_wd1e-2_wm50_ep2_acc93_mix \
        --target_val_acc 0.93 \
        --fused_optimizer True \
        --seed 57
echo "-----------------------adan done------------------------"
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