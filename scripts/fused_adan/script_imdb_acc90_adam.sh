clear
echo "-----------------------Benchmark start------------------------"
python ./benchmark/imdb_bert_base.py \
        --n_epochs 2 --warmup 160 \
        --lr 1e-5 --wd 0.01 \
        --optimizer adamw \
        --log_file_name imdb_adamw_lr1e-5_wd1e-2_wm160_ep2_acc90 \
        --target_val_acc 0.90
echo "--------------------------adamw done--------------------------"
python ./benchmark/imdb_bert_base.py \
        --n_epochs 2 --warmup 160 \
        --lr 1e-5 --wd 0.01 \
        --optimizer adam \
        --log_file_name imdb_adam_lr1e-5_wd1e-2_wm160_ep2_acc90 \
        --target_val_acc 0.90
echo "--------------------------adam done--------------------------"
# Plot the results
python ./benchmark/plot_loss_accuracy.py IMDB_acc92

# adamw:
#     lr=1e-5, wd=0.01, warmup=160
#     betas=(0.9, 0.999), eps=1e-8
# adan:
#     lr=5e-5, wd=0.005, warmup=320
#     betas=(0.98, 0.92, 0.99), eps=1e-8
# adam:
#     lr=1e-5, wd=0.01, warmup=160
#     betas=(0.9, 0.999), eps=1e-8