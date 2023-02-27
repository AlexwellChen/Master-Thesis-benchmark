clear
echo "-----------------------Benchmark start------------------------"
python ./benchmark/imdb_bert_base_adan.py \
        --n_epochs 2 --warmup 50 \
        --lr 1e-4 --wd 0.01 \
        --optimizer adan \
        --target_val_acc 0.90
echo "----------------------------- done----------------------------"
# Plot the results
python ./benchmark/plot_loss_accuracy.py IMDB_acc90
# adamw:
#     lr=1e-5, wd=0.01, warmup=160
#     betas=(0.9, 0.999), eps=1e-8
# adan:
#     lr=5e-5, wd=0.005, warmup=320
#     betas=(0.98, 0.92, 0.99), eps=1e-8
# adam:
#     lr=1e-5, wd=0.01, warmup=160
#     betas=(0.9, 0.999), eps=1e-8