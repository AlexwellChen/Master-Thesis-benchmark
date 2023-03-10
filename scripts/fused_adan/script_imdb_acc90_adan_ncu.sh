clear
pip install git+https://github.com/AlexwellChen/Adan.git
echo "-----------------------Benchmark start------------------------"
ncu -o ./ncu_log/imdb_acc90_adan_vector --target-processes all -k "adan_cuda_kernel" --nvtx --set full -f \
        python ./benchmark/imdb_bert_base_nvtx.py \
        --n_epochs 2 --warmup 50 \
        --lr 1e-4 --wd 0.01 \
        --optimizer adan \
        --log_file_name imdb_adan_lr1e-4_wd1e-2_wm50_ep2_acc90_nvtx \
        --target_val_acc 0.90 --foreach False --batch_size 8 --fused_optimizer True
echo "--------------------------vector done--------------------------"
ncu -o ./ncu_log/imdb_acc90_adan_ILP --target-processes all -k regex:multi --nvtx --set full -f \
        python ./benchmark/imdb_bert_base_nvtx.py \
        --n_epochs 2 --warmup 50 \
        --lr 1e-4 --wd 0.01 \
        --optimizer adan \
        --log_file_name imdb_adan_lr1e-4_wd1e-2_wm50_ep2_acc90_nvtx \
        --target_val_acc 0.90 --foreach True --batch_size 8 --fused_optimizer True
echo "--------------------------ILP done--------------------------"
pip uninstall adan
pip install git+https://github.com/AlexwellChen/Adan.git@warp
ncu -o ./ncu_log/imdb_acc90_adan_warp --target-processes all -k "adan_cuda_kernel" --nvtx --set full -f \
        python ./benchmark/imdb_bert_base_nvtx.py \
        --n_epochs 2 --warmup 50 \
        --lr 1e-4 --wd 0.01 \
        --optimizer adan \
        --log_file_name imdb_adan_lr1e-4_wd1e-2_wm50_ep2_acc90_ncu \
        --target_val_acc 0.90 --foreach False --batch_size 8 --fused_optimizer True
echo "----------------------------warp done---------------------------"
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