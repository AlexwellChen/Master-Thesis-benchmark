clear
echo "-----------------------Benchmark start------------------------"
python ./benchmark/imdb_bert_base.py \
        --n_epochs 2 --warmup 50 \
        --lr 1e-4 --wd 0.01 \
        --optimizer adan \
        --log_file_name imdb_adan_lr1e-4_wd1e-2_wm50_ep2_acc90 \
        --target_val_acc 0.90
echo "--------------------------adan done--------------------------"
python ./benchmark/imdb_bert_base.py \
        --n_epochs 2 --warmup 50 \
        --lr 1e-4 --wd 0.01 \
        --optimizer adan \
        --log_file_name imdb_adan_fused_lr1e-4_wd1e-2_wm50_ep2_acc90 \
        --target_val_acc 0.90 \
        --fused_optimizer True
echo "--------------------------adan done--------------------------"