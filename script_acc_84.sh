echo "-----------------------Benchmark start------------------------"
python glue_mrpc_bert_based.py \
        --n_epochs 10 --warmup 160 \
        --lr 1e-5 --wd 0.01 \
        --optimizer adamw \
        --log_file_name adamw_lr1e-5_wd1e-2_160_acc84 \
        --target_val_acc 0.84
echo "--------------------------adamw done--------------------------"
python glue_mrpc_bert_based.py \
        --n_epochs 10 --warmup 160 \
        --lr 1e-5 --wd 0.01 \
        --optimizer adam \
        --log_file_name adam_lr1e-5_wd1e-2_160_acc84 \
        --target_val_acc 0.84
echo "--------------------------adam done--------------------------"
python glue_mrpc_bert_based.py \
        --n_epochs 10 --warmup 320 \
        --lr 5e-5 --wd 0.005 \
        --optimizer adan \
        --log_file_name adan_lr5e-5_wd5e-3_320_acc84 \
        --target_val_acc 0.84
echo "--------------------------adan done--------------------------"

# adamw:
#     lr=1e-5, wd=0.01, warmup=160
#     betas=(0.9, 0.999), eps=1e-8
# adan:
#     lr=5e-5, wd=0.005, warmup=320
#     betas=(0.98, 0.92, 0.99), eps=1e-8
# adam:
#     lr=1e-5, wd=0.01, warmup=160
#     betas=(0.9, 0.999), eps=1e-8