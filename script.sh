echo "-----------------------Benchmark start------------------------"
python glue_mrpc_bert_based.py \
        --n_epochs 10 --warmup 160 \
        --lr 1e-5 --wd 0.01 \
        --optimizer adamw \
        --log_file_name adamw_1e-5_0.01_160_6
        --target_val_acc 0.85
echo "--------------------------adamw done--------------------------"
python glue_mrpc_bert_based.py \
        --n_epochs 10 --warmup 160 \
        --lr 1e-5 --wd 0.01 \
        --optimizer adam \
        --log_file_name adam_1e-5_0.01_160_6
        --target_val_acc 0.85
echo "--------------------------adam done--------------------------"
python glue_mrpc_bert_based.py \
        --n_epochs 10 --warmup 320 \
        --lr 5e-5 --wd 0.005 \
        --optimizer adan \
        --log_file_name adan_5e-5_0.005_320_6
        --target_val_acc 0.85
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