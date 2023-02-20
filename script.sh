python glue_mrpc_bert_based.py \
        --n_epochs 10 --warmup 160 \
        --lr 1e-5 --wd 0.01 \
        --optimizer adamw \
        --log_file_name adamw_1e-5_0.01_160_6

python glue_mrpc_bert_based.py \
        --n_epochs 10 --warmup 160 \
        --lr 1e-5 --wd 0.01 \
        --optimizer adam \
        --log_file_name adam_1e-5_0.01_160_6

python glue_mrpc_bert_based.py \
        --n_epochs 10 --warmup 320 \
        --lr 5e-5 --wd 0.005 \
        --optimizer adan \
        --log_file_name adan_5e-5_0.005_320_6