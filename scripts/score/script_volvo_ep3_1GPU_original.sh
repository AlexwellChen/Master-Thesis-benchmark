clear
echo "-----------------------adamw start------------------------"
accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc.yaml \
         ./benchmark/volvo_bert_base_accelerate.py \
        --n_epochs 3 --warmup 2000 \
        --lr 5e-5 --wd 0.01 \
        --optimizer adamw \
        --log_file_name volvo_adamw_unfused_lr5e-5_wd1e-2_wm2000_ep3_unmixed_hf \
        --fused_optimizer False \
        --batch_size 16 \
        --seed 38 \
        --module_type 0 