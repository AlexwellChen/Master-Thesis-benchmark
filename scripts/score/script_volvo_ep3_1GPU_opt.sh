clear
echo "-----------------------LightSeq Adan start------------------------"
accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc_mix.yaml \
         ./benchmark/volvo_bert_base_accelerate.py \
        --n_epochs 3 --warmup 50 \
        --lr 5e-5 --wd 0.01 \
        --optimizer adan \
        --log_file_name volvo_adan_fused_lr5e-5_wd1e-2_wm50_ep3_mixed_ls \
        --fused_optimizer True \
        --batch_size 16 \
        --seed 38 \
        --module_type 1