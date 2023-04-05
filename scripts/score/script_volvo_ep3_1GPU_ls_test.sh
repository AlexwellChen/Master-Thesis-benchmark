clear
echo "-----------------------LightSeq Adan FP16 start------------------------"
accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc_mix.yaml \
         ./benchmark/volvo_bert_base_accelerate.py \
        --n_epochs 3 --warmup 500 \
        --lr 1e-4 --wd 0.01 \
        --optimizer adan \
        --log_file_name volvo_adan_fused_lr1e-4_wd1e-2_wm500_ep3_FP16_ls_test \
        --fused_optimizer True \
        --batch_size 16 \
        --seed 38 \
        --module_type 1
