clear
# echo "-----------------------LightSeq Adan start------------------------"
# accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc_mix.yaml \
#          ./benchmark/volvo_bert_base_accelerate.py \
#         --n_epochs 3 --warmup 2000 \
#         --lr 1e-4 --wd 0.01 \
#         --optimizer adan \
#         --log_file_name volvo_adan_fused_lr1e-4_wd1e-2_wm2000_ep3_mixed_ls \
#         --fused_optimizer True \
#         --batch_size 16 \
#         --seed 38 \
#         --module_type 1
echo "-----------------------HF Adan start------------------------"
accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc.yaml \
         ./benchmark/volvo_bert_base_accelerate.py \
        --n_epochs 3 --warmup 2000 \
        --lr 1e-4 --wd 0.01 \
        --optimizer adan \
        --log_file_name volvo_adan_unfused_lr1e-4_wd1e-2_wm2000_ep3_unmix_hf \
        --fused_optimizer False \
        --batch_size 16 \
        --seed 38 \
        --module_type 0