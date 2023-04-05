clear
echo "-----------------------LightSeq Adan FP16 start------------------------"
accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc_mix.yaml \
         ./benchmark/volvo_bert_base_accelerate_epoch_val.py \
        --n_epochs 3 --warmup 2000 \
        --lr 7e-5 --wd 0.01 \
        --optimizer adan \
        --log_file_name volvo_adan_fused_lr7e-5_wd1e-2_wm2000_ep3_FP16_ls \
        --fused_optimizer True \
        --batch_size 16 \
        --seed 38 \
        --module_type 1
# echo "-----------------------Huggingface adamw FP32 start------------------------"
# accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc.yaml \
#          ./benchmark/volvo_bert_base_accelerate_epoch_val.py \
#         --n_epochs 3 --warmup 2000 \
#         --lr 5e-5 --wd 0.01 \
#         --optimizer adamw \
#         --log_file_name volvo_adamw_unfused_lr5e-5_wd1e-2_wm2000_ep3_FP32_hf \
#         --fused_optimizer False \
#         --batch_size 16 \
#         --seed 38 \
#         --module_type 0 
# echo "-----------------------Huggingface adan FP16 start------------------------"
# accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc_mix.yaml \
#          ./benchmark/volvo_bert_base_accelerate_epoch_val.py \
#         --n_epochs 3 --warmup 2000 \
#         --lr 5e-5 --wd 0.01 \
#         --optimizer adan \
#         --log_file_name volvo_adan_fused_lr5e-5_wd1e-2_wm2000_ep3_FP16_hf \
#         --fused_optimizer True \
#         --batch_size 16 \
#         --seed 38 \
#         --module_type 0 
