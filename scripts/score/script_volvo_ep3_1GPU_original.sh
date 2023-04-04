clear
# echo "-----------------------adamw start------------------------"
# accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc.yaml \
#          ./benchmark/volvo_bert_base_accelerate.py \
#         --n_epochs 3 --warmup 2000 \
#         --lr 5e-5 --wd 0.01 \
#         --optimizer adamw \
#         --log_file_name volvo_adamw_unfused_lr5e-5_wd1e-2_wm2000_ep3_unmixed_hf \
#         --fused_optimizer False \
#         --batch_size 16 \
#         --seed 38 \
#         --module_type 0 
# echo "-----------------------HF FP32 unfused Adan start------------------------"
# accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc.yaml \
#          ./benchmark/volvo_bert_base_accelerate.py \
#         --n_epochs 3 --warmup 2000 \
#         --lr 1e-4 --wd 0.01 \
#         --optimizer adan \
#         --log_file_name volvo_adan_unfused_lr1e-4_wd1e-2_wm2000_ep3_unmix_hf \
#         --fused_optimizer False \
#         --batch_size 16 \
#         --seed 38 \
#         --module_type 0
# echo "-----------------------Lightseq adamw start------------------------"
# accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc.yaml \
#          ./benchmark/volvo_bert_base_accelerate.py \
#         --n_epochs 3 --warmup 2000 \
#         --lr 5e-5 --wd 0.01 \
#         --optimizer adamw \
#         --log_file_name volvo_adamw_unfused_lr5e-5_wd1e-2_wm2000_ep3_unmixed_ls \
#         --fused_optimizer False \
#         --batch_size 16 \
#         --seed 38 \
#         --module_type 1
echo "-----------------------HF FP16 Adan start------------------------"
accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc_mix.yaml \
         ./benchmark/volvo_bert_base_accelerate.py \
        --n_epochs 3 --warmup 2000 \
        --lr 1e-4 --wd 0.01 \
        --optimizer adan \
        --log_file_name volvo_adan_fused_lr1e-4_wd1e-2_wm2000_ep3_mixed_hf \
        --fused_optimizer Ture \
        --batch_size 16 \
        --seed 38 \
        --module_type 0
# echo "-----------------------LS FP32 Adan start------------------------"
# accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc.yaml \
#          ./benchmark/volvo_bert_base_accelerate.py \
#         --n_epochs 3 --warmup 2000 \
#         --lr 1e-4 --wd 0.01 \
#         --optimizer adan \
#         --log_file_name volvo_adan_fused_lr1e-4_wd1e-2_wm2000_ep3_unmix_ls \
#         --fused_optimizer True \
#         --batch_size 16 \
#         --seed 38 \
#         --module_type 1