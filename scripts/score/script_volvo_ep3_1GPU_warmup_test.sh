clear
echo "-----------------------Warmup 2000------------------------"
accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc_mix.yaml \
         ./benchmark/volvo_bert_base_accelerate_epoch_val.py \
        --n_epochs 3 --warmup 2500 \
        --lr 5e-5 --wd 0.01 \
        --optimizer adan \
        --log_file_name volvo_adan_fused_lr5e-5_wd1e-2_wm2000_ep3_FP16_ls \
        --fused_optimizer True \
        --batch_size 16 \
        --seed 38 \
        --module_type 1
# echo "-----------------------Warmup 2500------------------------"
# accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc_mix.yaml \
#          ./benchmark/volvo_bert_base_accelerate_epoch_val.py \
#         --n_epochs 3 --warmup 2500 \
#         --lr 5e-5 --wd 0.01 \
#         --optimizer adan \
#         --log_file_name volvo_adan_fused_lr5e-5_wd1e-2_wm2500_ep3_FP16_ls \
#         --fused_optimizer True \
#         --batch_size 16 \
#         --seed 38 \
#         --module_type 1
# echo "-----------------------Warmup 1000------------------------"
# accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc_mix.yaml \
#          ./benchmark/volvo_bert_base_accelerate_epoch_val.py \
#         --n_epochs 3 --warmup 1000 \
#         --lr 5e-5 --wd 0.01 \
#         --optimizer adan \
#         --log_file_name volvo_adan_fused_lr5e-5_wd1e-2_wm1000_ep3_FP16_ls \
#         --fused_optimizer True \
#         --batch_size 16 \
#         --seed 38 \
#         --module_type 1
# echo "-----------------------Warmup 1500------------------------"
# accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc_mix.yaml \
#          ./benchmark/volvo_bert_base_accelerate_epoch_val.py \
#         --n_epochs 3 --warmup 1500 \
#         --lr 5e-5 --wd 0.01 \
#         --optimizer adan \
#         --log_file_name volvo_adan_fused_lr5e-5_wd1e-2_wm1500_ep3_FP16_ls \
#         --fused_optimizer True \
#         --batch_size 16 \
#         --seed 38 \
#         --module_type 1

