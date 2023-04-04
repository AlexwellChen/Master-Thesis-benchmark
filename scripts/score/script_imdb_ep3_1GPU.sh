clear
# echo "-----------------------IMDb_ep3_FP32 start------------------------"
# accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc.yaml \
#          ./benchmark/imdb_bert_base_accelerate_epoch_val.py \
#         --n_epochs 3 --warmup 50 \
#         --lr 1e-4 --wd 0.01 \
#         --optimizer adan \
#         --log_file_name IMDb_ep3_FP32 \
#         --fused_optimizer True \
#         --batch_size 16 \
#         --seed 38 \
#         --module_type 0
# echo "-----------------------IMDb_ep3_FP16 start------------------------"
# accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc_mix.yaml \
#          ./benchmark/imdb_bert_base_accelerate_epoch_val.py \
#         --n_epochs 3 --warmup 50 \
#         --lr 1e-4 --wd 0.01 \
#         --optimizer adan \
#         --log_file_name IMDb_ep3_FP16 \
#         --fused_optimizer True \
#         --batch_size 16 \
#         --seed 38 \
#         --module_type 0
# echo "-----------------------IMDb_ls_ep3_FP16 start------------------------"
# accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc_mix.yaml \
#          ./benchmark/imdb_bert_base_accelerate_epoch_val.py \
#         --n_epochs 3 --warmup 50 \
#         --lr 1e-4 --wd 0.01 \
#         --optimizer adan \
#         --log_file_name IMDb_ls_ep3_FP16 \
#         --fused_optimizer True \
#         --batch_size 16 \
#         --seed 38 \
#         --module_type 1 
echo "-----------------------adamw start------------------------"
accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc.yaml \
         ./benchmark/volvo_bert_base_accelerate.py \
        --n_epochs 3 --warmup 50 \
        --lr 5e-5 --wd 0.01 \
        --optimizer adamw \
        --log_file_name volvo_adamw_unfused_lr1e-5_wd1e-2_wm50_ep3_unmixed_hf \
        --fused_optimizer False \
        --batch_size 16 \
        --seed 38 \
        --module_type 0 
echo "-----------------------LightSeq Adan start------------------------"
accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc_mix.yaml \
         ./benchmark/volvo_bert_base_accelerate.py \
        --n_epochs 3 --warmup 50 \
        --lr 1e-4 --wd 0.01 \
        --optimizer adan \
        --log_file_name volvo_adan_fused_lr1e-4_wd1e-2_wm50_ep3_mixed_ls \
        --fused_optimizer True \
        --batch_size 16 \
        --seed 38 \
        --module_type 1
# echo "-----------------------IMDb_ls_ep3_FP16_2GPU start------------------------"
# accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc_mix_DDP_2gpu.yaml \
#          ./benchmark/imdb_bert_base_accelerate_epoch_val.py \
#         --n_epochs 3 --warmup 50 \
#         --lr 1e-4 --wd 0.01 \
#         --optimizer adan \
#         --log_file_name IMDb_ls_ep3_FP16_2GPU \
#         --fused_optimizer True \
#         --batch_size 16 \
#         --seed 38 \
#         --module_type 1 
# echo "-----------------------IMDb_ls_ep3_FP16_4GPU start------------------------"
# accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc_mix_DDP_4gpu.yaml \
#          ./benchmark/imdb_bert_base_accelerate_epoch_val.py \
#         --n_epochs 3 --warmup 50 \
#         --lr 1e-4 --wd 0.01 \
#         --optimizer adan \
#         --log_file_name IMDb_ls_ep3_FP16_4GPU \
#         --fused_optimizer True \
#         --batch_size 16 \
#         --seed 38 \
#         --module_type 1 

# Plot the results
# python ./benchmark/plot_loss_accuracy.py IMDB_acc90

# adamw:
#     lr=1e-5, wd=0.01, warmup=160
#     betas=(0.9, 0.999), eps=1e-8
# adan:
#     lr=5e-5, wd=0.005, warmup=320
#     betas=(0.98, 0.92, 0.99), eps=1e-8
# adam:
#     lr=1e-5, wd=0.01, warmup=160
#     betas=(0.9, 0.999), eps=1e-8