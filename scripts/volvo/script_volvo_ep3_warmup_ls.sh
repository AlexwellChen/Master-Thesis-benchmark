clear

# echo "-----------------------lightseq 500------------------------"
# accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc_mix.yaml \
#          ./benchmark/volvo_bert_base_accelerate_epoch_val.py \
#         --n_epochs 3 --warmup 500 \
#         --lr 1e-4 --wd 0.01 \
#         --optimizer adan \
#         --log_file_name warmup_500_ls \
#         --fused_optimizer True \
#         --batch_size 16 \
#         --seed 38 \
#         --module_type 1 \
#         --trf_level 10
# echo "-----------------------lightseq 1000------------------------"
# accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc_mix.yaml \
#          ./benchmark/volvo_bert_base_accelerate_epoch_val.py \
#         --n_epochs 3 --warmup 1000 \
#         --lr 1e-4 --wd 0.01 \
#         --optimizer adan \
#         --log_file_name warmup_1000_ls \
#         --fused_optimizer True \
#         --batch_size 16 \
#         --seed 38 \
#         --module_type 1 \
#         --trf_level 10
# echo "-----------------------lightseq 1500------------------------"
# accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc_mix.yaml \
#          ./benchmark/volvo_bert_base_accelerate_epoch_val.py \
#         --n_epochs 3 --warmup 1500 \
#         --lr 1e-4 --wd 0.01 \
#         --optimizer adan \
#         --log_file_name warmup_1500_ls \
#         --fused_optimizer True \
#         --batch_size 16 \
#         --seed 38 \
#         --module_type 1 \
#         --trf_level 10
# echo "-----------------------lightseq 2000------------------------"
# accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc_mix.yaml \
#          ./benchmark/volvo_bert_base_accelerate_epoch_val.py \
#         --n_epochs 3 --warmup 2000 \
#         --lr 1e-4 --wd 0.01 \
#         --optimizer adan \
#         --log_file_name warmup_2000_ls \
#         --fused_optimizer True \
#         --batch_size 16 \
#         --seed 38 \
#         --module_type 1 \
#         --trf_level 10
# echo "-----------------------lightseq 2500------------------------"
# accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc_mix.yaml \
#          ./benchmark/volvo_bert_base_accelerate_epoch_val.py \
#         --n_epochs 3 --warmup 2500 \
#         --lr 1e-4 --wd 0.01 \
#         --optimizer adan \
#         --log_file_name warmup_2500_ls \
#         --fused_optimizer True \
#         --batch_size 16 \
#         --seed 38 \
#         --module_type 1 \
#         --trf_level 10
echo "-----------------------lightseq 3000------------------------"
accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc_mix.yaml \
         ./benchmark/volvo_bert_base_accelerate_epoch_val.py \
        --n_epochs 3 --warmup 3000 \
        --lr 1e-4 --wd 0.01 \
        --optimizer adan \
        --log_file_name warmup_3000_ls \
        --fused_optimizer True \
        --batch_size 16 \
        --seed 38 \
        --module_type 1 \
        --trf_level 10
echo "-----------------------lightseq 3500------------------------"
accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc_mix.yaml \
         ./benchmark/volvo_bert_base_accelerate_epoch_val.py \
        --n_epochs 3 --warmup 3500 \
        --lr 1e-4 --wd 0.01 \
        --optimizer adan \
        --log_file_name warmup_3500_ls \
        --fused_optimizer True \
        --batch_size 16 \
        --seed 38 \
        --module_type 1 \
        --trf_level 10
# adamw:
#     lr=1e-5, wd=0.01, warmup=160
#     betas=(0.9, 0.999), eps=1e-8
# adan:
#     lr=5e-5, wd=0.005, warmup=320
#     betas=(0.98, 0.92, 0.99), eps=1e-8
# adam:
#     lr=1e-5, wd=0.01, warmup=160
#     betas=(0.9, 0.999), eps=1e-8