clear

# echo "-----------------------huggingface 10------------------------"
# accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc.yaml \
#          ./benchmark/volvo_bert_base_accelerate_epoch_val.py \
#         --n_epochs 3 --warmup 50 \
#         --lr 1e-4 --wd 0.01 \
#         --optimizer adan \
#         --log_file_name trf_10_hf_FP32 \
#         --fused_optimizer True \
#         --batch_size 16 \
#         --seed 38 \
#         --module_type 0 \
#         --trf_level 10
# echo "-----------------------huggingface 8------------------------"
# accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc.yaml \
#          ./benchmark/volvo_bert_base_accelerate_epoch_val.py \
#         --n_epochs 3 --warmup 50 \
#         --lr 1e-4 --wd 0.01 \
#         --optimizer adan \
#         --log_file_name trf_8_hf_FP32 \
#         --fused_optimizer True \
#         --batch_size 16 \
#         --seed 38 \
#         --module_type 0 \
#         --trf_level 8
echo "-----------------------huggingface 6------------------------"
accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc.yaml \
         ./benchmark/volvo_bert_base_accelerate_epoch_val.py \
        --n_epochs 3 --warmup 50 \
        --lr 1e-4 --wd 0.01 \
        --optimizer adan \
        --log_file_name trf_6_hf_FP32 \
        --fused_optimizer True \
        --batch_size 16 \
        --seed 38 \
        --module_type 0 \
        --trf_level 6
echo "-----------------------huggingface 4------------------------"
accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc.yaml \
         ./benchmark/volvo_bert_base_accelerate_epoch_val.py \
        --n_epochs 3 --warmup 50 \
        --lr 1e-4 --wd 0.01 \
        --optimizer adan \
        --log_file_name trf_4_hf_FP32 \
        --fused_optimizer True \
        --batch_size 16 \
        --seed 38 \
        --module_type 0 \
        --trf_level 4
echo "-----------------------huggingface 2------------------------"
accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc.yaml \
         ./benchmark/volvo_bert_base_accelerate_epoch_val.py \
        --n_epochs 3 --warmup 50 \
        --lr 1e-4 --wd 0.01 \
        --optimizer adan \
        --log_file_name trf_2_hf_FP32 \
        --fused_optimizer True \
        --batch_size 16 \
        --seed 38 \
        --module_type 0 \
        --trf_level 2
echo "-----------------------huggingface done, light start------------------------"
echo "-----------------------lightseq 10------------------------"
accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc.yaml \
         ./benchmark/volvo_bert_base_accelerate_epoch_val.py \
        --n_epochs 3 --warmup 50 \
        --lr 1e-4 --wd 0.01 \
        --optimizer adan \
        --log_file_name trf_10_ls_FP32 \
        --fused_optimizer True \
        --batch_size 16 \
        --seed 38 \
        --module_type 1 \
        --trf_level 10
echo "-----------------------lightseq 8------------------------"
accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc.yaml \
         ./benchmark/volvo_bert_base_accelerate_epoch_val.py \
        --n_epochs 3 --warmup 50 \
        --lr 1e-4 --wd 0.01 \
        --optimizer adan \
        --log_file_name trf_8_ls_FP32 \
        --fused_optimizer True \
        --batch_size 16 \
        --seed 38 \
        --module_type 1 \
        --trf_level 8
echo "-----------------------lightseq 6------------------------"
accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc.yaml \
         ./benchmark/volvo_bert_base_accelerate_epoch_val.py \
        --n_epochs 3 --warmup 50 \
        --lr 1e-4 --wd 0.01 \
        --optimizer adan \
        --log_file_name trf_6_ls_FP32 \
        --fused_optimizer True \
        --batch_size 16 \
        --seed 38 \
        --module_type 1 \
        --trf_level 6
echo "-----------------------lightseq 4------------------------"
accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc.yaml \
         ./benchmark/volvo_bert_base_accelerate_epoch_val.py \
        --n_epochs 3 --warmup 50 \
        --lr 1e-4 --wd 0.01 \
        --optimizer adan \
        --log_file_name trf_4_ls_FP32 \
        --fused_optimizer True \
        --batch_size 16 \
        --seed 38 \
        --module_type 1 \
        --trf_level 4
echo "-----------------------lightseq 2------------------------"
accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc.yaml \
         ./benchmark/volvo_bert_base_accelerate_epoch_val.py \
        --n_epochs 3 --warmup 50 \
        --lr 1e-4 --wd 0.01 \
        --optimizer adan \
        --log_file_name trf_2_ls_FP32 \
        --fused_optimizer True \
        --batch_size 16 \
        --seed 38 \
        --module_type 1 \
        --trf_level 2

# adamw:
#     lr=1e-5, wd=0.01, warmup=160
#     betas=(0.9, 0.999), eps=1e-8
# adan:
#     lr=5e-5, wd=0.005, warmup=320
#     betas=(0.98, 0.92, 0.99), eps=1e-8
# adam:
#     lr=1e-5, wd=0.01, warmup=160
#     betas=(0.9, 0.999), eps=1e-8