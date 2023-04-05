clear
echo "-----------------------LightSeq Adan FP16 start------------------------"
accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc_mix.yaml \
         ./benchmark/volvo_bert_base_accelerate.py \
        --n_epochs 3 --warmup 500 \
        --lr 1e-4 --wd 0.01 \
        --optimizer adan \
        --log_file_name volvo_adan_fused_lr1e-4_wd1e-2_wm500_ep3_FP16_ls \
        --fused_optimizer True \
        --batch_size 16 \
        --seed 38 \
        --module_type 1
echo "-----------------------Huggingface adamw FP32 start------------------------"
accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc.yaml \
         ./benchmark/volvo_bert_base_accelerate.py \
        --n_epochs 3 --warmup 500 \
        --lr 5e-5 --wd 0.01 \
        --optimizer adamw \
        --log_file_name volvo_adamw_unfused_lr5e-5_wd1e-2_wm500_ep3_FP32_hf \
        --fused_optimizer False \
        --batch_size 16 \
        --seed 38 \
        --module_type 0 
echo "-----------------------Huggingface adan FP16 start------------------------"
accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc_mix.yaml \
         ./benchmark/volvo_bert_base_accelerate.py \
        --n_epochs 3 --warmup 500 \
        --lr 1e-4 --wd 0.01 \
        --optimizer adan \
        --log_file_name volvo_adan_fused_lr1e-4_wd1e-2_wm500_ep3_FP16_hf \
        --fused_optimizer True \
        --batch_size 16 \
        --seed 38 \
        --module_type 0 
