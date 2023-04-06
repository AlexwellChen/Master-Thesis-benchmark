clear
echo "-----------------------LightSeq Adan start------------------------"
python ./benchmark/volvo_bert_base_hf_trainer.py \
        --n_epochs 3 --warmup 500 \
        --lr 1e-4 --wd 0.01 \
        --optimizer adan \
        --log_file_name volvo_adan_fused_lr1e-4_wd1e-2_wm500_ep3_FP16_ls_trainer \
        --fused_optimizer True \
        --batch_size 16 \
        --seed 38 \
        --module_type 1 \
        --fp16 True
echo "-----------------------Huggingface adamw start------------------------"
python ./benchmark/volvo_bert_base_hf_trainer.py \
        --n_epochs 3 --warmup 500 \
        --lr 5e-5 --wd 0.01 \
        --optimizer adamw \
        --log_file_name volvo_adamw_unfused_lr5e-5_wd1e-2_wm500_ep3_FP32_ls_trainer \
        --fused_optimizer False \
        --batch_size 16 \
        --seed 38 \
        --module_type 0 \
        --fp16 False
echo "-----------------------Huggingface adan start------------------------"
python ./benchmark/volvo_bert_base_hf_trainer.py \
        --n_epochs 3 --warmup 500 \
        --lr 5e-5 --wd 0.01 \
        --optimizer adan \
        --log_file_name volvo_adan_fused_lr1e-4_wd1e-2_wm500_ep3_FP32_hf_trainer \
        --fused_optimizer True \
        --batch_size 16 \
        --seed 38 \
        --module_type 0 \
        --fp16 True
