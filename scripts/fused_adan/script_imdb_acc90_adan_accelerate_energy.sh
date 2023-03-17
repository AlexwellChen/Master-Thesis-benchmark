clear
echo "-----------------------Benchmark start------------------------"
accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc.yaml \
         ./benchmark/imdb_bert_base_accelerate_energy.py \
        --n_epochs 2 --warmup 50 \
        --lr 1e-4 --wd 0.01 \
        --optimizer adan \
        --log_file_name multi_unfused_adan \
        --target_val_acc 0.90 \
        --fused_optimizer False \
        --foreach True \
        --module_type 0 
echo "--------------------------adan done--------------------------"
accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc.yaml \
         ./benchmark/imdb_bert_base_accelerate_energy.py \
        --n_epochs 2 --warmup 50 \
        --lr 1e-4 --wd 0.01 \
        --optimizer adan \
        --log_file_name multi_fused_adan \
        --target_val_acc 0.90 \
        --fused_optimizer True \
        --foreach True \
        --module_type 0 
echo "-----------------------fused adan done------------------------"
accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc.yaml \
         ./benchmark/imdb_bert_base_accelerate_energy.py \
        --n_epochs 2 --warmup 50 \
        --lr 1e-4 --wd 0.01 \
        --optimizer adan \
        --log_file_name single_unfused \
        --target_val_acc 0.90 \
        --fused_optimizer False \
        --foreach False \
        --module_type 0 
echo "------------------------single adan done--------------------------"
python ./benchmark/imdb_bert_base_energy.py \
        --n_epochs 2 --warmup 50 \
        --lr 1e-4 --wd 0.01 \
        --optimizer adan \
        --log_file_name single_fused \
        --target_val_acc 0.90 \
        --fused_optimizer True \
        --foreach False \
        --module_type 0 
echo "-----------------------fused single adan done------------------------"
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