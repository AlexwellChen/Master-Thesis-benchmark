clear
echo "-----------------------Benchmark start------------------------"
accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc.yaml \
         ./benchmark/imdb_bert_base_accelerate.py \
        --n_epochs 3 --warmup 50 \
        --lr 1e-4 --wd 0.01 \
        --optimizer adan \
        --log_file_name imdb_adan_fused_lr1e-4_wd1e-2_wm50_ep3_unmixed_lightseq \
        --fused_optimizer True \
        --batch_size 16 \
        --seed 38 \
        --module_type 1 
echo "-----------------------lightseq done------------------------"