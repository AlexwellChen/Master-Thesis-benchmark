clear
echo "-----------------------Benchmark start------------------------"
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu  --capture-range=cudaProfilerApi  --cudabacktrace=true -x true \
        -o ./nsys_log/imdb_adan_lr1e-4_wd1e-2_wm50_one_step_mix_nvtx \
        accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc_mix.yaml \
         ./benchmark/imdb_bert_base_accelerate_nvtx.py \
        --n_epochs 3 --warmup 50 \
        --lr 1e-4 --wd 0.01 \
        --optimizer adan \
        --log_file_name imdb_adan_fused_lr1e-4_wd1e-2_wm50_ep3_mix_nvtx \
        --fused_optimizer True \
        --seed 38
echo "-----------------------Mix huggingface one step done------------------------"
