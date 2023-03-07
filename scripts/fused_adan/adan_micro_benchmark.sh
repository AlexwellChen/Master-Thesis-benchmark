clear
echo "Adan Micro Benchmark with nvprof, size=2048, n_params=30"
echo "===================="
echo "foreach=False, fused=False"
nvprof -f -o ./nvprof_log/adan_each_f_fused_f.nvvp python ./benchmark/adan_micro_benchmark.py \
        --foreach False --fused False --size 2048 --n_params 30
echo "===================="
echo "foreach=True, fused=False"
nvprof -f -o ./nvprof_log/adan_each_t_fused_f.nvvp python ./benchmark/adan_micro_benchmark.py \
        --foreach True --fused False --size 2048 --n_params 30
echo "===================="
echo "foreach=False, fused=True"
nvprof -f -o ./nvprof_log/adan_each_f_fused_t.nvvp python ./benchmark/adan_micro_benchmark.py \
        --foreach False --fused True --size 2048 --n_params 30
echo "===================="
echo "foreach=True, fused=True"
nvprof -f -o ./nvprof_log/adan_each_t_fused_t.nvvp python ./benchmark/adan_micro_benchmark.py \
        --foreach True --fused True --size 2048 --n_params 30
echo "===================="
echo "Done"
