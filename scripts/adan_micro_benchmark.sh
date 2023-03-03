clear
echo "Adan Micro Benchmark with nvprof"
echo "===================="
echo "foreach=False, fused=False"
nvprof -o ./nvprof_log/adan_each_f_fused_f.nvvp python ./benchmark/adan_micro_benchmark.py \
        --foreach False --fused False
echo "===================="
echo "foreach=True, fused=False"
nvprof -o ./nvprof_log/adan_each_t_fused_f.nvvp python ./benchmark/adan_micro_benchmark.py \
        --foreach True --fused False
echo "===================="
echo "foreach=False, fused=True"
nvprof -o ./nvprof_log/adan_each_f_fused_t.nvvp python ./benchmark/adan_micro_benchmark.py \
        --foreach False --fused True
echo "===================="
echo "foreach=True, fused=True"
nvprof -o ./nvprof_log/adan_each_t_fused_t.nvvp python ./benchmark/adan_micro_benchmark.py \
        --foreach True --fused True
echo "===================="
echo "Done"
