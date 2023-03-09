clear
echo "Adan Micro Benchmark with ncu, size=2048, n_params=30"
echo "===================="
echo "foreach=False, fused=True"
ncu -o ./ncu_log/adan_each_f_fused_t_multi --target-processes all -k regex:multi --nvtx --set full\
        python ./benchmark/adan_micro_benchmark.py \
        --foreach True --fused True --size 2048 --n_params 30
echo "===================="
echo "Done"