clear
echo "Adan Micro Benchmark with ncu, size=4096, n_params=30, vector single"
echo "===================="
echo "foreach=False, fused=True"
ncu -o ./ncu_log/adan_each_f_fused_t_vector --target-processes all -k "adan_cuda_kernel" --nvtx --set full\
        python ./benchmark/adan_micro_benchmark.py \
        --foreach False --fused True --size 4096 --n_params 30
echo "===================="
echo "Adan Micro Benchmark with ncu, size=4096, n_params=30, ILP multi"
echo "===================="
echo "foreach=True, fused=True"
ncu -o ./ncu_log/adan_each_t_fused_t_multi --target-processes all -k regex:multi --nvtx --set full\
        python ./benchmark/adan_micro_benchmark.py \
        --foreach True --fused True --size 4096 --n_params 30
echo "===================="
echo "Uninstall vector Adan"
pip uninstall adan
echo "Install warp Adan"
pip install git+https://github.com/AlexwellChen/Adan.git
echo "===================="
echo "Adan Micro Benchmark with ncu, size=4096, n_params=30, warp"
echo "===================="
echo "foreach=False, fused=True"
ncu -o ./ncu_log/adan_each_f_fused_t_warp --target-processes all -k "adan_cuda_kernel" --nvtx --set full \
        python ./benchmark/adan_micro_benchmark.py \
        --foreach False --fused True --size 4096 --n_params 30
echo "===================="
echo "Uninstall warp Adan"
pip uninstall adan
echo "Install vector Adan"
pip install git+https://github.com/sail-sg/Adan.git
