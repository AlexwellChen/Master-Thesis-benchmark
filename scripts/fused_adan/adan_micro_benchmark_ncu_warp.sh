clear
echo "===================="
echo "Uninstall vector Adan"
pip uninstall adan
echo "Install warp Adan"
pip install git+https://github.com/AlexwellChen/Adan.git
echo "===================="
echo "Adan Micro Benchmark with ncu, size=4096, n_params=20, warp"
echo "===================="
echo "foreach=False, fused=True"
ncu -o ./ncu_log/adan_each_f_fused_t_warp --target-processes all -k "adan_cuda_kernel" --nvtx --set full \
        python ./benchmark/adan_micro_benchmark.py \
        --foreach False --fused True --size 4096 --n_params 20
echo "===================="