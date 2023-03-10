clear
pip uninstall adan
pip install git+https://github.com/AlexwellChen/Adan.git
echo "Adan Micro Benchmark with nsys, size=4096, n_params=20"
echo "===================="
echo "foreach=False, fused=False"
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu  --capture-range=cudaProfilerApi  --cudabacktrace=true -x true \
        -o ./nsys_log/adan_each_f_fused_f \
        python ./benchmark/adan_micro_benchmark.py \
        --foreach False --fused False --size 4096 --n_params 20
echo "===================="
echo "foreach=True, fused=False"
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu  --capture-range=cudaProfilerApi  --cudabacktrace=true -x true \
        -o ./nsys_log/adan_each_t_fused_f \
        python ./benchmark/adan_micro_benchmark.py \
        --foreach True --fused False --size 4096 --n_params 20
echo "===================="
echo "foreach=False, fused=True"
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu  --capture-range=cudaProfilerApi  --cudabacktrace=true -x true \
        -o ./nsys_log/adan_each_f_fused_t \
        python ./benchmark/adan_micro_benchmark.py \
        --foreach False --fused True --size 4096 --n_params 20
echo "===================="
echo "foreach=True, fused=True"
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu  --capture-range=cudaProfilerApi  --cudabacktrace=true -x true \
        -o ./nsys_log/adan_each_t_fused_t \
        python ./benchmark/adan_micro_benchmark.py \
        --foreach True --fused True --size 4096 --n_params 20
echo "===================="
echo "Done"
