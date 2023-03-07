clear
echo "Adan Micro Benchmark with nvprof, size=2048, n_params=30"
echo "===================="
echo "foreach=False, fused=False"
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu  --capture-range=cudaProfilerApi --stop-on-range-end=true --cudabacktrace=true -x true \ 
        -o ./nvprof_log/adan_each_f_fused_f \
        python ./benchmark/adan_micro_benchmark.py \
        --foreach False --fused False --size 2048 --n_params 30
echo "===================="
echo "foreach=True, fused=False"
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu  --capture-range=cudaProfilerApi --stop-on-range-end=true --cudabacktrace=true -x true \ 
        -o ./nvprof_log/adan_each_t_fused_f \
        python ./benchmark/adan_micro_benchmark.py \
        --foreach True --fused False --size 2048 --n_params 30
echo "===================="
echo "foreach=False, fused=True"
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu  --capture-range=cudaProfilerApi --stop-on-range-end=true --cudabacktrace=true -x true \ 
        -o ./nvprof_log/adan_each_f_fused_t \
        python ./benchmark/adan_micro_benchmark.py \
        --foreach False --fused True --size 2048 --n_params 30
echo "===================="
echo "foreach=True, fused=True"
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu  --capture-range=cudaProfilerApi --stop-on-range-end=true --cudabacktrace=true -x true \ 
        -o ./nvprof_log/adan_each_t_fused_t \
        python ./benchmark/adan_micro_benchmark.py \
        --foreach True --fused True --size 2048 --n_params 30
echo "===================="
echo "Done"
