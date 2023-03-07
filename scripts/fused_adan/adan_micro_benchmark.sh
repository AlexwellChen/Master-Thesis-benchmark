clear
echo "Adan Micro Benchmark with nsys, size=2048, n_params=30"
echo "===================="
echo "foreach=False, fused=False"
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu  --capture-range=cudaProfilerApi  --cudabacktrace=true -x true \
        -o ./nsys_log/adan_each_f_fused_f \
        python ./benchmark/adan_micro_benchmark.py \
        --foreach False --fused False --size 2048 --n_params 30
echo "===================="
echo "foreach=True, fused=False"
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu  --capture-range=cudaProfilerApi  --cudabacktrace=true -x true \
        -o ./nsys_log/adan_each_t_fused_f \
        python ./benchmark/adan_micro_benchmark.py \
        --foreach True --fused False --size 2048 --n_params 30
echo "===================="
echo "foreach=False, fused=True"
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu  --capture-range=cudaProfilerApi  --cudabacktrace=true -x true \
        -o ./nsys_log/adan_each_f_fused_t \
        python ./benchmark/adan_micro_benchmark.py \
        --foreach False --fused True --size 2048 --n_params 30
echo "===================="
echo "foreach=True, fused=True"
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu  --capture-range=cudaProfilerApi  --cudabacktrace=true -x true \
        -o ./nsys_log/adan_each_t_fused_t \
        python ./benchmark/adan_micro_benchmark.py \
        --foreach True --fused True --size 2048 --n_params 30
echo "===================="


echo "Adan Micro Benchmark with ncu, size=2048, n_params=30"
echo "===================="
echo "foreach=False, fused=True"
ncu -o ./ncu_log/adan_each_f_fused_t --target-processes all -k "adan_cuda_kernel" \
        python ./benchmark/adan_micro_benchmark.py \
        --foreach False --fused True --size 2048 --n_params 30
echo "===================="
echo "foreach=True, fused=True"
ncu -o ./ncu_log/adan_each_t_fused_t --target-processes all -k "multi_tensor_apply_kernel"\
        python ./benchmark/adan_micro_benchmark.py \
        --foreach True --fused True --size 2048 --n_params 30
echo "===================="
echo "foreach=False, fused=True, fp16=True"
ncu -o ./ncu_log/adan_each_f_fused_t_fp16 --target-processes all -k "adan_cuda_kernel" \
        python ./benchmark/adan_micro_benchmark.py \
        --foreach False --fused True --size 2048 --n_params 30 --fp16 True
echo "===================="
echo "Done"
