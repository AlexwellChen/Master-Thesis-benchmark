clear
# creat bash array
# optimizer_setup=(adamw adan)
# mixed_precision_setup=(fp16 fp32)
# lightseq_setup=(lightseq huggingface)
# batch_size_setup=(8 16 32)
optimizer_setup=(adamw)
mixed_precision_setup=(fp16)
lightseq_setup=(lightseq)
batch_size_setup=(32)

# write a bash loop to run all the combinations
for optimizer in ${optimizer_setup[@]}; do
        for mixed_precision in ${mixed_precision_setup[@]}; do
                for lightseq in ${lightseq_setup[@]}; do
                        for batch_size in ${batch_size_setup[@]}; do
                                echo "optimizer: $optimizer, mixed_precision: $mixed_precision, lightseq: $lightseq, batch_size: $batch_size"
                                accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc.yaml \
                                        ./benchmark/systematic.py \
                                        --n_epochs 1 \
                                        --device v100 \
                                        --batch_size $batch_size \
                                        --optimizer $optimizer \
                                        --fp16 $mixed_precision \
                                        --lightseq $lightseq \
                        done
                done
        done
done

# adamw:
#     lr=1e-5, wd=0.01, warmup=160
#     betas=(0.9, 0.999), eps=1e-8
# adan:
#     lr=5e-5, wd=0.005, warmup=320
#     betas=(0.98, 0.92, 0.99), eps=1e-8
# adam:
#     lr=1e-5, wd=0.01, warmup=160
#     betas=(0.9, 0.999), eps=1e-8