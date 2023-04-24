clear
# creat bash array
optimizer_setup=(adamw)
mixed_precision_setup=(no)
lightseq_setup=(huggingface)
batch_size_setup=(16)

for optimizer in ${optimizer_setup[@]}
        do
        for mixed_precision in ${mixed_precision_setup[@]}
                do
                for lightseq in ${lightseq_setup[@]}
                        do
                        for batch_size in ${batch_size_setup[@]} 
                                do
                                echo "optimizer: $optimizer, mixed_precision: $mixed_precision, lightseq: $lightseq, batch_size: $batch_size"
                                accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc.yaml \
                                        ./benchmark/systematic.py \
                                        --n_epochs 3 \
                                        --device v100 \
                                        --batch_size $batch_size \
                                        --optimizer $optimizer \
                                        --fp16 $mixed_precision \
                                        --lightseq $lightseq
                                done
                        done
                done
        done
