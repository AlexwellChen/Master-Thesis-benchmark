combaination_list[0]=('adamw fp16 huggingface 32 v100')
combaination_list[1]=('adamw no lightseq 8 v100')
combaination_list[2]=('adamw no lightseq 16 v100')
combaination_list[3]=('adamw no lightseq 32 v100')
combaination_list[4]=('adamw no huggingface 32 v100')
combaination_list[5]=('adan fp16 huggingface 32 v100')
combaination_list[6]=('adan no lightseq 8 v100')
combaination_list[7]=('adan no lightseq 16 v100')
combaination_list[8]=('adan no lightseq 32 v100')
combaination_list[9]=('adan no huggingface 32 v100')

for combaination in ${combaination_list[@]}
        do
        # split string to array
        combaination=(${combaination})
        echo "optimizer: ${combaination[0]}, mixed_precision: ${combaination[1]}, lightseq: ${combaination[2]}, batch_size: ${combaination[3]}, device: ${combaination[4]}"
        accelerate launch --config_file ./accelerate_config/imdb_bert_base_acc.yaml \
                ./benchmark/systematic.py \
                --n_epochs 1 \
                --device ${combaination[4]} \
                --batch_size ${combaination[3]} \
                --optimizer ${combaination[0]} \
                --fp16 ${combaination[1]} \
                --lightseq ${combaination[2]}
        done