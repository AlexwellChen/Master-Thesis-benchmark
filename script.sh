python glue_mrpc_roberta.py --log_file_name glue_cola_roberta_adam --n_epochs 10 --optimizer adam --fused_optimizer False --foreach True --lr 1e-5
python glue_mrpc_roberta.py --log_file_name glue_cola_roberta_adan --n_epochs 10 --optimizer adan --fused_optimizer False --foreach True --lr 4e-5

python glue_mrpc_bert_based.py --log_file_name glue_mrpc_bert_based_adam --n_epochs 10 --optimizer adam --fused_optimizer False --foreach True --lr 1e-5
python glue_mrpc_bert_based.py --log_file_name glue_mrpc_bert_based_adan --n_epochs 10 --optimizer adan --fused_optimizer False --foreach True --lr 4e-5
