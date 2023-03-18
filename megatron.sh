export MYDIR=/databricks/driver

git clone https://github.com/huggingface/transformers.git $MYDIR/transformers

mkdir -p $MYDIR/nvidia/megatron-bert-cased-345m

wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_bert_345m/versions/v0.1_cased/zip 
mv file.zip $MYDIR/nvidia/megatron-bert-cased-345m/checkpoint.zip

mkdir tmp
cd tmp
git clone https://github.com/NVIDIA/Megatron-LM
PYTHONPATH=/tmp/Megatron-LM 
python src/transformers/models/megatron_bert/convert_megatron_bert_checkpoint.py ...

python3 $MYDIR/transformers/src/transformers/models/megatron_bert/convert_megatron_bert_checkpoint.py $MYDIR/nvidia/megatron-bert-cased-345m/checkpoint.zip
