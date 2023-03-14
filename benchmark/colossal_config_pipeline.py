from colossalai.amp import AMP_TYPE

fp16 = dict(
  mode=AMP_TYPE.TORCH
)

BATCH_SIZE = 16
SEQ_LENGTH = 8192
HIDDEN_SIZE = 768
NUM_EPOCHS = 3

parallel = dict(
    pipeline=2
)
# pipeline config
NUM_MICRO_BATCHES = parallel['pipeline']
TENSOR_SHAPE = (BATCH_SIZE // NUM_MICRO_BATCHES, SEQ_LENGTH, HIDDEN_SIZE)
fp16 = dict(mode=AMP_TYPE.NAIVE)
clip_grad_norm = 1.0