from colossalai.amp import AMP_TYPE

fp16 = dict(
  mode=AMP_TYPE.TORCH
)

BATCH_SIZE = 16

NUM_EPOCHS = 3