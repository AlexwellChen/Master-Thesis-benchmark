# optimizer_setup=(adamw adan)
# mixed_precision_setup=(fp16 no)
# lightseq_setup=(lightseq huggingface)
# batch_size_setup=(8 16 32)
import pandas as pd

optimizer_setup=['adamw', 'adan']
mixed_precision_setup=['fp16', 'fp32', 'bf16']
lightseq_setup=['lightseq', 'huggingface']
batch_size_setup=[8, 16, 32, 64, 128, 256, 512]
device_setup=['V100', 'A100', 'A10', 'T4', 'TPUv2']

# create a dataframe
df = pd.DataFrame(columns=['optimizer', 'mixed_precision', 'module', 'batch_size', 'device', 'time', 'energy', 'accuracy'])
all_df = pd.read_csv('profiling_all.csv')

# rewrite the for loop in benchmark/systematic.py
for optimizer in optimizer_setup:
    for mixed_precision in mixed_precision_setup:
        for lightseq in lightseq_setup:
            for batch_size in batch_size_setup:
                for device in device_setup:
                    # skip TPUv2 with fp16
                    if device == 'TPUv2' and mixed_precision == 'fp16':
                        continue
                    # skip GPU with bf16
                    if device != 'TPUv2' and mixed_precision == 'bf16':
                        continue
                    # skip GPU with batch size >= 64
                    if device != 'TPUv2' and batch_size >= 64:
                        continue
                    # skip TPU with lightseq
                    if device == 'TPUv2' and lightseq == 'lightseq':
                        continue
                    # skip TPU with batch size <= 64
                    if device == 'TPUv2' and batch_size <= 64:
                        continue
                    # skip TPU with adan
                    if device == 'TPUv2' and optimizer == 'adan':
                        continue
                    # create a new row
                    new_row = {'optimizer': optimizer, 'mixed_precision': mixed_precision, 'module': lightseq, 'batch_size': batch_size, 'device': device, 'time': 0, 'energy': 0, 'accuracy': 0}
                    # append the row to the dataframe
                    df = df.append(new_row, ignore_index=True)

# cost factor
cost_factor = {'V100': 2, 'A100': 4, 'T4': 0.4, 'A10': 1.1, 'TPUv2':1.24} # $/h

# A10
for index, row in df.iterrows():
    # find the corresponding row in act_df
    act_row = all_df[(all_df['optimizer'] == row['optimizer']) & (all_df['mixed_precision'] == row['mixed_precision']) & (all_df['module'] == row['module']) & (all_df['batch_size'] == row['batch_size']) & (all_df['device'] == row['device'])]
    # update the row
    df.loc[index, 'time'] = act_row['time'].values
    df.loc[index, 'energy'] = act_row['energy'].values
    df.loc[index, 'accuracy'] = act_row['accuracy'].values

# add the cost column
df['cost'] = 0
for index, row in df.iterrows():
    # time has values and device is A10
    if df.loc[index, 'time'].size > 0:
        cost = df.loc[index, 'time'] * cost_factor[df.loc[index, 'device']] / 3600
        # keep 4 digits
        cost = round(cost.item(), 4)
        df.loc[index, 'cost'] = cost
    # df.loc[index, 'cost'] = df.loc[index, 'time'] * cost_factor[df.loc[index, 'device']] / 3600


# change no to fp32 in mixed precision column
pass


# save the dataframe to a csv file
df.to_csv('profiling.csv', index=False)