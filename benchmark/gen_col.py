# optimizer_setup=(adamw adan)
# mixed_precision_setup=(fp16 no)
# lightseq_setup=(lightseq huggingface)
# batch_size_setup=(8 16 32)
import pandas as pd

optimizer_setup=['adamw', 'adan']
mixed_precision_setup=['fp16', 'no']
lightseq_setup=['lightseq', 'huggingface']
batch_size_setup=[8, 16, 32]

# create a dataframe
df = pd.DataFrame(columns=['optimizer', 'mixed_precision', 'module', 'batch_size', 'device', 'time', 'energy', 'accuracy'])
act_df = pd.read_csv('profiling_v100.csv')
# rewrite the for loop in benchmark/systematic.py
for optimizer in optimizer_setup:
    for mixed_precision in mixed_precision_setup:
        for lightseq in lightseq_setup:
            for batch_size in batch_size_setup:
                # create a new row
                new_row = {'optimizer': optimizer, 'mixed_precision': mixed_precision, 'module': lightseq, 'batch_size': batch_size, 'device': 'v100', 'time': 0, 'energy': 0, 'accuracy': 0}
                # append the row to the dataframe
                df = df.append(new_row, ignore_index=True)

# iterate over the dataframe
for index, row in df.iterrows():
    # find the corresponding row in act_df
    act_row = act_df[(act_df['optimizer'] == row['optimizer']) & (act_df['mixed_precision'] == row['mixed_precision']) & (act_df['module'] == row['module']) & (act_df['batch_size'] == row['batch_size']) & (act_df['device'] == row['device'])]
    # update the row
    df.loc[index, 'time'] = act_row['time'].values
    df.loc[index, 'energy'] = act_row['energy'].values
    df.loc[index, 'accuracy'] = act_row['accuracy'].values
# save the dataframe to a csv file
df.to_csv('profiling.csv', index=False)