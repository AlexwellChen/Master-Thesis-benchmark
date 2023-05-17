import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# read the csv file
df = pd.read_csv('profiling.csv')
# optimizer,mixed_precision,module,batch_size,device,time,energy,accuracy
# x axis is the time, y axis is the energy for all graphs
markers = {'V100': 's', 'A100': 'o', 'A10': 'X'}
optimizer = {'adamw': 'red', 'adan': 'blue'}
mixed_precision = {'fp16': 'green', 'fp32': 'chocolate', 'bf16': 'purple'}
module = {'lightseq': 'purple', 'huggingface': 'orange'}
batch_size = {8: 'black', 16: 'pink', 32: 'brown'}

# fig size is 10x10
# device_color = {'V100': 'red', 'A100': 'blue', 'A10': 'green', 'T4': 'orange'}
device_color = "tab10"

# (2, 2) subplots
figs, ax = plt.subplots(2, 2, figsize=(12, 12))

# 1st subplot, hue is optimizer
ax[0, 1].set(xlabel='Time (second)', ylabel='Energy (mJ)')
# subplot title
ax[0, 1].set_title('(B) Training Energy and Time')
sns.scatterplot(data=df[df['device']!='TPUv2'], x='time', y='energy', hue='device', palette=device_color, ax=ax[0, 1])

# 2nd boxplot, y is accuracy, x is time, hue is mixed_precision, shape is optimizer
ax[1, 1].set(xlabel='Time (second)', ylabel='Accuracy (%)')
ax[1, 1].set_title('(D) Test Accuracy and Time')
sns.scatterplot(data=df, x='time', y='accuracy', hue='mixed_precision', palette=mixed_precision, ax=ax[1, 1], style='optimizer')

# 3nd subplot, x is cost, y is time
ax[1, 0].set(ylabel='Cost ($)', xlabel='Time (second)')
ax[1, 0].set_title('(C) Training Cost and Time')
sns.scatterplot(data=df, y='cost', x='time', hue='device', palette=device_color, ax=ax[1, 0])

# 4th subplot, x is cost, y is energy
ax[0, 0].set(ylabel='Cost ($)', xlabel='Energy (mJ)')
ax[0, 0].set_title('(A) Training Cost and Energy')
sns.scatterplot(data=df[df['device']!='TPUv2'], y='cost', x='energy', hue='device', palette=device_color, ax=ax[0, 0])

plt.savefig('overall_results.png')

# # new figure for {v100 lightseq fused adan, batch size=16, 32}, {A100 huggingface fused adan, batch size=16 32}
# fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# ax.set(xlabel='Time (second)', ylabel='Energy (mJ)')
# ax.set_title('Software Optimization Energy and Time')
# df_1 = df[(df['device']=='V100') & (df['module']=='lightseq') & (df['optimizer']=='adan') & (df['batch_size']==16)]
# df_2 = df[(df['device']=='V100') & (df['module']=='lightseq') & (df['optimizer']=='adan') & (df['batch_size']==32)]
# df_3 = df[(df['device']=='A100') & (df['module']=='huggingface') & (df['optimizer']=='adan') & (df['batch_size']==16)]
# df_4 = df[(df['device']=='A100') & (df['module']=='huggingface') & (df['optimizer']=='adan') & (df['batch_size']==32)]
# # concat the dataframes
# df_new = pd.concat([df_1, df_2, df_3, df_4])
# # drop fp32
# df_new = df_new[df_new['mixed_precision']!='fp32']
# sns.scatterplot(data=df_new, x='time', y='energy', hue='device', palette=device_color, ax=ax)
# plt.savefig('Software_Optimization.png')

# # new figure for mixed precision
# fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# ax.set(xlabel='Time (second)', ylabel='Energy (mJ)')
# ax.set_title('Mixed Precision Energy and Time')
# sns.scatterplot(data=df[df['device']!='TPUv2'], x='time', y='energy', hue='device', palette=device_color, ax=ax, style='mixed_precision')
# plt.savefig('Mixed_Precision.png')

# # new figure for different hardware on their best training time and cost
# fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# ax.set(xlabel='Time (second)', ylabel='Cost ($)')
# ax.set_title('Best training time and cost for each device')
# # get rows of the best time for each device
# df_best_time = df.loc[df.groupby(['device'])['time'].idxmin()]
# sns.scatterplot(data=df_best_time, x='time', y='cost', hue='device', palette=device_color, ax=ax)
# plt.savefig('Best_Time_Cost.png')

# # # new figure for different optimizer, x is time, y is accuracy
# # fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# # ax.set(xlabel='Time (second)', ylabel='Accuracy (%)')
# # ax.set_title('Optimizer Accuracy and Time')
# # sns.scatterplot(data=df, x='time', y='accuracy', hue='optimizer', palette=optimizer, ax=ax)
# # plt.savefig('Optimizer_Accuracy_Time.png')

