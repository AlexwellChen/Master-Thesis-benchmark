import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# read the csv file
df = pd.read_csv('profiling copy.csv')
# convert energy from mJ to kW·h
df['energy'] = df['energy'] / 3600000000
# optimizer,mixed_precision,module,batch_size,device,time,energy,accuracy
# x axis is the time, y axis is the energy for all graphs
markers = {'V100': 's', 'A100': 'o', 'A10': 'X'}
optimizer = {'adamw': 'red', 'adan': 'blue'}
mixed_precision = {'fp16': 'green', 'fp32': 'chocolate', 'bf16': 'purple'}
module = {'lightseq': 'purple', 'huggingface': 'orange'}
batch_size = {8: 'black', 16: 'pink', 32: 'brown'}

# fig size is 10x10
# device_color = {'V100': 'red', 'A100': 'blue', 'A10': 'green', 'T4': 'orange', 'TPUv2': 'purple', 'TPUv3': 'brown'}
device_color = "tab10"

# (2, 2) subplots
figs, ax = plt.subplots(2, 2, figsize=(10, 10), dpi=160)


# 1st subplot, hue is optimizer
# ax[0, 1].set(xlabel='Time (second)', ylabel='Energy (kW·h)')
ax[0, 1].set_xlabel(xlabel='Time (second)', fontsize=12)
ax[0, 1].set_ylabel(ylabel='Energy (kW·h)', fontsize=12)
# subplot title
ax[0, 1].set_title('(B) Training Energy and Time', fontsize=12)
# df without TPUv2 and TPUv3
df_no_TPU = df[(df['device']!='TPUv2') & (df['device']!='TPUv3')]
sns.scatterplot(data=df_no_TPU, x='time', y='energy', hue='device', palette=device_color, ax=ax[0, 1])

# 2nd boxplot, y is accuracy, x is time, hue is mixed_precision, shape is optimizer
# ax[1, 1].set(xlabel='Time (second)', ylabel='Accuracy (%)')
ax[1, 1].set_xlabel(xlabel='Time (second)', fontsize=12)
ax[1, 1].set_ylabel(ylabel='Accuracy (%)', fontsize=12)
ax[1, 1].set_title('(D) Test Accuracy and Time', fontsize=12)
sns.scatterplot(data=df, x='time', y='accuracy', hue='mixed_precision', palette=mixed_precision, ax=ax[1, 1], style='optimizer')

# 3nd subplot, x is cost, y is time
# ax[1, 0].set(ylabel='Cost ($)', xlabel='Time (second)')
ax[1, 0].set_xlabel(xlabel='Time (second)', fontsize=12)
ax[1, 0].set_ylabel(ylabel='Cost ($)', fontsize=12)
ax[1, 0].set_title('(C) Training Cost and Time', fontsize=12)
sns.scatterplot(data=df, y='cost', x='time', hue='device', palette=device_color, ax=ax[1, 0])

# 4th subplot, x is cost, y is energy
# ax[0, 0].set(ylabel='Cost ($)', xlabel='Energy (kW·h)')
ax[0, 0].set_xlabel(xlabel='Energy (kW·h)', fontsize=12)
ax[0, 0].set_ylabel(ylabel='Cost ($)', fontsize=12)
ax[0, 0].set_title('(A) Training Cost and Energy', fontsize=12)
sns.scatterplot(data=df[(df['device']!='TPUv2') & (df['device']!='TPUv3')], y='cost', x='energy', hue='device', palette=device_color, ax=ax[0, 0])


plt.savefig('overall_results_v3_fnt12.png')

# new figure for {v100 lightseq fused adan, batch size=16, 32}, {A100 huggingface fused adan, batch size=16 32}, {V100 huggingface fused adan, batch size=16 32}
fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=160)
# ax.set(xlabel='Time (second)', ylabel='Energy (kW·h)')
ax.set_xlabel(xlabel='Time (second)', fontsize=12)
ax.set_ylabel(ylabel='Energy (kW·h)', fontsize=12)
ax.set_title('Software Optimization Energy and Time', fontsize=12)
df_1 = df[(df['device']=='V100') & (df['module']=='lightseq') & (df['optimizer']=='adan') & (df['batch_size']==16)]
df_2 = df[(df['device']=='V100') & (df['module']=='lightseq') & (df['optimizer']=='adan') & (df['batch_size']==32)]
df_3 = df[(df['device']=='A100') & (df['module']=='huggingface') & (df['optimizer']=='adan') & (df['batch_size']==16)]
df_4 = df[(df['device']=='A100') & (df['module']=='huggingface') & (df['optimizer']=='adan') & (df['batch_size']==32)]
df_5 = df[(df['device']=='V100') & (df['module']=='huggingface') & (df['optimizer']=='adan') & (df['batch_size']==16)]
# concat the dataframes
df_new = pd.concat([df_1, df_2, df_3, df_4, df_5])
# drop fp32
df_new = df_new[df_new['mixed_precision']!='fp32']
sns.scatterplot(data=df_new, x='time', y='energy', hue='device', palette=device_color, ax=ax)
plt.savefig('Software_Optimization_v3_fnt12.png')

# new figure for mixed precision
fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=160)
# ax.set(xlabel='Time (second)', ylabel='Energy (kW·h)')
# ax.set_title('Mixed Precision Energy and Time')
ax.set_xlabel(xlabel='Time (second)', fontsize=12)
ax.set_ylabel(ylabel='Energy (kW·h)', fontsize=12)
ax.set_title('Mixed Precision Energy and Time', fontsize=12)
sns.scatterplot(data=df[(df['device']!='TPUv2') & (df['device']!='TPUv3')], x='time', y='energy', hue='device', palette=device_color, ax=ax, style='mixed_precision')
plt.savefig('Mixed_Precision_v3_fnt12.png')

# new figure for different hardware on their best training time and cost
fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=160)
# ax.set(xlabel='Time (second)', ylabel='Cost ($)')
# ax.set_title('Best training time and cost for each device')
ax.set_xlabel(xlabel='Time (second)', fontsize=12)
ax.set_ylabel(ylabel='Cost ($)', fontsize=12)
ax.set_title('Best training time and cost for each device', fontsize=12)
# get rows of the best time for each device
df_best_time = df.loc[df.groupby(['device'])['time'].idxmin()]
print(df_best_time)
sns.scatterplot(data=df_best_time, x='time', y='cost', hue='device', palette=device_color, ax=ax)
plt.savefig('Best_Time_Cost_v3_fnt12.png')

# new figure for different optimizer, x is time, y is accuracy
fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=160)
# ax.set(xlabel='Time (second)', ylabel='Accuracy (%)')
# ax.set_title('Optimizer Accuracy and Time')
ax.set_xlabel(xlabel='Time (second)', fontsize=12)
ax.set_ylabel(ylabel='Accuracy (%)', fontsize=12)
ax.set_title('Optimizer Accuracy and Time', fontsize=12)
sns.scatterplot(data=df, x='time', y='accuracy', hue='optimizer', palette=optimizer, ax=ax)
plt.savefig('Optimizer_Accuracy_Time_v3_fnt12.png')

# (2, 2) subplots for accuracy on devices, on different batch size, on different mixed precision, on different module
figs, ax = plt.subplots(2, 2, figsize=(10, 10), dpi=160)

# 1st subplot, hue is optimizer
# ax[0, 0].set(xlabel='Device', ylabel='Accuracy (%)')
ax[0, 0].set_ylabel(ylabel='Accuracy (%)', fontsize=12)
ax[0, 0].set_xlabel(xlabel='Device', fontsize=12)
# subplot title
ax[0, 0].set_title('(A) Test Accuracy on Different Devices', fontsize=12)
sns.boxplot(data=df, x='device', y='accuracy', ax=ax[0, 0])

# 2nd boxplot, y is accuracy, x is batch_size, hue is mixed_precision, shape is optimizer
# ax[0, 1].set(xlabel='Batch Size', ylabel='Accuracy (%)')
# ax[0, 1].set_title('(B) Test Accuracy on Different Batch Size')
ax[0, 1].set_ylabel(ylabel='Accuracy (%)', fontsize=12)
ax[0, 1].set_xlabel(xlabel='Batch Size', fontsize=12)
ax[0, 1].set_title('(B) Test Accuracy on Different Batch Size', fontsize=12)
sns.boxplot(data=df, x='batch_size', y='accuracy', ax=ax[0, 1])

# 3nd subplot, x is module, y is accuracy
# ax[1, 0].set(ylabel='Accuracy (%)', xlabel='Module')
ax[1, 0].set_ylabel(ylabel='Accuracy (%)', fontsize=12)
ax[1, 0].set_xlabel(xlabel='Module', fontsize=12)
ax[1, 0].set_title('(C) Test Accuracy on Different Modules', fontsize=12)
sns.boxplot(data=df, y='accuracy', x='module', ax=ax[1, 0])

# 4th subplot, x is mixed_precision, y is accuracy
# ax[1, 1].set(ylabel='Accuracy (%)', xlabel='Data types')
ax[1, 1].set_ylabel(ylabel='Accuracy (%)', fontsize=12)
ax[1, 1].set_xlabel(xlabel='Data types', fontsize=12)
ax[1, 1].set_title('(D) Test Accuracy on Different Data types', fontsize=12)
sns.boxplot(data=df, y='accuracy', x='mixed_precision', ax=ax[1, 1])

plt.savefig('Accuracy_with_para_v3_fnt12.png')

