import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# read the csv file
df = pd.read_csv('profiling.csv')
# optimizer,mixed_precision,module,batch_size,device,time,energy,accuracy
# x axis is the time, y axis is the energy for all graphs
markers = {'V100': 's', 'A100': 'o', 'A10': 'X'}
optimizer = {'adamw': 'red', 'fused adan': 'blue'}
mixed_precision = {'fp16': 'green', 'no': 'chocolate'}
module = {'lightseq': 'purple', 'huggingface': 'orange'}
batch_size = {8: 'black', 16: 'pink', 32: 'brown'}

# fig size is 10x10
# device_color = {'V100': 'red', 'A100': 'blue', 'A10': 'green', 'T4': 'orange'}
device_color = "tab10"

# (2, 2) subplots
figs, ax = plt.subplots(2, 2, figsize=(20, 20))

# 1st subplot, hue is optimizer
ax[0, 1].set(xlabel='Time (second)', ylabel='Energy (mJ)')
# subplot title
ax[0, 1].set_title('Training Energy and Time')
sns.scatterplot(data=df[df['device']!='TPUv2'], x='time', y='energy', hue='device', palette=device_color, ax=ax[0, 1])

# 2nd boxplot, y is accuracy, group by device
ax[1, 1].set(xlabel='Device', ylabel='Accuracy')
ax[1, 1].set_title('Test Accuracy')
sns.boxplot(data=df, x='device', y='accuracy', palette=device_color, ax=ax[1, 1])

# 3nd subplot, x is cost, y is time
ax[1, 0].set(ylabel='Cost', xlabel='Time (second)')
ax[1, 0].set_title('Training Cost and Time')
sns.scatterplot(data=df, y='cost', x='time', hue='device', palette=device_color, ax=ax[1, 0])

# 4th subplot, x is cost, y is energy
ax[0, 0].set(ylabel='Cost', xlabel='Energy (mJ)')
ax[0, 0].set_title('Training Cost and Energy')
sns.scatterplot(data=df[df['device']!='TPUv2'], y='cost', x='energy', hue='device', palette=device_color, ax=ax[0, 0])

plt.savefig('test_graph.png')

# new figure for {v100 lightseq fused adan, batch size=16, 32}, {A100 huggingface fused adan, batch size=16 32}

# new figure for mixed precision

# new figure for fused adan and adamw on V100 and A100

# new figure for different hardware on their best training time and cost


