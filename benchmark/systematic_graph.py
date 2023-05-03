import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# read the csv file
df = pd.read_csv('profiling.csv')
# optimizer,mixed_precision,module,batch_size,device,time,energy,accuracy
# delete the row which device is T4, V100, A100
# df = df[(df['device'] == 'A10')]
# x axis is the time, y axis is the energy for all graphs
markers = {'V100': 's', 'A100': 'o', 'A10': 'X'}
optimizer = {'adamw': 'red', 'fused adan': 'blue'}
mixed_precision = {'fp16': 'green', 'no': 'chocolate'}
module = {'lightseq': 'purple', 'huggingface': 'orange'}
batch_size = {8: 'black', 16: 'pink', 32: 'brown'}
accuracy_color = {0.92: 'red', 0.93: 'blue', 0.94: 'green'}
# fig size is 10x10
decive_color = {'V100': 'red', 'A100': 'blue', 'A10': 'green', 'T4': 'orange'}

# (2, 2) subplots
figs, ax = plt.subplots(2, 2, figsize=(20, 20))

# 1st subplot, hue is optimizer
ax[0, 1].set(xlabel='Time (second)', ylabel='Energy (mJ)')
# subplot title
ax[0, 1].set_title('Training Time and Energy')
sns.scatterplot(data=df, x='time', y='energy', hue='device', ax=ax[0, 1])

# 2nd boxplot, y is accuracy, group by device
ax[1, 1].set(xlabel='Device', ylabel='Accuracy')
ax[1, 1].set_title('Test Accuracy')
sns.boxplot(data=df, x='device', y='accuracy', palette=decive_color, ax=ax[1, 1])

# 3nd subplot, x is cost, y is time
ax[1, 0].set(ylabel='Cost', xlabel='Time (second)')
ax[1, 0].set_title('Cost Time graph')
sns.scatterplot(data=df, y='cost', x='time', hue='device', ax=ax[1, 0])

# 4th subplot, x is cost, y is energy
ax[0, 0].set(ylabel='Cost', xlabel='Energy (mJ)')
ax[0, 0].set_title('Cost Energy graph')
sns.scatterplot(data=df, y='cost', x='energy', hue='device', ax=ax[0, 0])


# ax[1].set(xlabel='Device', ylabel='Accuracy')
# ax[1].set_title('Test Accuracy')
# sns.boxplot(data=df, x='device', y='accuracy', palette=palette, ax=ax[1])

# # 2nd subplot, accuracy
# ax[1].set(xlabel='Time (second)', ylabel='Test accuracy')
# ax[1].set_title('Test Accuracy')
# # group by accuracy, x is accuracy, y is the number of this accuracy
# sns.countplot(data=df, x='accuracy', ax=ax[1])


# # 3rd subplot, hue is module
# ax[1, 0].set(xlabel='Time (second)', ylabel='Energy (mJ)')
# ax[1, 0].set_title('Encoder Modules')
# sns.scatterplot(data=df, x='time', y='energy', hue='module', palette=module, style='device', markers=markers, ax=ax[1, 0])

# # 4th subplot, hue is batch_size
# ax[1, 1].set(xlabel='Time (second)', ylabel='Energy (mJ)')
# ax[1, 1].set_title('Batch size')
# sns.scatterplot(data=df, x='time', y='energy', hue='batch_size', palette=batch_size, style='device', markers=markers, ax=ax[1, 1])

# plt.title('Time-Energy graph for different optimization techniques')

# save the figure
plt.savefig('test_graph.png')


