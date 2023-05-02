import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# read the csv file
df = pd.read_csv('profiling_A10.csv')
# optimizer,mixed_precision,module,batch_size,device,time,energy,accuracy
# delete the row which device is T4
df = df[df['device'] != 'T4']
# x axis is the time, y axis is the energy for all graphs
markers = {'V100': 's', 'A100': 'o', 'A10': 'X'}
optimizer = {'adamw': 'red', 'fused adan': 'blue'}
mixed_precision = {'fp16': 'green', 'no': 'chocolate'}
module = {'lightseq': 'purple', 'huggingface': 'orange'}
batch_size = {8: 'black', 16: 'pink', 32: 'brown'}
accuracy_color = {0.92: 'red', 0.93: 'blue', 0.94: 'green'}
# fig size is 10x10
palette = sns.light_palette("seagreen")

# (2, 2) subplots
figs, ax = plt.subplots(1, 2, figsize=(16, 8))

# 1st subplot, hue is optimizer
ax[0].set(xlabel='Time (second)', ylabel='Energy (mJ)')
# subplot title
ax[0].set_title('Training Time and Energy')
sns.scatterplot(data=df, x='time', y='energy', style='device', markers=markers, ax=ax[0])

# 2nd subplot, accuracy boxplot, group by device
ax[1].set(xlabel='Device', ylabel='Accuracy')
ax[1].set_title('Test Accuracy')
sns.boxplot(data=df, x='device', y='accuracy', palette=palette, ax=ax[1])

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


