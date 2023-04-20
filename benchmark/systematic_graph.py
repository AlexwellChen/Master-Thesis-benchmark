import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# read the csv file
df = pd.read_csv('profiling.csv')
# optimizer,mixed_precision,module,batch_size,device,time,energy,accuracy

# x axis is the time, y axis is the energy for all graphs

# fig size is 10x10
plt.figure(figsize=(10, 5))
sns.scatterplot(data=df, x='time', y='energy', hue='optimizer', palette='Set1')
plt.title('Time vs Energy for different optimizers')
plt.show()


