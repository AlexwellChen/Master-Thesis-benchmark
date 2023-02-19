import os
import glob

loss_dict = {}

# 遍历文件夹中的所有文件
for file_path in glob.glob("./loss_val/*"):
    # 从文件路径中获取文件名
    file_name = os.path.basename(file_path)
    # 读取文件中的所有行
    with open(file_path, "r") as f:
        lines = f.readlines()
    # 将每行loss值转换为float类型，并存储到list中
    loss_values = [float(line.strip()) for line in lines]
    # smooth the loss values
    loss_values = [sum(loss_values[max(0, i-10):i+1])/len(loss_values[max(0, i-10):i+1]) for i in range(len(loss_values))]
    # 将文件名和loss值组成的list存储到字典中
    loss_dict[file_name] = loss_values

import pandas as pd
import seaborn as sns

# 将字典转换为DataFrame，便于绘图
df = pd.DataFrame.from_dict(loss_dict, orient="index").T

# 使用seaborn绘制线图
sns.lineplot(data=df)