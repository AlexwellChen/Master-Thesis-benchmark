import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

def get_loss_val():
    loss_dict = {}
    for file_path in glob.glob("./benchmark/loss_val/imdb_acc90/*"):
        file_name = os.path.basename(file_path)
        # Skip markdown files
        if file_name.endswith(".md"):
            continue
        # skip directory

        optmizer_name = file_name.split("_")[1] # get the optimizer name
        is_fused = True if file_name.split("_")[2] else False # get the fused status
        if is_fused:
            file_name = optmizer_name + "_fused"
        else:
            file_name = optmizer_name

        if os.path.isdir(file_path):
            continue
        with open(file_path, "r") as f:
            lines = f.readlines()
        loss_values = [float(line.strip()) for line in lines]
        loss_values = [sum(loss_values[max(0, i-20):i+1])/len(loss_values[max(0, i-20):i+1]) for i in range(len(loss_values))]
        loss_dict[file_name] = loss_values
    loss_df = pd.DataFrame.from_dict(loss_dict, orient="index").T
    return loss_df

def get_acc_val():
    acc_dict = {}
    for file_path in glob.glob("./benchmark/acc_val/imdb_acc90/*"):
        file_name = os.path.basename(file_path)
        # Skip markdown files
        if file_name.endswith(".md"):
            continue
        # skip directory
        
        optmizer_name = file_name.split("_")[1] # get the optimizer name
        is_fused = True if file_name.split("_")[2] else False # get the fused status
        if is_fused:
            file_name = optmizer_name + "_fused"
        else:
            file_name = optmizer_name

        if os.path.isdir(file_path):
            continue
        with open(file_path, "r") as f:
            lines = f.readlines()
        
        acc_values = [float(line.strip()) for line in lines]
        acc_dict[file_name] = acc_values
    acc_df = pd.DataFrame.from_dict(acc_dict, orient="index").T
    return acc_df

if __name__ == "__main__":
    # get file name from os.argv
    file_name = sys.argv[1]
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    loss_df = get_loss_val()
    acc_df = get_acc_val()

    sns.lineplot(data=loss_df, ax=axes[0])
    sns.lineplot(data=acc_df, ax=axes[1])

    axes[0].set_title("Loss Curve")
    axes[1].set_title("Accuracy Curve")

    plt.savefig("./benchmark/figure/"+file_name+"_acc_loss_curve.png")

