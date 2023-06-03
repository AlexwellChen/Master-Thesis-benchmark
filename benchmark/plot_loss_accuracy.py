import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

def get_loss_val(name):
    loss_dict = {}
    for file_path in glob.glob("./benchmark/loss_val/"+name+"/*"):
        file_name = os.path.basename(file_path)
        # Skip markdown files
        if file_name.endswith(".md"):
            continue
        # skip directory

        if os.path.isdir(file_path):
            continue
        
        file_name = file_name.split(".")[0]

        if name == "fused_unfused_3ep":
            file_name = file_name.split("_")
            if "fused" in file_name:
                file_name = "fused"
            elif "unfused" in file_name:
                file_name = "unfused"

        if name == "lightseq":
            file_name = file_name.split("_")
            if "lightseq" in file_name:
                file_name = "lightseq"
            elif "huggingface" in file_name:
                file_name = "huggingface"

        if name == "mix_precision":
            file_name = file_name.split("_")
            if "mixed" in file_name:
                file_name = "fp16"
            elif "unmix" in file_name:
                file_name = "fp32"

        with open(file_path, "r") as f:
            lines = f.readlines()
        loss_values = [float(line.strip()) for line in lines]
        loss_values = [sum(loss_values[max(0, i-20):i+1])/len(loss_values[max(0, i-20):i+1]) for i in range(len(loss_values))]
        loss_dict[file_name] = loss_values
    loss_df = pd.DataFrame.from_dict(loss_dict, orient="index").T
    return loss_df

def get_acc_val(name):
    acc_dict = {}
    for file_path in glob.glob("./benchmark/acc_val/"+name+"/*"):
        file_name = os.path.basename(file_path)
        # Skip markdown files
        if file_name.endswith(".md"):
            continue
        # skip directory

        if os.path.isdir(file_path):
            continue
        with open(file_path, "r") as f:
            lines = f.readlines()

        file_name = file_name.split(".")[0]

        if name == "fused_unfused_3ep":
            file_name = file_name.split("_")
            if "fused" in file_name:
                file_name = "fused"
            elif "unfused" in file_name:
                file_name = "unfused"

        if name == "lightseq":
            file_name = file_name.split("_")
            if "lightseq" in file_name:
                file_name = "lightseq"
            elif "huggingface" in file_name:
                file_name = "huggingface"
            
        if name == "mix_precision":
            file_name = file_name.split("_")
            if "mixed" in file_name:
                file_name = "fp16"
            elif "unmix" in file_name:
                file_name = "fp32"

        acc_values = [float(line.strip()) for line in lines]
        acc_dict[file_name] = acc_values
    acc_df = pd.DataFrame.from_dict(acc_dict, orient="index").T
    return acc_df

if __name__ == "__main__":
    # get file name from os.argv
    file_name = "fused_unfused_3ep"
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=160)

    loss_df = get_loss_val(file_name).sort_index(axis=1)
    acc_df = get_acc_val(file_name).sort_index(axis=1)


    sns.lineplot(data=loss_df, ax=axes[0])
    sns.lineplot(data=acc_df, ax=axes[1])

    axes[0].set_title("Loss Curve", fontsize=12)
    axes[1].set_title("Accuracy Curve", fontsize=12)

    plt.savefig("./benchmark/figure/"+file_name+"_acc_loss_curve.png")

