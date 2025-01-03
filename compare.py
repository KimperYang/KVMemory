import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Example directory containing the .jsonl files
input_dir = "result/12-15/baseline"

# Regex pattern to extract information from the filename
# Filename format assumed: NQ_baseline_bsz256_ckpt2000_at0_0.516_20241216-002110.jsonl
pattern = re.compile(r"(.*)_(ckpt\d+)_at(\d+)_(\d+\.\d+)_\d{8}-\d{6}\.jsonl")

data = []

# Iterate over all files in the given directory
for filename in os.listdir(input_dir):
    if filename.endswith(".jsonl"):
        match = pattern.match(filename)
        if match:
            prefix = match.group(1)   # e.g. "NQ_baseline_bsz256"
            ckpt_str = match.group(2) # e.g. "ckpt2000"
            at_val   = match.group(3) # e.g. "0"
            acc_str  = match.group(4) # e.g. "0.516"
            
            ckpt = int(ckpt_str.replace("ckpt", ""))
            accuracy = float(acc_str)
            
            data.append({
                "prefix": prefix,
                "at": int(at_val),
                "ckpt": ckpt,
                "accuracy": accuracy
            })

# Create DataFrame
df = pd.DataFrame(data)

# Sort values by ckpt
df = df.sort_values(by="ckpt")

# Compute the average accuracy at each training step
avg_acc = df.groupby('ckpt')['accuracy'].mean().reset_index()
print("Average accuracy at each training step:")
print(avg_acc)

# Get unique at values and create a fading color palette
at_values = sorted(df['at'].unique())
palette = sns.color_palette("Blues", n_colors=len(at_values))  # Choose your preferred palette

plt.figure(figsize=(10, 6))

# Plot each condition line separately to control colors
for i, a in enumerate(at_values):
    subset = df[df['at'] == a]
    color = palette[i]

    # Plot the line for this at value
    sns.lineplot(data=subset, x="ckpt", y="accuracy", color=color, marker="o", label=f"at{a}")

    # Annotate each point with its accuracy
    for idx, row in subset.iterrows():
        plt.text(row['ckpt'], row['accuracy'], f"{row['accuracy']:.3f}", 
                 ha='center', va='bottom', color=color, fontsize=9)

# Plot the average line
sns.lineplot(data=avg_acc, x="ckpt", y="accuracy", color='black', marker="o", label='Average')
for idx, row in avg_acc.iterrows():
    plt.text(row['ckpt'], row['accuracy'], f"{row['accuracy']:.3f}",
             ha='center', va='bottom', color='black', fontsize=9)

plt.title("Accuracy vs Training Steps")
plt.xlabel("Training Steps (ckpt)")
plt.ylabel("Accuracy")
plt.legend(title="Condition")
plt.tight_layout()

plt.savefig("tem.png")

# multiple
# import os
# import re
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Directories containing .jsonl files
# dirs = [
#     "result/12-15/bias",
#     "result/12-15/baseline",
#     "result/12-15/reencode"
# ]

# pattern = re.compile(r"(.*)_ckpt(\d+)_at(\d+)_(\d+\.\d+)_\d{8}-\d{6}\.jsonl")

# all_data = []

# for d in dirs:
#     # You might want to name the setting based on the directory name
#     setting_name = os.path.basename(d.strip("/"))
    
#     for filename in os.listdir(d):
#         if filename.endswith(".jsonl"):
#             match = pattern.match(filename)
#             if match:
#                 prefix = match.group(1)
#                 ckpt_str = match.group(2)
#                 at_val = match.group(3)
#                 acc_str = match.group(4)
                
#                 ckpt = int(ckpt_str)
#                 accuracy = float(acc_str)
                
#                 all_data.append({
#                     "setting": setting_name,
#                     "prefix": prefix,
#                     "at": at_val,
#                     "ckpt": ckpt,
#                     "accuracy": accuracy
#                 })

# df = pd.DataFrame(all_data)

# # Sort by ckpt
# df = df.sort_values(by="ckpt")

# # Now we want the average accuracy for each setting at each ckpt across all at-values
# avg_df = df.groupby(["setting", "ckpt"])["accuracy"].mean().reset_index()

# # Plotting
# plt.figure(figsize=(10, 6))
# sns.lineplot(data=avg_df, x="ckpt", y="accuracy", hue="setting", marker="o")

# # Annotate each data point with its exact number
# # We'll loop through each setting separately to annotate
# for setting in avg_df["setting"].unique():
#     subset = avg_df[avg_df["setting"] == setting]
#     # For a nicely spaced annotation, we'll shift slightly to the right
#     for x, y in zip(subset["ckpt"], subset["accuracy"]):
#         plt.text(x, y, f"{y:.3f}", va='bottom', ha='center', fontsize=9)

# plt.title("Average Accuracy vs Training Steps for Different Settings")
# plt.xlabel("Training Steps (ckpt)")
# plt.ylabel("Average Accuracy")
# plt.ylim(0.44, 0.53)
# plt.tight_layout()
# plt.savefig("tem.png")
