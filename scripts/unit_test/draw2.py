import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager
# ------------------------------------------------------
# 1. Create Some Example Data (Dummy Data)
# ------------------------------------------------------
data = {
    "answer_position": [ "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
    "accuracy": [66.2, 56.8, 56.2, 52.4, 50.8, 50.2, 50.2, 50.6, 51.8, 48],
}

df = pd.DataFrame(data)

# ---------------------------------------------
# 2. Set a Seaborn Theme & Color Palette
# ---------------------------------------------
# context='talk' gives slightly larger fonts suitable for presentation
# style='whitegrid' or 'darkgrid' etc. can be chosen
sns.set_theme(context='talk', style='whitegrid')

# Define a custom color palette for our three methods
custom_palette = ["#FFBF61"]  # or any other set of colors

# ---------------------------------------------
# 3. Create a Figure with a Custom Background
# ---------------------------------------------

font_manager.fontManager.addfont("scripts/unit_test/Consolas.ttf")
# plt.rcParams['font.sans-serif'] = 'Consolas'
plt.rcParams['font.family'] = 'Consolas'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

plt.rcParams['axes.titlesize'] = 24
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 18


fig, ax = plt.subplots(figsize=(10, 7.5), facecolor="#ffffff")  # figure background
ax.set_facecolor("#ffffff")  # axis background

# ---------------------------------------------
# 4. Plot the Data
# ---------------------------------------------
sns.barplot(
    data=df,
    x="answer_position",
    y="accuracy",
    # hue="method",
    palette=custom_palette,
    # marker="o",
    # linewidth=2,      # Thicker line
    # markersize=8,     # Larger markers
    ax=ax
)

# ---------------------------------------------
# 5. Customize Labels and Title
# ---------------------------------------------
# ax.set_title("TTFT with Ten Reused Contexts", pad=15, color="#000000")
ax.set_xlabel("Answer Document Index", color="#000000")
ax.set_ylabel("Accuracy", color="#000000")

# Optionally, customize tick labels
ax.tick_params(axis='x', colors='#000000')
ax.tick_params(axis='y', colors='#000000')

# ---------------------------------------------
# 6. Customize Legend
# ---------------------------------------------
legend = ax.legend( facecolor='white')
# plt.ylim(40, 65)
ax.set_ylim(40, 70)
# ---------------------------------------------
# 7. Adjust Layout and Show the Plot
# ---------------------------------------------
    # Add one legend below all subplots
handles, labels = plt.gca().get_legend_handles_labels()
# plt.figlegend(handles, labels, loc='lower center', ncol=3, fontsize=FONT_SIZE - 5, title="Model Scaling")
# plt.figlegend(handles, labels, loc='lower center', ncol=3, fontsize=FONT_SIZE - 5, fancybox=True, shadow=True, framealpha=0.95)
# plt.figlegend(handles, labels, loc='lower center', ncol=3, fontsize=20, fancybox=True, shadow=True, framealpha=0.95)
plt.tight_layout(rect=[0, 0.06, 1, 1])

# plt.rcParams.update({
#     'font.size': 20,          # base font size
#     'axes.titlesize': 22,     # title font size
#     'axes.labelsize': 20,     # x and y axis labels
#     'xtick.labelsize': 18,    # x tick labels
#     'ytick.labelsize': 18,    # y tick labels
#     'legend.fontsize': 18,    # legend font size
#     'axes.titleweight': 'bold',   # make titles bold
#     'axes.labelweight': 'bold'    # make label text bold
# })

plt.savefig('block_nq.pdf')
# plt.savefig('time.png')
