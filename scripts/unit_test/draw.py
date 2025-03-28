import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager
# ------------------------------------------------------
# 1. Create Some Example Data (Dummy Data)
# ------------------------------------------------------
data = {
    "length": [
        "10 * 100", "10 * 100", "10 * 100",
        "10 * 200", "10 * 200", "10 * 200",
        "10 * 300", "10 * 300", "10 * 300",
        "10 * 400", "10 * 400", "10 * 400",
        "10 * 500", "10 * 500", "10 * 500"
    ],
    "time_to_first_token": [
        0.0273, 0.0266, 0.074,
        0.0273, 0.0270, 0.1188,
        0.0282, 0.0274, 0.1721,
        0.0291, 0.0284, 0.2197,
        0.0291, 0.0293, 0.2670
    ],
    "method": [
        "MemorySum5", "MemorySum1", "NormalDecoding",
        "MemorySum5", "MemorySum1", "NormalDecoding",
        "MemorySum5", "MemorySum1", "NormalDecoding",
        "MemorySum5", "MemorySum1", "NormalDecoding",
        "MemorySum5", "MemorySum1", "NormalDecoding"
    ]
}

data2 = {
    "context number": [
        2, 2, 2,
        4, 4, 4,
        6, 6, 6,
        8, 8, 8,
        10, 10, 10
    ],
    "time_to_first_token": [
        0.0303, 0.0304, 0.3403,
        0.0309, 0.0305, 0.3398,
        0.0315, 0.0306, 0.3396,
        0.0307, 0.0302, 0.3402,
        0.0301, 0.0303, 0.3395
    ],
    "method": [
        "MemorySum5", "MemorySum1", "NormalDecoding",
        "MemorySum5", "MemorySum1", "NormalDecoding",
        "MemorySum5", "MemorySum1", "NormalDecoding",
        "MemorySum5", "MemorySum1", "NormalDecoding",
        "MemorySum5", "MemorySum1", "NormalDecoding"
    ]
}

df = pd.DataFrame(data)

# ---------------------------------------------
# 2. Set a Seaborn Theme & Color Palette
# ---------------------------------------------
# context='talk' gives slightly larger fonts suitable for presentation
# style='whitegrid' or 'darkgrid' etc. can be chosen
sns.set_theme(context='talk', style='whitegrid')

# Define a custom color palette for our three methods
custom_palette = ["#612DBD", "#459B99", "#0E2747"]  # or any other set of colors

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

plt.grid()
fig, ax = plt.subplots(figsize=(10, 7.5), facecolor="#ffffff")  # figure background
ax.set_facecolor("#ffffff")  # axis background

# ---------------------------------------------
# 4. Plot the Data
# ---------------------------------------------
sns.barplot(
    data=df,
    x="length",
    y="time_to_first_token",
    hue="method",
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
ax.set_xlabel("Context Length (tokens)", color="#000000")
ax.set_ylabel("TTFT (seconds)", color="#000000")

# Optionally, customize tick labels
ax.tick_params(axis='x', colors='#000000')
ax.tick_params(axis='y', colors='#000000')

# ---------------------------------------------
# 6. Customize Legend
# ---------------------------------------------
legend = ax.legend( facecolor='white')

# ---------------------------------------------
# 7. Adjust Layout and Show the Plot
# ---------------------------------------------
    # Add one legend below all subplots
plt.tight_layout(rect=[0, 0.06, 1, 1])

plt.savefig('time.png')
