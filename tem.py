import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Example data
training_steps = list(range(2000, 32000, 2000))  # Training steps from 2000 to 30000 in every 2000 steps
accuracy_setting_1 = [0.336, 0.314, 0.28, 0.306, 0.388, 0.426, 0.388, 0.448, 0.446, 0.48, 0.454, 0.46, 0.46, 0.458, 0.47]
accuracy_setting_2 = [0.378, 0.34, 0.316, 0.336, 0.39, 0.414, 0.412, 0.44, 0.414, 0.452, 0.44, 0.444, 0.46, 0.448, 0.45]
accuracy_setting_3 = [0.35, 0.314, 0.29, 0.266, 0.316, 0.354, 0.41, 0.39, 0.4, 0.37, 0.358, 0.366, 0.366, 0.378, 0.384]

# Baseline accuracy
baseline_accuracy1 = 0.552
baseline_accuracy2 = 0.432
baseline_accuracy3 = 0.42

# Create a dictionary to hold the data for seaborn
data = {
    'Training Steps': training_steps * 3,
    'Accuracy': accuracy_setting_1 + accuracy_setting_2 + accuracy_setting_3,
    'Setting': ['0th'] * len(accuracy_setting_1) + ['4th'] * len(accuracy_setting_2) + ['9th'] * len(accuracy_setting_3)
}

# Convert data to DataFrame
import pandas as pd
df = pd.DataFrame(data)

# Plot using seaborn
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
# Plot each setting separately to control color
sns.lineplot(x=training_steps, y=accuracy_setting_1, marker="o", color='red', label="0th")
sns.lineplot(x=training_steps, y=accuracy_setting_2, marker="o", color='blue', label="4th")
sns.lineplot(x=training_steps, y=accuracy_setting_3, marker="o", color='green', label="9th")

# Add baseline line
plt.axhline(y=baseline_accuracy1, color='red', linestyle='--', label='0th')
# Add baseline line
plt.axhline(y=baseline_accuracy2, color='blue', linestyle='--', label='4th')
# Add baseline line
plt.axhline(y=baseline_accuracy3, color='green', linestyle='--', label='9th')

# Add labels and legend
plt.title("Accuracy over Training Steps")
plt.xlabel("Training Steps")
plt.ylabel("Accuracy")
plt.legend()

# Save the figure
plt.savefig("accuracy_over_training_steps.png")
