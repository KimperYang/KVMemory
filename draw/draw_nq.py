import matplotlib.pyplot as plt

# Sample data for plotting
x = ['0th', '4th', '9th']
accuracy_claude_13 = [155 / 2655  * 100, 173 / 2655  * 100, 153 / 2655  * 100]
accuracy_claude_13_100k = [1716 / 2655  * 100, 1571 / 2655  * 100, 1561 / 2655  * 100]
accuracy_gpt = [31.75, 50 / 479 * 100, 54 / 481 * 100]

# Plotting
plt.figure(figsize=(6, 4))

# Plot each line with different styles and markers
plt.plot(x, accuracy_claude_13, label='KV Memory(10 Doc)', marker='o', linestyle='-', color='cornflowerblue', linewidth=2)
plt.plot(x, accuracy_claude_13_100k, label='Upper Bound(10 Doc)', marker='o', linestyle='-', color='tan', linewidth=2)
plt.plot(x, accuracy_gpt, label='KV Memory(3 Doc)', marker='o', linestyle='-', color='slategray', linewidth=2)

# Customizing the plot
plt.title('Natural Questions')
plt.xlabel('Position of the Answer')
plt.ylabel('Accuracy')
plt.ylim([0, 80])

# Adding legend
plt.legend()

# Display the plot
plt.tight_layout()
plt.savefig('../result/figure/nq_res.png',dpi = 300)
