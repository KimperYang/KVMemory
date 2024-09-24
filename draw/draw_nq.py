import matplotlib.pyplot as plt

# Sample data for plotting
loss = {
        "1": 1.8297557709654961,
        "2": 2.0815369892278,
        "3": 2.0788785890637045,
        "4": 2.0396675860394025,
        "5": 2.021290003316488,
        "6": 2.0193761908551475,
        "7": 2.028433900434308,
        "8": 2.040873513599772,
        "9": 2.056359146009692,
        "10": 2.0697534081866285
    }

# loss2 = {
#         "1": 1.8298580955245538,
#         "2": 1.8508882448058612,
#         "3": 1.871770836280923,
#         "4": 1.8920899423382092,
#         "5": 1.9135232144639551,
#         "6": 1.9347833750654242,
#         "7": 1.9562568763408534,
#         "8": 1.9765955668664026,
#         "9": 1.996237723301806,
#         "10": 2.0163359790668633
#     }

loss3 = {
        "1": 1.8243271416748414,
        "2": 1.8285643948546932,
        "3": 1.8335045022730103,
        "4": 1.8383430682992685,
        "5": 1.8438561746177677,
        "6": 1.8486796359804845,
        "7": 1.8545910325403252,
        "8": 1.8601401515152611,
        "9": 1.865784920560579,
        "10": 1.8724461120912088
    }
# accuracy_claude_13_100k = [100, 95, 93, 94]
# accuracy_gpt = [100, 92, 90, 91]

# Plotting
plt.figure(figsize=(6, 4))

# Plot each line with different styles and markers
plt.plot(loss.keys(), loss.values(), label='Original Model', marker='o', linestyle='-', color='cornflowerblue', linewidth=2)
# plt.plot(loss2.keys(), loss2.values(), label='With position info', marker='o', linestyle='-', color='tan', linewidth=2)
plt.plot(loss3.keys(), loss3.values(), label='Finetuned Model', marker='o', linestyle='-', color='slategray', linewidth=2)
plt.axhline(y=2.0528990041590975, color='red', linestyle='--', linewidth=2, label='Baseline')
plt.axhline(y=1.813626458174075, color='green', linestyle='--', linewidth=2, label='UpperBound')
# Customizing the plot
plt.title('Average Next Token Loss on 2000 Data Samples')
plt.xlabel('Num of Memorys')
plt.ylabel('Loss')
plt.ylim([1.80, 2.20])

# Adding legend
plt.legend(loc='upper right')

# Display the plot
plt.tight_layout()
plt.savefig('finetune.png')

# import matplotlib.pyplot as plt

# # Sample data for plotting
# x = ['0th', '4th', '9th']
# accuracy_claude_13 = [64.6, 59.2, 58.8]
# accuracy_claude_13_100k = [20.8, 24.6, 49.4]
# accuracy_gpt = [5, 6, 5]

# # Plotting
# plt.figure(figsize=(6, 4))

# # Plot each line with different styles and markers
# plt.plot(x, accuracy_claude_13, label='normal', marker='o', linestyle='-', color='cornflowerblue', linewidth=2)
# plt.plot(x, accuracy_claude_13_100k, label='kvmemory_cheat', marker='o', linestyle='-', color='tan', linewidth=2)
# plt.plot(x, accuracy_gpt, label='kv_memory', marker='o', linestyle='-', color='slategray', linewidth=2)

# # Customizing the plot
# plt.title('QA Accuracy')
# plt.xlabel('Position')
# plt.ylabel('Accuracy')
# plt.ylim([0, 80])

# # Adding legend
# plt.legend()

# # Display the plot
# plt.tight_layout()
# plt.savefig('nq.png')
