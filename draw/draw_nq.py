import matplotlib.pyplot as plt

# Sample data for plotting
# loss = {
#         "1": 1.8297557709654961,
#         "2": 2.0815369892278,
#         "3": 2.0788785890637045,
#         "4": 2.0396675860394025,
#         "5": 2.021290003316488,
#         "6": 2.0193761908551475,
#         "7": 2.028433900434308,
#         "8": 2.040873513599772,
#         "9": 2.056359146009692,
#         "10": 2.0697534081866285
#     }

loss2 = {
        "1": 1.875140547045988,
        "2": 1.8775941343167555,
        "3": 1.8801402652119388,
        "4": 1.8825745206110611,
        "5": 1.8853234514731638,
        "6": 1.8870132734363307,
        "7": 1.889431428229306,
        "8": 1.8916498774554795,
        "9": 1.8937261783389632,
        "10": 1.895896255284057
    }

loss3 = {
        "1": 1.9003554125020872,
        "2": 1.9022951618286812,
        "3": 1.9053146421022624,
        "4": 1.9083975025818882,
        "5": 1.9115223559567671,
        "6": 1.913911387813231,
        "7": 1.9167457351557717,
        "8": 1.9194175950260814,
        "9": 1.9220569041407036,
        "10": 1.9248186036156205
    
}

# accuracy_claude_13_100k = [100, 95, 93, 94]
# accuracy_gpt = [100, 92, 90, 91]

# Plotting
plt.figure(figsize=(6, 4))

# Plot each line with different styles and markers
# plt.plot(loss.keys(), loss.values(), label='Original Model', marker='o', linestyle='-', color='cornflowerblue', linewidth=2)
plt.plot(loss2.keys(), loss2.values(), label='KVMemory(Special Token)', marker='o', linestyle='-', color='tan', linewidth=2)
plt.plot(loss3.keys(), loss3.values(), label='KVMemory(No Special Token)', marker='o', linestyle='-', color='slategray', linewidth=2)
plt.axhline(y=2.0452591440400587, color='red', linestyle='--', linewidth=2, label='Llama2-7b-chat')
plt.axhline(y=1.8297557709654961, color='green', linestyle='--', linewidth=2, label='Llama2-7b-base')
# Customizing the plot
plt.title('Average Next Token Loss on 2000 Data Samples')
plt.xlabel('Num of Memorys')
plt.ylabel('Loss')
plt.ylim([1.80, 2.20])

# Adding legend
plt.legend(loc='upper right')

# Display the plot
plt.tight_layout()
plt.savefig('finetune_cheatcombine.png')

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
