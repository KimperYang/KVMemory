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
        "1": 1.9066307102404385,
        "2": 1.9088274213437089,
        "3": 1.9120724506456843,
        "4": 1.9154816194067519,
        "5": 1.918934540668811,
        "6": 1.9215957291271843,
        "7": 1.9248291780677682,
        "8": 1.9280335223320835,
        "9": 1.93110067622658,
        "10": 1.9346309569097326
    
}
# accuracy_claude_13_100k = [100, 95, 93, 94]
# accuracy_gpt = [100, 92, 90, 91]

# Plotting
plt.figure(figsize=(6, 4))

# Plot each line with different styles and markers
# plt.plot(loss.keys(), loss.values(), label='Original Model', marker='o', linestyle='-', color='cornflowerblue', linewidth=2)
# plt.plot(loss2.keys(), loss2.values(), label='With position info', marker='o', linestyle='-', color='tan', linewidth=2)
plt.plot(loss3.keys(), loss3.values(), label='Finetuned KVMemory Model', marker='o', linestyle='-', color='slategray', linewidth=2)
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
