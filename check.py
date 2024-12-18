my_list = [10, 20, 30, 40, 50]
n = 0

# Remove element at index 0
element = my_list.pop(0)  # element = 10, my_list now = [20, 30, 40, 50]

# Insert it at the new index (n = 2)
my_list.insert(n, element)  # my_list = [20, 30, 10, 40, 50]

print(my_list)
# Output: [20, 30, 10, 40, 50]
