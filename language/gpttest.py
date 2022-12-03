# Calculate the relative frequencies of each item in the list
lst = [4, 4, 4, 1, 5, 5, 1, 1, 1, 4, 4, 5, 5]
counts = {}
for x in lst:
    counts[x] = counts.get(x, 0) + 1
total = len(lst)
freqs = {x: counts[x]/total for x in counts}

# Calculate the number of each item to keep in the shortened list
n = 8
numbers = {x: n * freqs[x] for x in freqs}

# Round the number of each item to keep to the nearest integer
numbers = {x: round(numbers[x]) for x in numbers}

# Keep the first n occurrences of each item in the list
shortened_list = []
for x in lst:
    if numbers[x] > 0:
        shortened_list.append(x)
        numbers[x] -= 1

# Print the shortened list
print(shortened_list)