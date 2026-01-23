num = input("Enter a series of single-digit numbers with nothing separating them: ")

_sum = 0

for n in num:
    _sum += int(n)

print(_sum)