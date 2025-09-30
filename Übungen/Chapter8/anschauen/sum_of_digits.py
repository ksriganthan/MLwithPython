# Ziffernsumme berechnen
my_string = input("String eingeben, um Zeichensumme zu berechnen: ")

print(my_string)

_sum = 0

for digit in my_string:
    _sum += int(digit)
print(_sum)