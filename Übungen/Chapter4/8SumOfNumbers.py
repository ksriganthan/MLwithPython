
num = 0
sum = 0
temp = 0
while num >= 0:
    num = float(input("Enter a series of positive numbers: "))
    sum += num
    temp = num


print(f"The sum is: {sum-temp:.2f}")