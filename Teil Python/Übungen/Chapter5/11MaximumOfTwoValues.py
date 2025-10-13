

num1 = float(input("Enter the first number: "))
num2 = float(input("Enter the second number: "))

def max(a,b):
    if a > b:
     return a
    elif a < b:
     return b
    else:
     return a

print(f"The greater number is: {max(num1,num2)}")


