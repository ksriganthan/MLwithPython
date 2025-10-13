import sympy as symp

num = int(input("Enter an integer: "))

bool = symp.isprime(num)

if bool:
    print("The number is a prime")
else:
    print("The number is not a prime")