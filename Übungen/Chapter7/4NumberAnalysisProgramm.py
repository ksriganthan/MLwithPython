from random import randint
from numpy import average

my_list = [randint(0,100) for _ in range(20)]

my_dict = {"Lowest": min(my_list),
           "Highest": max(my_list),
           "Sum": sum(my_list),
           "Average": average(my_list)}


for key, value in my_dict.items():
    print(key)
    print(value)
    print("-------")

