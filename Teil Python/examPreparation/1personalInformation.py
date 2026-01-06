from random import randint

my_dict = {}

for i in range(100):
    randomNum = randint(1,10)
    if randomNum not in my_dict:
        my_dict[randomNum] = 1
    else:
        my_dict[randomNum] += 1

my_newDict = dict(sorted(my_dict.items()))

print(type(my_newDict.keys()))
print(type(my_newDict.values()))
print(type(my_newDict.items()))

for key, value in my_newDict.items():
     print(key, "\t\t",value)

