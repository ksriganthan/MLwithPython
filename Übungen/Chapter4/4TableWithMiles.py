
values = [10,20,30,40,50,60,70,80]
net = 1.60934
milesToKilometers = {}

for i in range(0, len(values)):
    milesToKilometers[values[i]] = format((values[i] * net),".2f")


print("Miles\t\tKilometers")
for key in milesToKilometers.keys():
    print(str(key) + "\t\t\t" + str(milesToKilometers.get(key)))
