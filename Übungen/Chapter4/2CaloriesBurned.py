caloriesPerMinute = 4.2
minutes = [10,15,20,25,30]
caloriesBurned = {}

for num in minutes:
    caloriesBurned[num] = num * caloriesPerMinute

print('Minutes\t\tCalories Burned')
print('-------------------------------')
for key in caloriesBurned.keys():
    print(str(key)+"\t\t\t"+str(caloriesBurned.get(key)))
