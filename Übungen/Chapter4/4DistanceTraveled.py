perHour = 40
speedOfTheVehicle = float(input("Enter the speed of your vehicle: "))
hoursTravelled = int(input("Enter the hour you have travelled with the verhicle: "))

print("Hour\t\tDistance Traveled")
for i in range(1, hoursTravelled +1):
    print(str(i) + "\t\t\t" + str(i * 40))