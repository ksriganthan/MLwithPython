
year = int(input("Enter the year: "))
leapYear = 29
normalYear = 28
daysInFebruary = 0

if year % 100 == 0:
    if year % 400 == 0:
        daysInFebruary = leapYear
    else:
        daysInFebruary = normalYear
else:
    if year % 4 == 0:
        daysInFebruary = leapYear
    else:
        daysInFebruary = normalYear

print(f"In {year} February has {daysInFebruary} days.")