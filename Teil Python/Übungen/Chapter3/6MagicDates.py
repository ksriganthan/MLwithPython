month = int(input("Enter a month in numeric form: "))
day = int(input("Enter a day in numeric form: "))
year = int(input("Enter a year in numeric form and as a two-digit: "))

if month < 1 or month > 12:
    print("Month is not valid")
if day < 1 or day > 31:
    print("Day is not valid")
if year < 0 or year > 99:
    print("Year is not valid")
else:
# Input is valid
    print(f"The date is: {month:02d}/{day:02d}/{year}")
    if month * day == year:
        print("The date is magic")
    else:
        print("The date isn't magic")