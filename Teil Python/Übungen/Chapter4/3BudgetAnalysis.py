budget = float(input("Enter the amount that you have budgeted for a month: "))

totalExpenses = 0.0

numOfLaps = int(input("Enter the number of your expenses: "))

for i in range(numOfLaps):
    expense = float(input("Enter the next expense: "))
    totalExpenses += expense


print(f"The amount which is still left: {budget - totalExpenses}")