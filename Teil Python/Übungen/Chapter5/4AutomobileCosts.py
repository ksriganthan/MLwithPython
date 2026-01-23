userInput = input("Enter the monthly costs for the following expenses incurred from operating from your automobile (seperated with a komma): \n")
parts = userInput.split(",") # A B C ...

def autoMobileCosts(*strings): #Bedeutet mehrere Strings
    sum = 0.0
    for c in strings:
        sum += float(c)

    monthlyInput = sum
    yearlyInput = monthlyInput * 12
    print(f"The monthly costs are: {monthlyInput:.2f}\nThe yearly costs are: {yearlyInput:.2f}")

autoMobileCosts(*parts)
print(type(parts))