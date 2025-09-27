
num1 = float(input("Enter the first number: "))
num2 = float(input("Enter the second number: "))
num3 = float(input("Enter the third number: "))

nums =[num1,num2,num3]

def max(nums):
    index = 0
    highestValue = -1

    for i in range(0,len(nums)):
        if nums[i] > highestValue:
            index = i
            highestValue = nums[i]

    return highestValue

max(nums)

highestNum = max(nums)

print(f"The highest value is: {highestNum}")
