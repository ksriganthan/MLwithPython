
userInput = input("Enter a String: ")

myDict = {}

for char in userInput:
    if char not in myDict.keys():
        myDict[char] = 1
    else:
        myDict[char] = myDict.get(char) +1

mostKey = ""
frequency = 0

for key, value in myDict.items():
    if value > frequency:
        frequency = value
        mostKey = key

print("Letter: ", mostKey, " Frequency: ", frequency)