
full_name = input("Enter the full name: ")

words = full_name.split()

newWord = ""

for word in words:
    newWord = newWord +  word[0] + "."

print(newWord)
