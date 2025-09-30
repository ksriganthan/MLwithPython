# Slicing with Strings

s = "Telesko"

print(s[1:5]) # eles
print(s[1:]) # elesko
print(s[:1]) # T
print(s[:]) # Telesko
print(s[::]) # Telesko
print(s[:-1]) # Telesk
print(s[:-2]) # Teles
print(s[::-1]) # okseleT
print(s[-4:]) # esko - 4. vom Ende fängt er an und geht bis Ende
print(s[-4:-2]) # es - von Minus 4 bis Minus 2
print(s[:-4]) # Tel
print(s[::len(s)-1]) # To - nur der erste und der letzte Buchstabe
print(s[-4:0:-2]) # ee

# S[start:stop:ink]
# Alle Werte könnten negativ sein
# Standard-Inkrement = 1
# Stop ist exklusiv
# Bei negativ Index: die letzte Buchstabe ist -1
print(s[-2::6])