
file_object = open("test.txt","r")
file_object2 = open("test.txt", "a")

firstLineReached = False
for line in file_object:
    if not firstLineReached:
       file_object2.write("\n")
       file_object2.write(line)
       firstLineReached = True
    else:
        file_object2.write(line)
    print(line.rstrip()) #mitrstrip() den zusätzlichen Absatz nach jeder Zeile wieder löschen




