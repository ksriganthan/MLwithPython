
file_object = open("test.txt", "r")
file_object2 = open("testtt.txt", "a") #Wenn das File existiert wird der Inhalt um den aktuellen ergänzt

firstLineReached = False
for line in file_object:
    file_object2.write(line)
    print(line.rstrip()) #mitrstrip()

# Bei write(line) gibt es automatisch einen Absatz nach einer Zeile
# Bei print gibt es einen zusätzlichen Absatz -> dieser wird mit rstrip() gelöscht




