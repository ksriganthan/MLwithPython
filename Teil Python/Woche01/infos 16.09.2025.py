#Es gibt bei gitignore Files die nicht im Git sind
#z.B. idea files und venv
#Package turtle ist nicht relevant für die Prüfung

# // floor division
print(10 / 3)
print(10 // 3)  # Rundet ab

#Exercise 3-7
# Get point for the tests and exam from the user.
test1Point = int(input('Enter points out of 25 for Test 1: '))
test2Point = int(input('Enter points out of 25 for Test 2: '))
examPoint = int(input('Enter points out of 50 for the Exam: '))

# Check if any of the points are invalid.
#Backslash hier ist, um zu zeigen, dass eigentlich alles auf einer Zeile wäre
if (test1Point < 0 or test1Point > 25) or \
        (test2Point < 0 or test2Point > 25) or \
        (examPoint < 0 or examPoint > 50):
    print('Error: Invalid points entered.')

# If the points were valid...
else:
    # Determine the total point and display it.
    totalPoint = test1Point + test2Point + examPoint
    print('Total points:', totalPoint)

    # Determine the grade and display it.
    if totalPoint < 50 or examPoint < 25:
        print('Fail')
    elif totalPoint >= 50 and totalPoint <= 59:
        print('Pass')
    elif totalPoint >= 60 and totalPoint <= 79:
        print('Credit')
    elif totalPoint >= 80 and totalPoint <= 100:
        print('Distinction')

#Python hat keine do ... while, aber dafür hat die for-Schleife eine grössere Bedeutung

# Tabulatoren: \t

#Ein Modul ist eine Datei mit Python-Code
#Ein Package ist ein Ordner mit mehreren Modulen - muss mit PIP installiert werden
#Ein Library ist eine Sammlung von Modulen und Packages


#Requirements.txt
#Ist eine Datei mit allen benötigten Modulen, um die Repository vollumfänglich benutzen zu können

#PIP
#Ist der Python Package Installer (Manager für Python Module)
