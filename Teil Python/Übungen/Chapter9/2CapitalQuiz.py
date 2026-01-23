from random import randint

def main():
    laender_und_hauptstaedte = {
        "Schweiz": "Bern",
        "Deutschland": "Berlin",
        "Österreich": "Wien",
        "Frankreich": "Paris",
        "Italien": "Rom",
        "Spanien": "Madrid",
        "Portugal": "Lissabon",
        "Niederlande": "Amsterdam",
        "Belgien": "Brüssel",
        "Grossbritannien": "London"
    }

    answer = 0
    correct = 0
    false = 0
    while answer != "STOP":
        randomZahl = randint(0, len(laender_und_hauptstaedte) -1)
        randomLand = list(laender_und_hauptstaedte.keys())[randomZahl]
        answer = input("Was ist die Hauptstadt von " + randomLand + ": \n")

        if answer == laender_und_hauptstaedte.get(randomLand):
            correct = correct + 1
            print("Correct")
        else:
            false = false + 1
            print("Incorrect")
    print("Your stats: ")
    print("Correct\t\tIncorrect")
    print(correct,"          ", false - 1)


if __name__ == "__main__":
    main()

