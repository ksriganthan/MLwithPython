from random import randint

def main():
    numOfDicing = int(input("Enter how many times the dice should be rolled: "))
    myList = []
    while numOfDicing != 0:
        num = randint(1,6)
        myList.append(num)
        numOfDicing = numOfDicing -1

    myList.sort()

    print(myList)


if __name__ == "__main__":
    main()
