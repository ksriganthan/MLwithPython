
def main():
    file_object = open("charge_accounts.txt", "r")
    file = file_object.read()
    file_object.close()

    my_List = file.split("\n")
    inputNum = input("Enter the account number: ")

    if inputNum in my_List:
        print("The number is valid")
    else:
        print("The number is invalid")


if __name__ == "__main__":
    main()