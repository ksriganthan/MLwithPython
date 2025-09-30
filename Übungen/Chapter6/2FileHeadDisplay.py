
def main():
    file_name = input("Please enter the file-name:\n")
    file_object = open(file_name, "r")

    for i in range(1, 6):
        line = file_object.readline().rstrip() #rstrip, um den zusätzlichen Absatz nach readline zu löschen
        if line == "":
         break
        print(line)

    file_object.close()


if __name__ == "__main__":
    main()

