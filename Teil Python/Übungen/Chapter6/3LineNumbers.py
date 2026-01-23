
def main():
    file_name = input("Please enter the file-name:\n")
    file_object = open(file_name,"r")
    line = file_object.readline()
    counter = 1

    while line != "":
        line = line.rstrip()
        print(str(counter) + ": " + line)
        counter = counter++1
        line = file_object.readline()
    file_object.close()

if __name__ == "__main__":
    main()
