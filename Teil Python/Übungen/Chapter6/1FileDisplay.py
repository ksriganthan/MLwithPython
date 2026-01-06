
def main():
    file_object = open("numbers.txt", "r")

    content = file_object.read()

    file_object.close()

    print(content)



if __name__ == "__main__":
    main()