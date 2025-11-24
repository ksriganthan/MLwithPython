def main():
    file_object = open("numbers.txt", "r")
    sum = 0

    content = file_object.read()  # Zuerst die ganze Datei einlesen

    for char in content:
        if char.isnumeric():
            sum += int(char)

    file_object.close()
    print(sum)


if __name__ == "__main__":
    main()


