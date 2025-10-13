def main():
    try:
        with open("numbers.txt", "r") as file_object:
            total = 0.0
            for line in file_object:
                line = line.rstrip()
                if line != "":
                    for char in line:
                        if char.isnumeric():
                            total += int(char)
            print(total)

    except IOError:
        print("Reading Error")
    except ValueError:
        print("Error data type")
    except:
        print("Error") #Zum Beispiel wenn ich Buchstabe teilen möchten


if __name__ == "__main__":
    main()
