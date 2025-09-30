
my_dict = {}


def readFile():
    with open("my_text.txt", "r") as file:
        for line in file:
            words = line.split()
            for word in words:
                if word not in my_dict.keys():
                    my_dict[word] = 1
                else:
                    my_dict[word] = my_dict.get(word) + 1


def writeFile():
    with open("wordFrequency_my_text.txt", "w") as file:
        for key, value in my_dict.items():
            file.write(key + " : " + str(value) + "\n")


def main():
    readFile()
    writeFile()


if __name__ == "__main__":
    main()



