def main():
    input_name = input("Bitte Dateinamen eingeben: ")

    # File handling -> with ist hier der Context-Handler
    with open(input_name,"r") as input_file:
        text = input_file.read()
        words = text.split()

        unique_words = set(words) #Case-sensitive!

        print("Unique Words: ", unique_words)


if __name__ == '__main__':
    main()
