import random

def main():
    file_name = input("Please enter the output-file name:\n")
    numberOfRandomNumbers = int(input("Please enter the number of random numbers you want to write:\n"))

    file_object = open(file_name,"w") # Überschreibt das File oder erstellt ein neues

    for i in range(0, numberOfRandomNumbers):
        file_object.write(str(random.randint(1, 500)) + "\n") # 500 ist Inklusive!

    file_object.close()

if __name__ == "__main__":
    main()

