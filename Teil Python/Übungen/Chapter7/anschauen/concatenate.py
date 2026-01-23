# This program concatenates strings.

def main():
    name = 'Carmen'
    print(f'The name is: {name}')
    print("{} characters long.".format(len(name)))
    name = name + ' Brown'
    print(f'Now the name is: {name}')

# Call the main function.
if __name__ == '__main__':
    main()