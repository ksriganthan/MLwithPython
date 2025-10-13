# This program demonstrates keyword-only function arguments.
#Parameternamen muss genau den Werten in Funktionsdefinition entsprechen

def main():
    total = sum(b=1, a=2, c=3, d=4)
    print(total)


def sum(*, a, b, c, d): #Nach * müssen alle Parameter benannt werden
    return a + b + c + d


if __name__ == '__main__':
    main()