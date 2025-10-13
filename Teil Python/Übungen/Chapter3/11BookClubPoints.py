numberOfBooks = int(input("Enter the number of books you purchased this month: "))

if numberOfBooks < 2:
    print(0)
elif numberOfBooks <= 2:
    print(5)
elif numberOfBooks <= 4:
    print(15)
elif numberOfBooks <= 6:
    print(30)
else:
    print(60)

