length1 = float(input("Please enter the length of the first rectangle: "))
width1 = float(input("Please enter the width of the first rectangle: "))

length2 = float(input("Please enter the length of the second rectangle: "))
width2 = float(input("Please enter the width of the second rectangle: "))

area1 = length1 * width1
area2 = length2 * width2

if area1 > area2:
    print("The first rectangle is bigger")
elif area1 < area2:
    print("The second rectangle is bigger")
else:
    print("Both rectangle have the same size")