
numOfFeets = float(input("Please enter a number of feet:\n"))

def named_feet_to_inches(input):
    foot = 12
    inches = input * foot
    print("{} feet are {} inches.".format(input, inches))

named_feet_to_inches(numOfFeets)
