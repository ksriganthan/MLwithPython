distanceInKilometer = float(int(input("Enter a distance in kilometers: ")))


def kilometerConverter(kilometer):
    return kilometer * 0.6214

miles = kilometerConverter(distanceInKilometer)

print(f"The kilometer {distanceInKilometer} are {miles:.2f} miles")

