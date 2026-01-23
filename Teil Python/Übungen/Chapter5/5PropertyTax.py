
actualValuePieceProperty = float(input("Please enter the actual value of the piece:\n"))

def collector(actual_value):
    propertytax = 0.72
    assessment_value = actualValuePieceProperty * 0.6
    tax = assessment_value / 100 * propertytax
    print(
        f"The actual value: {actual_value:.2f}\nThe assessment value: {assessment_value}\nPropertyTax: {propertytax}\nTax total: {tax:.2f}")

collector(actualValuePieceProperty)

