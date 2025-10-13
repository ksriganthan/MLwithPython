
numberOfPackages = int(input("Enter the number of packages you purchased: "))
price = 99
discount = 0.0

if 10 <= numberOfPackages <= 19:
    discount = 0.1
elif 20 <= numberOfPackages <= 49:
    discount = 0.2
elif 50 <= numberOfPackages <= 99:
    discount = 0.3
elif numberOfPackages >= 100:
    discount = 0.4

discountAmount = numberOfPackages * price * discount
totalAmount = (numberOfPackages * price) - discountAmount

print(f"The amount of discount: {discountAmount:.2f}\n"
      f"The total amount of the purchase after the discount: {totalAmount:.2f}")

