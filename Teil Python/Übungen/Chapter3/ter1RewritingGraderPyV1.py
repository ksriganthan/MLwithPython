# This program gets a numeric test score from the
# user and displays the corresponding letter grade.

# Variables to represent the grade thresholds
A_score = 90
B_score = 80
C_score = 70
D_score = 60

# Get a test score from the user.
score = int(input('Enter your test score: '))

# Determine the grade.
if score <= 60:
    print("Your grade is D")
elif 60<score<=79:
    print("Your grade is C")
elif 79 < score < 90:
    print("Your grade is B")
else:
    print("Your grade is A")

