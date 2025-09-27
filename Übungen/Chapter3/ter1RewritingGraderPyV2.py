# This program gets a numeric test score from the
# user and displays the corresponding letter grade.

# Variables to represent the grade thresholds
A_score = 90
B_score = 80
C_score = 70
D_score = 60

# Get a test score from the user.
score = int(input('Enter your test score: '))

grade = ""
# Determine the grade.

#Um Zwischenbereiche abzudecken:
#case 90 if score >= 90 ...

match score:
    case 90:
        grade = grade + " " + str(A_score)
    case 80:
        grade = grade + " " + str(B_score)
    case 70:
        grade = grade + " " + str(C_score)
    case 60:
        grade = grade + " " + str(D_score)
    case _:
        grade = grade + "Unknown"

print(f"The grade is: {grade}")

match score:
    case int(score) if score >= A_score:
        print('Your grade is A.')
    case int(score) if score >= B_score:
        print('Your grade is B.')
    case int(score) if score >= C_score:
        print('Your grade is C.')
    case int(score) if score >= D_score:
        print('Your grade is D.')
    case _:
        print('Your grade is F.')

