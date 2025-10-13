
with open("text.txt", "r") as file:
    upperCase = 0
    LowerCase = 0
    digit = 0
    space = 0
    for line in file:
        for char in line:
            if char.isupper():
                upperCase = upperCase +1
            if char.islower():
                LowerCase = LowerCase + 1
            if char.isdigit():
                digit = digit + 1
            if char.isspace():
                space = space + 1

    dict_ = {}
    dict_["UpperCase"] = upperCase
    dict_["LowerCase"] = LowerCase
    dict_["Digit"] = digit
    dict_["Space"] = space

    print(dict_)
