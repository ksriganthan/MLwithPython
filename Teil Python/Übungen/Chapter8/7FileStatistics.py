
with open("text.txt", "r") as file:
    upperCase = 0
    LowerCase = 0
    digit = 0
    space = 0
    for line in file:
        for char in line:
            if char.isupper():
                upperCase +=1
            if char.islower():
                LowerCase += 1
            if char.isdigit():
                digit += 1
            if char.isspace():
                space += 1

    dict_ = {}
    dict_["UpperCase"] = upperCase
    dict_["LowerCase"] = LowerCase
    print(dict_.get("LowerCase"))
    dict_["Digit"] = digit
    dict_["Space"] = space

    print(dict_)
