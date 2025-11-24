
with open("my_text.txt", "r") as file:
    set_ = set()
    for line in file:
        words = line.split()
        for word in words:
            set_.add(word)

    print(set_)

