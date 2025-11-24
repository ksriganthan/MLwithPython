try:
    with open("text.txt", "r") as file:
        lines = 0
        _sum = 0
        for line in file:
            lines += 1
            _sum += len(line.split())

    print(_sum / lines)

except IOError:
    print("Error reading file")
except ValueError:
    print("Error casting")
except:
    print("Error")
