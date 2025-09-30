def _calculate():
    _sum = 0

    try:
        with open('numbers.txt', 'r') as file:
            for line in file:
                line = line.strip()  # White spaces in der Datei werden entfernt
                _sum += int(line)
        return _sum
    except FileNotFoundError:
        print("Datei konnte nicht gefunden werden")
        return "No result"
    except ValueError:
        print("Format der Datei ist ungültig")
        return "No result"
    finally:
        print("Calculation done")


if __name__ == "__main__":
    print(_calculate())