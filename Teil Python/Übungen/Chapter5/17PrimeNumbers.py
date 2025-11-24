number = int(input("Please enter an integer: "))


def in_prime(number):

    # 1 is not being a prime number, is ignored
    if number > 1:

       for i in range(2, int(number / 2) + 1):  # Kein Teiler kann grösser als die Hälfte von number sein
            if (number % i) == 0:
             return False
       else:

        return True # Wenn man aus dieser Schleife fällt, dann ist es eine Primzahl

    else:
      return False


print(in_prime(number))

