from random import  randint

# Mit ->Type gibt man den return-type an für die Lesbarkeit
def create_numbers() ->list:
    my_list = []
    for i in range(100):
        my_list.append(randint(1,100))
    return my_list

def determine_frequency(liste):
    # Als Key können nur Immutable-Werte verwendet werden (Integer, String, Tupel)
    my_dict = {}
    for i in range(len(liste)):
        random_number = liste[i]
        if random_number not in my_dict:
            my_dict[random_number] = 1
        else:
            my_dict[random_number] += 1
    return  my_dict

def print_table(dicti):
    my_dict_sorted = dict(sorted(dicti.items()))
    print("Number\tFrequency")
    print("-----------------")
    for number,frequency in my_dict_sorted.items():
        print(f"{number}\t\t{frequency}")

if __name__ == "__main__":
    my_list = create_numbers()
    my_dict = determine_frequency(my_list)
    print_table(my_dict)
