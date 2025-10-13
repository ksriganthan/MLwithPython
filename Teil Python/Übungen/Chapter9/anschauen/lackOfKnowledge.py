#Dictionary
input_list = [1, 2, 3, 4, 5, 6, 7]

#Der key ist var und value ist var^3
dict_using_comp = {var: var ** 3 for var in input_list if var % 2 != 0}


print("Key\t\tValue")
print("-------------------")
for key,value in dict_using_comp.items():
    print("Key: ", key, " Value: ", value)


# Using Dictionary comprehensions for constructing output dictionary
player = ["Messi", "Ronaldo", "Neymar", "Hazard", "Pogba"]
club = ["Barcelona", "Real Madrid", "Barcelona", "Chelsea", "Juventus"]

# Wichtig:
# zip nimmt das erste Objekt der ersten Liste und das erste Objekt der zweiten Liste
# The zip() function returns a zip object, which is an iterator of tuples
dict_using_comp = {key: value for (key,value) in zip(player,club)}
print(dict_using_comp) #Type: Dictionary

# Shorter Alternative:
dict_using_comp = zip(player, club)
print(type(dict_using_comp)) #Type: Zip
print(dict(dict_using_comp))


#Loop durch Dict mit 3 Werten:
person = {"David": 12, "John": 8, "Jill": 7}

# Bis zu 3 Werten schreiben!
# Alternative using enumerate (i = index)
for i, (name, value) in enumerate(person.items()):
    print(f'Index: {i}, Name: {name}, Age: {value}')

# This program demonstrates various set operations.
baseball = set(['Jodi', 'Carmen', 'Aida', 'Alicia'])
basketball = set(['Eva', 'Carmen', 'Alicia', 'Sarah'])

# Display members of the baseball set.
print('The following students are on the baseball team:')
for name in baseball:
    print(name)

# Display members of the basketball set.
print()
print('The following students are on the basketball team:')
for name in basketball:
    print(name)

# Demonstrate intersection -> Schnittmenge
print()
print('The following students play both baseball and basketball:')
for name in baseball.intersection(basketball):
    print(name)

# Demonstrate union -> Alle zusammen
print()
print('The following students play either baseball or basketball:')
for name in baseball.union(basketball):
    print(name)

# Demonstrate difference of baseball and basketball
print()
print('The following students play baseball, but not basketball:')
for name in baseball.difference(basketball):
    print(name)

# Demonstrate difference of basketball and baseball
print()
print('The following students play basketball, but not baseball:')
for name in basketball.difference(baseball):
    print(name)

# Demonstrate symmetric difference -> Nur in einer der Gruppen vorhanden
print()
print('The following students play one sport, but not both:')
for name in baseball.symmetric_difference(basketball):
    print(name)




