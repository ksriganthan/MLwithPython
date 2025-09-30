from random import randint
# Exercise with lists

# Zahlen generieren
# Generiert Zahlen zwischen 1 - 10 (inklusive)
# Für jede i in range bis 20 wird eine generierte Zahl hinzugefügt
my_list = [randint(1,10) for _ in range(20)]

# Liste ausgeben
print(my_list)
# Summe
print("Summe: ", sum(my_list))
# Maximum
print("Max: ", max(my_list))
# Minimum
print("Min: ", min(my_list))
# Durchschnitt
print("Durchschnitt: ", sum(my_list) / len(my_list))

#Für die Prüfung: Nur List Comprehension verstehen können und nicht schreiben!
#Jede List Comprehension kann von while/for - Schleife ersetzt werden!


#Example Chapter 7 tuples.py anschauen