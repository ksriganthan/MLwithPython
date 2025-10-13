#Tupels könnte man z.B. für GPS-Daten verwendet werden!
#Denn diese sollten nicht verändert werden

#Tupel kann auch für RGB-Daten verwendet werden

#Tupel und String sind unveränderbar -> man könnte sie hashen und daraus
#einen eindeutigen Key erstellen

#List, Set und Dictionary sind mutable

#Bei String-Objekt wenn man einen neuen Wert zuordnet, dann wird das alte zerstört und einen neuen erstellt


#Hier kann man noch 1 Value frei nochhinzufügen
tup5 = (50,)

#Tupel ändern
tup6 = (12,34,56)
#In Liste konvertieren
tup6 = list(tup6)
#In die Liste ein neues Objekt hinzufügen
tup6[0] = 100
#Die Liste als ein neues Tuple ausgeben
print(tuple(tup6))

# Example Chapter 7 tuples.py anschauen