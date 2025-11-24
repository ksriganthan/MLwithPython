
set1 = {"Kapi", "Susi", "Karim"}
set2 = {"Kapi","Susi","Loico"}

print(set1.intersection(set2)) #Kapi, Susi

print(set1.union(set2)) # Kapi, Susi, Karim, Loico

print(set1.difference(set2)) # Karim

print(set2.difference(set1)) # Loico

print(set1.symmetric_difference(set2)) # Loico, Karim

