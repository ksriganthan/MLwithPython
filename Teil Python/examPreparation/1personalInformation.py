my_list = ["Apfel", "Banana", "Ananas"]
my_list2 = list()
my_list.append("Apfel")
my_list[0] = "Banana"
my_tuple = (5,)
my_tuple2 = tuple()
my_dict = {} # nur Immutable-Werte (Keys): String, Tupel, Integer und nicht duplikat
my_dict2 = dict()
my_dict["1"] = "one"
my_dict2["1"] = "two"
del my_dict["1"] #löscht Key-Value Paar "1" - "one"
my_dict.get("1")
print(my_dict.get("fasf"))
dict_using_comp = {key: value for (key, value) in zip(player, club)}
print(dict_using_comp)
dict_using_comp = zip(player, club)
my_set = set()
my_set2 = {"Blueberry"}
my_set.add("Blueberry")
my_set.add("Banana")
my_set.remove("Ananas") # gibt Exception, wenn es fehlt
my_set.discard("Ananas") #gibt keine Meldung, wenn es fehlt
my_set2.issubset(my_set) #Gibt True aus, wenn alles in my_set2 auch in my_set drin ist
my_set2.issuperset(my_set) #Gibt True aus, wenn my_set2 das gleiche drin hat wie my_set


