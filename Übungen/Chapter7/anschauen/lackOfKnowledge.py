#Tupel

tup3 = "a", "b", "c", "d"  # WICHTIG  #Type ist Tupel

# Tuple containing a single value
tup5 = (50,)

# Updating Tuples
tup6 = (12, 34.56)
tup7 = ('abc', 'xyz')
# Following action is not valid for tuples because they are not mutable
# tup6[0] = 100;
# Using a list (which is mutable)
tup6 = list(tup6)
tup6[0] = 100
print(tuple(tup6))

# Deleting Tuple Elements
tup9 = ('C#', 'Python', 'Java', 'JavaScript')
# tuple = class for generating a tuple object
tup10 = tuple(item for item in tup9 if item != 'Java')
print(tup10)