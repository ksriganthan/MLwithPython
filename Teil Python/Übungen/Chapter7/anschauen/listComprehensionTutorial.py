
# List Comprehension

# [Ausdruck for Element in Iterable if Bedingung]

# Ausdruck -> was ich in die Liste einfügen möchte
# Element -> Variable, die nacheinander die Werte aus der vorhandenen Liste/Range bekommt
# Iterable -> die "Quelle" (z.B. Liste, Range)
# if Bedingung (optional) -> filtert, welche Elemente übernommen werden

# 1. Einfache Kopie:
zahlen = [1,2,3,4]
quadrateVonZahlen = [x**2 for x in zahlen]
print(quadrateVonZahlen) # [1,4,9,16]


# 2. Mit Bedingung: 
zahlen = [1,2,3,4,5,6]
gerade = [x for x in zahlen if x % 2 == 0]
print(gerade) #[2,4,6]

# 3. Mit Transformation + Bedingung:
woerter = ["Apfel", "Banane", "Kiwi", "Pfirsich"]

kurze = [w.upper() for w in woerter if len(w) <= 5]
print(kurze)