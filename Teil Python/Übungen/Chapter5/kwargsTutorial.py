

#Nach Stern MUSS der Rest zum Parameter zugewiesen werden
#Default kann auch gegeben werden
#Vor dem Stern kann man entweder position-only oder keyword-only
def add (a,*,b=10,c):
  print(a+b+c)

add(a=1,c=3)



#Alles vor / MUSS positionsweise übergeben werden und nicht keyword-only
#Man könnte nach dem Strich - c und d - auch positions-only machen
def sub(a,b,/,c,d):
  print(a - b - c - d)

sub(10,5,6,d=7)


  
  

