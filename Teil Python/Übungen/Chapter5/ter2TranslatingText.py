from deep_translator import GoogleTranslator

def translation(string):
    return GoogleTranslator(source="auto", target="en").translate(string)

text = input("Please enter an arbitrary text for translation (stop to terminate): ")
while text.lower() != "stop":
     translated_text = translation(text)
     print(translated_text)
     text = input("Please enter text: ")
print("Programm terminated ...")

#Wenn die Eingabe leer ist oder unverständlich, wird intern eine Exception geworfen und es
#kommt ein leerer String zurück




