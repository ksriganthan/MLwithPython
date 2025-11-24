import json

#Umwandlung DICT -> JSON (mit json.dumps())

#Dictionary definieren
data = {
    "name": "Kapischan",
    "age": 25,
    "city": "Allschwil BL",
    "hobbies": ["programming", "studying", "hitting gym"]
}

#Python-Dictionary in einen JSON-String umwandeln
pretty_json = json.dumps(data, indent=4) #4 Ebenen in JSON

print(type(pretty_json)) #Class str
print(pretty_json) #JSON-Format

print(data["hobbies"][0]) #programming



##########################################################################
#Umwandlung JSON -> DICT mit (json.load()/json.loads(file)


#JSON-String definieren
employee_string = ('{'
                   '"first_name": "Kapischan",'
                   '"last_name": "Sriganthan",'
                   '"department": "Software Engineering Leadership"'
                   '}')

print(type(employee_string)) #Class str

#JSON-String in Python-Dictionary umwandeln (MIT ERROR HANDLING !!!)
#json.loads() -> JSON-String -> Dictionary
#json.load(file) -> JSON-Datei -> Dictionary

try:
    json_dict = json.loads(employee_string)
    print(type(json_dict)) #class dict
    print(json_dict["first_name"]) #Kapischan
except json.JSONDecodeError as e:
    print(f"Invalid JSON in employee_string: {e}")




########################################################################
#Umwandlung JSON -> DICT mit (json.load()/json.loads(file)
#JSON String mit mehreren Objekten

employees_string = '''
{
    "employees" : [
       {
           "first_name": "Michael", 
           "last_name": "Rodgers", 
           "department": "Marketing"
        },
       {
           "first_name": "Michelle", 
           "last_name": "Williams", 
           "department": "Engineering"
        }
    ]
}
'''

try:
    data = json.loads(employees_string)
    print(type(data))
    for employee in data["employees"]:
        print(employee["last_name"])
except json.JSONDecodeError as e:
    print(f"Invalid JSON in employees_string: {e}")
