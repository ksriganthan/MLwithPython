import json

# Define data as a dict
data = {
    "name": "John",
    "age": 30,
    "city": "New York",
    "hobbies": ["reading", "traveling", "cooking"]
}

# Pretty print JSON data as string with dumps
pretty_json = json.dumps(data, indent=4) # 4 Merkmale also Ebenen (Key-Value-Paare)

# Print pretty JSON
print(pretty_json)
print(type(pretty_json)) #Class str

# Printing first hobby
print("First Hobby in List: " + data["hobbies"][0])

# JSON string data
employee_string = '{"first_name": "Michael", "last_name": "Rodgers", "department": "Marketing"}'

# Check data type with type() method
print(type(employee_string))

# Convert string to dict object with loads (with error handling)
try:
    json_object = json.loads(employee_string) #mit json.load(file) file lesen
    # Check new data type
    print(type(json_object))
    # Access first_name in dictionary
    print(json_object["first_name"])
except json.JSONDecodeError as e:
    print(f"Invalid JSON in employee_string: {e}")

# JSON string with multiple employees
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

# Convert string to object (with error handling)
try:
    data = json.loads(employees_string)
    print(type(data))
    # Access last_name values
    for employee in data["employees"]:
        print(employee["last_name"])
except json.JSONDecodeError as e:
    print(f"Invalid JSON in employees_string: {e}")
