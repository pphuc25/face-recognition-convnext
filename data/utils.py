import pickle

def load_database(data_path):
    with open(data_path, 'rb') as file:
        serialized_data = file.read()
    database = pickle.loads(serialized_data)
    return database

def save_file(serialized_data, args):
    with open(args.save_file_path, 'wb') as file:
        file.write(serialized_data)

def choice_availabel_options():
    print("\n")
    print("="*40)
    options = """
1: View names
2: Change names
3: Delete data
4: Break
"""
    print(options)
    choice = int(input("Type the choice you want: "))
    return choice