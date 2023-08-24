import argparse
import pickle
from typing import List
from utils import load_database, save_file, choice_availabel_options

def config():
    parser = argparse.ArgumentParser(description='...')
    parser.add_argument('--data_path', type=str, default='/home/pphuc/Downloads/testing-code/Multi-Face-Recognize/Multi-Face-Recognize/data/database.pickle',
                                    help='')
    parser.add_argument('--save_file_path', type=str, default='/home/pphuc/Downloads/testing-code/Multi-Face-Recognize/Multi-Face-Recognize/data/database.pickle',
                                    help='')
    args = parser.parse_args()
    return args

def view_names(database: List[dict]) -> list:
    names = [person['name'] for person in database]
    return names

def change_name(database: List[dict]) -> List[dict]:
    names = view_names(database)
    name_want_change = input("What's name you want to change: ")
    new_name = input("New name:")
    index = names.index(name_want_change)
    database[index]['name'] = new_name
    return database

def delete_data(database: List[dict]) -> List[dict]:
    names = view_names(database)
    name_want_delete = input("What's name you want to delete: ")
    index = names.index(name_want_delete)
    database.pop(index)
    return database

if __name__ == "__main__":
    args = config()
    database = load_database(args.data_path)
    manipulated = False
    while True:
        choice = choice_availabel_options()
        if choice == 1:
            names = view_names(database)
            print(names)
        if choice == 2:
            database = change_name(database)
            manipulated = True
        if choice == 3:
            database = delete_data(database)
            manipulated = True
        if choice == 4:
            break
    if manipulated:
        serialized_data = pickle.dumps(database)
        save_file(serialized_data, args)