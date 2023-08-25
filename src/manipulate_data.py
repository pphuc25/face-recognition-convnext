from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
import pickle
from typing import List, Literal
from PIL import Image

import convfacenet


class InputData:
    def load_config() -> DictConfig:
        """
        Load config from file yaml
        """
        initialize(version_base=None, config_path="../config")
        config = compose("main.yaml")
        return config

    def load_database(config) -> List[dict]:
        """
        Function that load entire database.
        """
        with open(config.data.final, 'rb') as file: serialized_data = file.read()
        database = pickle.loads(serialized_data)
        return database

    @staticmethod
    def load_image(config) -> Image:
        """
        Recieve path of image and turn into PIL.Image
        """
        person_image = convfacenet.load_image(config.process.new_data.image_path)
        return person_image
    
    def get_user_choice() -> None:
        """
        Receive the option user want to manipulate database
        """
        turn_int_to_str = {1: "View names", 2: "Add new person", 3: "Change names", 4: "Delete data", 5: "Exit"}
        while True:
            try:
                choice = int(input("Type the choice you want: "))
                if 1 <= choice <= 5:
                    return turn_int_to_str[choice]
                else:
                    print("Invalid choice. Please select a valid option.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    
    def get_old_and_new_name() -> (str, str):
        """
        Receive the current name and new name user want to change.
        """
        name_want_change = input("What's name you want to change: ")
        new_name = input("New name:")
        return (name_want_change, new_name)
    
    def get_name_want_delete() -> str:
        """
        Receive the current user want to delete.
        """
        name_want_delete = input("What's name you want to delete: ")
        return name_want_delete
    


class ManipulateData:
    def __init__(self, config: DictConfig):
        """
        Initialize current database, get the name of persons availabel in database and
        load config.
        """
        self.config = config
        self.database: List[dict] = InputData.load_database(self.config)
        self.names = [person['name'] for person in self.database]

    def add_person(self):
        """
        Add new person to the current database.
        """
        person_image = InputData.load_image(self.config)
        new_person_embedding = self.__embedding_image(person_image)

        new_data = {
            'name': self.config.process.new_data.name,
            'face_feature': new_person_embedding[1][0]
        }

        self.database.append(new_data)
        self.names.append(self.config.process.new_data.name)

    def __embedding_image(self, person_image) -> tuple[Literal[True], list]:
        new_person_embedding = convfacenet.faces_features(person_image)
        return new_person_embedding

    def change_name(self, name_want_change: str, new_name: str):
        """
        Change the name of person in the current database.
        """
        index = get_index_of_person_name(self.names, name_want_change)
        self.database[index]['name'] = new_name; self.names[index] = new_name

    def delete_person(self, name_want_delete: str) -> List[dict]:
        """
        Delete the person in the current database.
        """
        index = get_index_of_person_name(self.names, name_want_delete)
        self.database.pop(index); self.names.pop(index)
    
    def dump_database(self) -> bytes:
        serialized_data = pickle.dumps(self.database)
        return serialized_data

def get_index_of_person_name(names_list, name) -> int:
    """
    Get the index of person name to indicate to list location.
    """
    index = names_list.index(name)
    return index
    

class OuputData:
    def display_menu() -> None:
        MENU_OPTIONS = """
        1: View names
        2: Add new person
        3: Change names
        4: Delete data
        5: Exit
        """
        print("\n" + "=" * 40)
        print(MENU_OPTIONS)

    def save_file(serialized_data: List[dict], save_data_path: str) -> None:
        with open(save_data_path, 'wb') as file: file.write(serialized_data)

def check_data_has_changed(choice):
    if choice not in [1, 5]:  # These options not affect to the current database
        return True
    return None



def main():
    # print(OmegaConf.to_yaml(config, resolve=True))
    changed_data = None
    input_class, output_class = InputData, OuputData
    config = input_class.load_config()
    data_manipulator = ManipulateData(config)

    while True:
        output_class.display_menu()
        choice = input_class.get_user_choice()
        if not changed_data:
            changed_data = check_data_has_changed(choice)
        if choice == "View names":
            print(data_manipulator.names)
        if choice == "Add new person":
            data_manipulator.add_person()
        if choice == "Change names":
            name_want_change, new_name = input_class.get_old_and_new_name()
            data_manipulator.change_name(name_want_change, new_name)
        if choice == "Delete data":
            name_want_delete = input_class.get_name_want_delete()
            data_manipulator.delete_person(name_want_delete)
        if choice == "Exit":
            break
    if changed_data:
        serialized_data = data_manipulator.dump_database()
        output_class.save_file(serialized_data, config.process.save_data_path.processed)



if __name__ == "__main__":
    main()