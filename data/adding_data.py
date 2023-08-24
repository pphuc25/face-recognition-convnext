import sys
import os
import argparse
import pickle
from utils import load_database, save_file

current_dir = os.getcwd()
target_dir = os.path.abspath(os.path.join(current_dir, '..'))
dir_code = 'src'
code_path = os.path.join(target_dir)
sys.path.append(code_path)  # Add the directory to the module search path

from src import convfacenet


def config():
    parser = argparse.ArgumentParser(description='...')
    parser.add_argument('--image_path', type=str, default='/home/pphuc/Downloads/testing-code/Multi-Face-Recognize/Multi-Face-Recognize/data/images/minh_khoi.jpeg',
                                    help='path of new person image')
    parser.add_argument('--save_file_path', type=str, default='database.pickle',
                                    help='')
    parser.add_argument('--name_person', type=str, default="Minh Khoi",
                                    help='name of new person')
    args = parser.parse_args()
    return args


def main():
    args = config()

    data_path = os.path.join(current_dir, args.save_file_path)
    database = load_database(data_path)

    person_image=convfacenet.load_image(args.image_path)
    new_person_embedding = convfacenet.faces_features(person_image)

    new_data = {
        'name': args.name_person,
        'face_feature': new_person_embedding[1][0]
    }
    database.append(new_data)

    serialized_data = pickle.dumps(database)
    save_file(serialized_data, args)



if __name__ == "__main__":
    main()