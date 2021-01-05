import os,argparse
from pathlib import Path
import shutil

def remove_implementation(filepath:Path):
    print(filepath)

if __name__ == '__main__':
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument(dest="language", help="This is the first argument")
    # args=parser.parse_args()
    # language = parser.language
    generated_path = Path("generated")
    folders_to_export = ["datasets","simplenn"]
    for folder in folders_to_export:
        shutil.copy(folder, generated_path/folder)


    for root, dirs, files in os.walk(generated_path):
        for file in files:
            if file.endswith(".py"):
                filepath = Path(os.path.join(root, file))
                remove_implementation(filepath)

