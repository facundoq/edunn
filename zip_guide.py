import os,argparse
from pathlib import Path
import sys
import zipfile

def zipdir(path, zip_file,skip_hidden=True):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.startswith(".") and skip_hidden:
                continue
            zip_file.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), os.path.join(path, '..')))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="language", help="This is the first argument")
    args=parser.parse_args()
    language = args.language

    guides_folderpath=Path("guides")

    guide_folderpath = guides_folderpath / language
    if not guide_folderpath.exists():
        sys.exit(f"Language {language} not found. Check `guides` folder for available languages.")

    zip_filepath = guides_folderpath / f"{language}.zip"

    zip_file = zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED)
    zipdir(guide_folderpath, zip_file)
    zip_file.close()
    
