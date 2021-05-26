#!/usr/bin/env python3
import os,argparse
from pathlib import Path
import sys
import zipfile
from export_code import generated_path
import subprocess
import shutil


def delete_checkpoints(folderpath:Path):
    for f in folderpath.rglob("*.ipynb_checkpoints"):
        if f.is_dir() and f.name==".ipynb_checkpoints":
            print(f"Deleting {f.absolute()}..")
            shutil.rmtree(f.absolute())

def clear_notebooks(folderpath:Path):
    for f in folderpath.rglob("*.ipynb"):
        if not f.is_file():
            continue
        command = f"jupyter nbconvert --clear-output --inplace '{f.absolute()}'"
        subprocess.run(command,shell=True)

def zip_all(path,zip_file):
    for f in path.iterdir():
        if f.is_file():
            zip_file.write(f, f.name)
        if f.is_dir():
            zipdir(f, zip_file)

def zipdir(path, zip_file,skip_hidden=True):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.startswith(".") and skip_hidden:
                continue
            zip_file.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), os.path.join(path, '..')))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="language", help="Language of guide to export")
    args=parser.parse_args()
    language = args.language

    guides_folderpath=Path("guides")
    releases_folderpath=Path("releases")
    guide_folderpath = guides_folderpath / language
    if not guide_folderpath.exists():
        sys.exit(f"Language {language} not found. Check `guides` folder for available languages.")
    print(f"Language {language} available.")
    print(f"Delete checkpoints in {guide_folderpath}...")
    delete_checkpoints(guide_folderpath)

    print(f"Clear notebooks in {guide_folderpath}...")
    clear_notebooks(guide_folderpath)

    zip_filepath = releases_folderpath / f"{language}.zip"

    if not generated_path.exists():
        sys.exit(f"Code skeleton not found in {generated_path.absolute()}")

    print(f"Creating zip file...")
    zip_file = zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED)
    print(f"Adding guide to zip...")
    zip_all(guide_folderpath, zip_file)
    print(f"Adding code to zip...")
    zip_all(generated_path,zip_file)

    print(f"Saving to file...")
    zip_file.close()
    print(f"Done")
    
