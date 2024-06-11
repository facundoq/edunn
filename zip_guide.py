#!/usr/bin/env python3

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from export_code import output_dir, lib_name, Language, supported_languages


def delete_checkpoints(folderpath: Path):
    for f in folderpath.rglob("*.ipynb_checkpoints"):
        if f.is_dir() and f.name == ".ipynb_checkpoints":
            print(f"    Deleting {f.absolute()}..")
            shutil.rmtree(f.absolute())


def clear_notebooks(folderpath: Path):
    for f in folderpath.rglob("*.ipynb"):
        if not f.is_file():
            continue
        command = (f"jupyter nbconvert --clear-output "
                   f"--ClearOutputPreprocessor.remove_metadata_fields='[(\"ExecuteTime\")]' "
                   f"--inplace '{f.absolute()}'")
        subprocess.run(command, shell=True)


def zip_all(path, zip_file):
    for f in path.iterdir():
        if f.is_file():
            zip_file.write(f, f.name)
        if f.is_dir():
            zipdir(f, zip_file)


def zipdir(path, zip_file, skip_hidden=True):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.startswith(".") and skip_hidden:
                continue
            zip_file.write(os.path.join(root, file),
                           os.path.relpath(os.path.join(root, file), os.path.join(path, '..')))


def generate_zip(output_path: Path, lang: Language):
    language = lang.value
    final_path = output_path / language

    guides_folderpath = Path("guides")
    releases_folderpath = Path("releases")
    guide_folderpath = guides_folderpath / language
    if not guide_folderpath.exists():
        print(f"Language {language} not found. Check `guides` folder for available languages.", file=sys.stderr)
        return
    print(f"Language *{language}* available.")
    print(f"Deleting checkpoints in {guide_folderpath}...")
    delete_checkpoints(guide_folderpath)

    print(f"Clearing notebooks in {guide_folderpath}...")
    clear_notebooks(guide_folderpath)

    zip_filepath = releases_folderpath / f"{lib_name}-{language}.zip"

    if not final_path.exists():
        sys.exit(f"Code skeleton not found in {final_path.absolute()}")

    print(f"Creating zip file...")
    zip_file = zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED)
    print(f"Adding guide to zip...")
    zip_all(guide_folderpath, zip_file)
    print(f"Adding code to zip...")
    zip_all(final_path, zip_file)

    print(f"Saving to file...")
    zip_file.close()
    print(f"Done: {zip_filepath}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    all_langs = [l.value for l in supported_languages]
    parser.add_argument("-l", "--languages",
                        nargs='*',
                        choices=all_langs,
                        default=all_langs,
                        help="Language of guide to export")
    args = parser.parse_args()

    print(f"""
    ********************************************
    * This script will compile and zip a guide *
    * Only run this command from the root of   *
    * the edunn library                        * 
    ********************************************
    """)

    languages = [Language[l] for l in args.languages]
    print(f'Generating zip for languages: {", ".join([x.value for x in languages])}')

    for language in languages:
        print(language)
        generate_zip(output_path=output_dir, lang=language)
        print()
