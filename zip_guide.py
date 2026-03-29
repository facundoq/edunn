#!/usr/bin/env python3
import os, argparse
from pathlib import Path
import sys
import zipfile
from export_code import generated_path, lib_name
import subprocess
import shutil


def delete_checkpoints(folderpath: Path):
    for f in folderpath.rglob("*.ipynb_checkpoints"):
        if f.is_dir() and f.name == ".ipynb_checkpoints":
            print(f"    Deleting {f.absolute()}..")
            shutil.rmtree(f.absolute())


def clear_notebooks(folderpath: Path):
    for f in folderpath.rglob("*.py"):
        if not f.is_file():
            continue
        command = f"uv run marimo check --fix \"{f.absolute()}\""
        subprocess.run(command, shell=True)

def convert_notebooks(folderpath: Path, build_folder: Path):
    build_folder.mkdir(parents=True, exist_ok=True)
    for f in folderpath.rglob("*.py"):
        if not f.is_file():
            continue
        rel_path = f.relative_to(folderpath)
        ipynb_out = build_folder / rel_path.with_suffix(".ipynb")
        ipynb_out.parent.mkdir(parents=True, exist_ok=True)
        command = f"uv run marimo export ipynb \"{f.absolute()}\" -o \"{ipynb_out.absolute()}\""
        subprocess.run(command, shell=True)


def zip_all(path, zip_file, exclude_ext=None):
    for f in path.iterdir():
        if f.is_file():
            if exclude_ext and f.name.endswith(exclude_ext):
                continue
            zip_file.write(f, f.name)
        if f.is_dir():
            zipdir(f, zip_file, exclude_ext=exclude_ext)


def zipdir(path, zip_file, skip_hidden=True, exclude_ext=None):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.startswith(".") and skip_hidden:
                continue
            if exclude_ext and file.endswith(exclude_ext):
                continue
            zip_file.write(
                os.path.join(root, file), os.path.relpath(os.path.join(root, file), os.path.join(path, ".."))
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="language", help="Language of guide to export")
    args = parser.parse_args()
    language = args.language
    print(
        f"""
    ********************************************
    * This script will compile and zip a guide *
    * Only run this command from the root of   *
    * the edunn library                        *
    ********************************************
    """
    )

    guides_folderpath = Path("guides")
    releases_folderpath = Path("releases")
    guide_folderpath = guides_folderpath / language
    if not guide_folderpath.exists():
        sys.exit(f"Language {language} not found. Check `guides` folder for available languages.")
    print(f"Language *{language}* available.")
    print(f"Deleting checkpoints in {guide_folderpath}...")
    delete_checkpoints(guide_folderpath)

    print(f"Clearing notebooks in {guide_folderpath}...")
    clear_notebooks(guide_folderpath)

    build_base = Path("_build")
    build_folder = build_base / "guides"
    print(f"Converting marimo notebooks to Jupyter notebooks in {build_folder}...")
    convert_notebooks(guide_folderpath, build_folder)

    print(f"Arranging edunn for solving (generating skeleton)...")
    subprocess.run([sys.executable, "export_code.py"], check=True)

    zip_filepath = releases_folderpath / f"{lib_name}-{language}.zip"

    print(f"Creating zip file...")
    zip_file = zipfile.ZipFile(zip_filepath, "w", zipfile.ZIP_DEFLATED)
    print(f"Adding guide to zip without .py files...")
    zip_all(guide_folderpath, zip_file, exclude_ext=".py")
    print(f"Adding compiled Jupyter notebooks to zip...")
    zip_all(build_folder, zip_file)
    print(f"Adding code to zip...")
    zip_all(generated_path, zip_file)

    print(f"Saving to file...")
    zip_file.close()

    print(f"Cleaning up build folder...")
    shutil.rmtree(build_base)

    print(f"Done: {zip_filepath}")
