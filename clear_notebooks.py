#!/usr/bin/env python3

import argparse
import subprocess
from pathlib import Path
from export_code import Language, supported_languages


def clear_notebooks(folderpath: Path):
    for f in folderpath.rglob("*.ipynb"):
        if not f.is_file():
            continue
        command = (
            f"jupyter nbconvert --clear-output "
            f"--ClearOutputPreprocessor.remove_metadata_fields='[(\"ExecuteTime\")]' "
            f'--inplace "{f.absolute()}"'
        )
        subprocess.run(command, shell=True)


def format_notebooks(folderpath: Path):
    for f in folderpath.rglob("*.ipynb"):
        if not f.is_file():
            continue
        command = f'black --line-length 999 "{f.absolute()}"'
        subprocess.run(command, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    all_langs = [l.value for l in supported_languages]
    parser.add_argument(
        "-l", "--languages", nargs="*", choices=all_langs, default=all_langs, help="Language of guide to export"
    )
    args = parser.parse_args()

    languages = [Language[l] for l in args.languages]
    print(f'Clearing notebooks for languages: {", ".join([x.value for x in languages])}')

    guides_folderpath = Path("guides")

    for language in languages:
        print(language)
        guide_folderpath = guides_folderpath / language.value
        clear_notebooks(guide_folderpath)
        format_notebooks(guide_folderpath)
        print()
