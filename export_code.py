#!/usr/bin/env python3

import argparse
import os
import shutil
from enum import Enum
from pathlib import Path


class Language(Enum):
    en = 'en'
    es = 'es'


start_marker = '### YOUR IMPLEMENTATION START  ###'
end_marker = '### YOUR IMPLEMENTATION END  ###'
default_marker = '# default: '

supported_languages = (Language.en, Language.es)
marker_delimiter = '###'
start_marker_lang = {Language.en: 'YOUR IMPLEMENTATION START', Language.es: 'COMIENZO DE TU IMPLEMENTACION'}
end_marker_lang = {Language.en: 'YOUR IMPLEMENTATION END', Language.es: 'FIN DE TU IMPLEMENTACION'}


def remove_implementation(filepath: Path, lang: Language) -> int:

    def line_indent(s: str) -> str:
        return ' ' * (len(s) - len(s.lstrip()))

    def is_start(s: str):
        return s.strip() == start_marker

    def is_comment(s: str):
        return s.lstrip().startswith("#")

    def is_default_line(s: str):
        return s.lstrip().startswith(default_marker)

    def convert_default_line(s: str):
        # just in case the line has the default marker in other place, only replace first occurrence
        return line_indent(s) + s.lstrip().replace(default_marker, '', 1)

    def is_comment_lang(s: str) -> bool:
        return get_comment_lang(s) is not None

    def get_comment_lang(s: str) -> Language | None:
        for language in supported_languages:
            if s.lstrip().startswith(f'# {language.value}:'):
                return language
        return None

    def convert_locale_comment(s: str, lang: Language) -> str:
        # just in case the line has the lang prefix in other place, only replace first occurrence
        return line_indent(s) + s.lstrip().replace(f'# {lang.value}:', '#', 1)

    def is_end(s: str):
        return s.strip() == end_marker

    modifications = 0
    with open(filepath, 'r+') as f:
        is_inside_implementation = False
        new_lines = []

        for line in f.readlines():
            if (is_comment_lang(line) and get_comment_lang(line) != lang) \
                    or (is_inside_implementation and not is_comment(line)):
                continue

            if is_start(line):
                is_inside_implementation = True
                modifications += 1
                new_lines.append(line.replace(start_marker, f'{marker_delimiter} {start_marker_lang[lang]} {marker_delimiter}'))
            elif is_end(line):
                is_inside_implementation = False
                new_lines.append(f'{line_indent(line)}pass\n')
                new_lines.append(line.replace(end_marker, f'{marker_delimiter} {end_marker_lang[lang]} {marker_delimiter}'))
            elif is_default_line(line):
                new_lines.append(convert_default_line(line))
            elif is_comment_lang(line) and get_comment_lang(line) == lang:
                new_lines.append(convert_locale_comment(line, lang))
            else:
                new_lines.append(line)

        f.seek(0)
        f.truncate(0)
        f.writelines(new_lines)

    return modifications


def generate(output_path: Path, lang: Language, keep: bool):
    final_path = output_path / lang.value
    lib_path = final_path / lib_name

    print(f'Generating unimplemented library version to {final_path}')

    if not final_path.exists():
        final_path.mkdir()
    elif not keep:
        print(f'Deleting folder {final_path.absolute()}...')
        shutil.rmtree(final_path)
        final_path.mkdir()
    elif lib_path.exists():
        print(f'Deleting folder {lib_path.absolute()}...')
        shutil.rmtree(lib_path)

    print(f'Copying new version from {lib_name} to {final_path.absolute()}...')
    shutil.copytree(lib_name, lib_path)

    print(f"Removing implementation code...")
    total_files = 0
    modified_files = 0
    total_modifications = 0
    for root, dirs, files in os.walk(final_path):
        for file in files:
            if file.endswith(".py"):
                filepath = Path(os.path.join(root, file))
                modifications = remove_implementation(filepath, lang)
                total_files += 1
                total_modifications += modifications
                if modifications > 0:
                    modified_files += 1

    print(f'Done, {total_modifications} modifications in {modified_files} files out of {total_files} python files.')

    extra_files = ["requirements.txt"]
    print("Copying additional files...")
    for f in extra_files:
        print(f)
        shutil.copy(f, final_path / f)
    print(f"Done, {len(extra_files)} additional files copied")


output_dir = Path("generated")
lib_name = 'edunn'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    all_langs = [l.value for l in supported_languages]
    parser.add_argument('-l', '--languages',
                        nargs='+',
                        choices=all_langs,
                        default=all_langs,
                        help='Target language of the generated files')

    parser.add_argument('-o', '--output',
                        help=f'Output directory. Default is "{output_dir}"')

    parser.add_argument('-k', '--keep',
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help=f'Keep other files in target directory. Default is False')

    args = parser.parse_args()

    languages = [Language[l] for l in args.languages]
    print(f'Generating code for languages: {", ".join([x.value for x in languages])}')

    if args.output:
        output_dir = Path(args.output)
    print(f'Output directory: "{output_dir}"')

    for lang in languages:
        print(lang)
        generate(output_path=output_dir, lang=lang, keep=args.keep)
        print()
