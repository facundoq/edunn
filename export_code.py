#!/usr/bin/env python3
import os,argparse
from pathlib import Path
import shutil

def remove_implementation(filepath:Path):

    def is_start(s:str):
        start = "### YOUR IMPLEMENTATION START  ###"
        s=s.strip()
        return s == start

    def comment(s:str):
        s=s.strip()
        return s.startswith("#")
    def is_end(s:str):
        end = "### YOUR IMPLEMENTATION END  ###"
        s=s.strip()
        return s == end

    with open(filepath,'r+') as f:
        lines = f.readlines()
        i=0
        modifications=0
        n =len(lines)
        new_lines=[]
        while i<n:
            # read lines until start of implementation
            while (i<n) and not is_start(lines[i]):
                new_lines.append(lines[i])
                i+=1
            # add start of implementation comment
            if i<n:
                new_lines.append(lines[i])
                pass_str = lines[i][:lines[i].index("#")]+"pass\n"
                new_lines.append(pass_str)
                modifications+=1
                i+=1
            # read lines until end of implementation
            while i<n and not is_end(lines[i]):
                if comment(lines[i]):
                    new_lines.append(lines[i])
                i+=1
            # add end of implementation comment
            if i<n:
                new_lines.append(lines[i])
                i+=1
        f.seek(0)
        f.truncate(0)
        f.writelines(new_lines)

    return modifications

generated_path = Path("generated")
lib_name = "simplenn"
lib_folderpath = generated_path / lib_name
import sys

if __name__ == '__main__':
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument(dest="language", help="This is the first argument")
    # args=parser.parse_args()
    # language = parser.language

    print(f"Generating unimplemented library version to {generated_path}")

    if generated_path.exists():
        print(f"Deleting folder {generated_path.absolute()}...")
        shutil.rmtree(generated_path)
    generated_path.mkdir()

    print(f"Copying new version from {lib_name} to {generated_path.absolute()}...")
    shutil.copytree(lib_name, lib_folderpath)

    print(f"Removing implementation code...")
    total_files = 0
    modified_files = 0
    for root, dirs, files in os.walk(generated_path):
        for file in files:
            if file.endswith(".py"):
                filepath = Path(os.path.join(root, file))
                modifications = remove_implementation(filepath)
                total_files+=1
                if modifications>0:
                    modified_files+=1

    print(f"Done, {modified_files} files modified out of {total_files} python files.")

    extra_files = ["requirements.txt"]
    print("Copying additional files...")
    for f in extra_files:
        print(f)
        shutil.copy(f,generated_path/f)
    print(f"Done, {len(extra_files)} files copied")