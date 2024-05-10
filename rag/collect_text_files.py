import glob
import shutil
import hashlib
import os

from argparse import ArgumentParser


def main():
    parser = ArgumentParser(description='Find text files and copy them to project')
    parser.add_argument('--path', '-p', required=True, help='Path to search for .txt files in')
    parser.add_argument('--target', '-t', required=True, help='Path to search for .txt files in')

    args = parser.parse_args()
    os.makedirs(args.target, exist_ok=True)

    for filepath in glob.glob(f"{args.path}/**/*.txt", recursive=True):
        print(f"Copy: {filepath}")

        shutil.copyfile(filepath, f'{args.target}/{hashlib.md5(filepath.encode()).hexdigest()}.txt')


if __name__ == "__main__":
    main()
