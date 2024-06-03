import json
from pathlib import Path
from shutil import rmtree


def main():
    with open('./project_config.json', 'r') as config_file:
        config = json.load(config_file)

    path_folders = config['data']
    for key in path_folders:
        for path in Path(path_folders[key]).glob('**/*'):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                rmtree(path)

        print(f'{path_folders[key]} is cleaned.')


if __name__ == '__main__':
    main()
