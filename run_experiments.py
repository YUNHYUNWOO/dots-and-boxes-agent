import os
import argparse

from simulate import run_config

parser = argparse.ArgumentParser()

parser.add_argument('-p', '--path', dest='path', required=True, help="Path to a JSON file, or a directory containing multiple JSON experiment configs.")
parser.add_argument('--human', dest='human',action='store_true', help="When --human is activated, rendering gaming window")

def main():
    global parser
    args = parser.parse_args()
    path:str = args.path

    config_paths = []
    if path.endswith('.json'):
        # When Single JSON file
        config_paths.append(path)
    else :
        # When Directory
        for config_path in os.listdir(path):
            if not config_path.endswith('.json'):
                continue
            config_paths.append(os.path.join(path, config_path))
        
    print(config_paths)
    for config_path in config_paths:
        run_config(config_path=config_path, render_mode = 'human' if args.human else 'rgb_array')

if __name__ == '__main__':
    main()