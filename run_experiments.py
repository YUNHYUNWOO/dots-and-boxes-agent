import os
import sys

from simulate import run_config


def main(argv):
    configs_path:str = argv[1]

    if configs_path.endswith('.json'):
        configs = [configs_path]
    else :
        configs = os.listdir(configs_path)
    print(configs)
    
    for config in configs:
        if not config.endswith('.json'):
            continue
        config_path = os.path.join(configs_path, config)
        run_config(config_path=config_path)

if __name__ == '__main__':
    main(sys.argv)