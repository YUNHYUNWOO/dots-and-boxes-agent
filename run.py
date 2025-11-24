import os
import sys
import subprocess

from simulate import run_config


def main():
    configs_dir = './config/configs'
    configs = os.listdir(configs_dir)
    print(configs)
    
    for config in configs:
        config_path = os.path.join(configs_dir, config)
        run_config(config_path=config_path)

if __name__ == '__main__':
    main()