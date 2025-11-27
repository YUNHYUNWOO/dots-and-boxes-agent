import sys

from simulate import play_against_policy

def main(argv):
    policy_config_path, user_first = argv[1:]
    play_against_policy(policy_config_path, user_first)

if __name__ == '__main__':
    argv = sys.argv
    main(argv)