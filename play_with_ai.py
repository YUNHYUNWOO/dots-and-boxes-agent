import argparse

from simulate import play_against_policy

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', dest='path', required=True, help="Path to a JSON file containing policy config")
parser.add_argument('--agent_first', dest='agent_first', action='store_true', help="when --agnet_first is activated Agent plays first (False).")

def main():
    global parser
    args = parser.parse_args()

    play_against_policy(args.path, (not args.agent_first))
    
if __name__ == '__main__':
    main()