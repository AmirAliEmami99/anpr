import sys
import json

if __name__ == "__main__":
    string = json.loads(sys.argv[1])
    print(string[3],type(string))
