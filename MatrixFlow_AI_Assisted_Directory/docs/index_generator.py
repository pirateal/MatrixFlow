import os
import json

def generate_index(base_path):
    index = {}
    for root, dirs, files in os.walk(base_path):
        for f in files:
            path = os.path.join(root, f)
            rel = os.path.relpath(path, base_path)
            index[rel] = {
                "size": os.path.getsize(path),
                "last_modified": os.path.getmtime(path)
            }
    with open(os.path.join(base_path, 'index.json'), 'w') as out:
        json.dump(index, out, indent=2)

if __name__ == "__main__":
    generate_index(os.path.dirname(__file__))
