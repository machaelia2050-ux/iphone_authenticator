import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

real_path = os.path.join(BASE_DIR, "data", "raw", "real")
fake_path = os.path.join(BASE_DIR, "data", "raw", "fake")

print("Real images:", len(os.listdir(real_path)))
print("Fake images:", len(os.listdir(fake_path)))

import os, hashlib
from pathlib import Path

folder = Path("data/raw/fake")
seen   = {}
dupes  = 0

for f in sorted(folder.iterdir()):
    if f.is_file():
        h = hashlib.md5(f.read_bytes()).hexdigest()
        if h in seen:
            print(f"DUPLICATE: {f.name}  ==  {seen[h]}")
            dupes += 1
        else:
            seen[h] = f.name

print(f"\nUnique images : {len(seen)}")
print(f"Duplicates    : {dupes}")
print(f"Total files   : {len(seen) + dupes}")