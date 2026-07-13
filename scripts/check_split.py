from pathlib import Path

train = len(list(Path("data/train").glob("*/*.png")))
val = len(list(Path("data/val").glob("*/*.png")))
test = len(list(Path("data/test").glob("*/*.png")))

print("Training:", train)
print("Validation:", val)
print("Testing:", test)
print("Total:", train + val + test)