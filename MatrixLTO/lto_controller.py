import numpy as np

class MatrixLTOTape:
    def __init__(self, width=1024, height=1024):
        self.matrix = np.zeros((width, height), dtype=int)
        self.position = (0, 0)
        self.loaded = False

    def load(self):
        self.loaded = True
        print("Tape loaded")

    def unload(self):
        self.loaded = False
        print("Tape unloaded")

    def write(self, value):
        if not self.loaded:
            print("Error: Tape not loaded")
            return
        x, y = self.position
        self.matrix[x][y] = value
        print(f"Wrote {value} at ({x}, {y})")

    def read(self):
        if not self.loaded:
            print("Error: Tape not loaded")
            return
        x, y = self.position
        value = self.matrix[x][y]
        print(f"Read {value} at ({x}, {y})")
        return value

    def seek(self, x, y):
        if not self.loaded:
            print("Error: Tape not loaded")
            return
        self.position = (x, y)
        print(f"Position set to ({x}, {y})")

    def rewind(self):
        self.position = (0, 0)
        print("Tape rewound to beginning")

# Test sequence
if __name__ == "__main__":
    tape = MatrixLTOTape()
    tape.load()
    tape.write(42)
    tape.read()
    tape.seek(0, 1)
    tape.write(99)
    tape.read()
    tape.rewind()
    tape.read()
    tape.unload()