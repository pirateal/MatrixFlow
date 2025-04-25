# Memory Manager for MatrixFlow
class MemoryManager:
    def __init__(self):
        self.storage = {}

    def allocate(self, key, matrix):
        self.storage[key] = matrix

    def retrieve(self, key):
        return self.storage.get(key, None)

if __name__ == "__main__":
    from cupy import array
    mm = MemoryManager()
    mm.allocate('test', array([1,2,3]))
    print(mm.retrieve('test'))
