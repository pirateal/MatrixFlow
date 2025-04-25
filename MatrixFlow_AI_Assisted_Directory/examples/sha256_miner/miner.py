# SHA-256 Miner Template
import hashlib

def mine(prefix, difficulty):
    target = '0' * difficulty
    nonce = 0
    while True:
        text = f"{prefix}{nonce}".encode()
        h = hashlib.sha256(text).hexdigest()
        if h.startswith(target):
            return nonce, h
        nonce += 1

if __name__ == "__main__":
    nonce, h = mine('MatrixFlow', 4)
    print("Nonce:", nonce, "Hash:", h)
