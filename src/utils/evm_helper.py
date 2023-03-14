from Crypto.Hash import keccak
from typing import List


def keccak_of_func_signature(func_name: str, param_types: List[str]):
    # Create a new Keccak-256 hash object
    h = keccak.new(digest_bits=256)
    input_str = func_name + "(" + ",".join(param_types) + ")"
    h.update(str.encode(input_str))
    hex_hash = h.hexdigest()
    return hex_hash[0:8].encode('utf-8')

def pad_argument(arg: any):
    if arg.startswith('0x'):
        arg = arg[2:]
    encoded = arg.encode('utf-8')
    length = len(encoded)
    padded_encoded = b'0' * (64 - length) + encoded \
        if length < 64 else encoded
    return padded_encoded