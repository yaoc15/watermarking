from hamming import encode,decode
from bitarray import bitarray
data = bitarray('1111')
data_with_parity = encode(data)
print(data_with_parity)
data_with_parity[3] = not data_with_parity
data_with_parity[4] = not data_with_parity
print(decode(data_with_parity))
