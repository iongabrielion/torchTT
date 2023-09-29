import torchtt as tntt
import torch as tn
import datetime

def create_banded(a, m, A, band=1):
    A_band = tn.zeros(a, m, m, A, dtype=tn.float64)
    for k in range(-band, band+1):
        diag = tn.diagonal(A_band, k, 1, 2)
        diag += 100 * tn.rand(a, A, diag.shape[-1])
    return A_band

A = tntt.random([(4, 4), (5, 5), (6, 6), (3, 3)], [1, 2, 3, 2, 1])
B = tntt.random([(4, 4), (5, 5), (6, 6), (3, 3)], [1, 3, 2, 2, 1])

Cr = A @ B

C = tntt.amen_mm(A, B, kickrank=8, verbose=False)


print((C-Cr).norm()/Cr.norm())

A = tntt.random([(4, 4), (5, 5), (6, 6), (3, 3)], [1, 2, 3, 2, 1])
B = tntt.random([(4, 3), (5, 2), (6, 5), (3, 6)], [1, 3, 2, 2, 1])

Cr = A @ B

C = tntt.amen_mm(A, B, kickrank=8, verbose=False)


print((C-Cr).norm()/Cr.norm())

def create_banded(a, m, A, band=1):
    A_band = tn.zeros(a, m, m, A, dtype=tn.float64)
    for k in range(-band, band+1):
        diag = tn.diagonal(A_band, k, 1, 2)
        diag += 100 * tn.rand(a, A, diag.shape[-1])
    return A_band
    
bands_A = [2, 1, 0]
bands_B = [3, 2, 0]
A = tntt.random([(128, 128), (32, 32), (16, 16)], [1, 4, 20, 1])
B = tntt.random([(128, 128), (32, 32), (16, 16)], [1, 20, 5, 1])
for i in range(len(A.cores)):
    A_core = A.cores[i]
    B_core = B.cores[i]
    A.cores[i] = create_banded(A_core.shape[0], A_core.shape[1], A_core.shape[-1], bands_A[i])
    B.cores[i] = create_banded(B_core.shape[0], B_core.shape[1], B_core.shape[-1], bands_B[i])
Cr = 25 * A @ B

A = A + A + A + A + A
B = B + B + B + B + B

print("Not banded")
time = datetime.datetime.now()
C = tntt.amen_mm(A, B)
time = datetime.datetime.now() - time
assert ((C-Cr).norm()/Cr.norm()) < 1e-11
print('Multiplication time: ',time)

print("Band A")
time = datetime.datetime.now()
C = tntt.amen_mm(A, B, bandsA=bands_A)
time = datetime.datetime.now() - time
assert ((C-Cr).norm()/Cr.norm()) < 1e-11
print('Multiplication time: ',time)

print("Band B")
time = datetime.datetime.now()
C = tntt.amen_mm(A, B, bandsB=bands_B)
time = datetime.datetime.now() - time
assert ((C-Cr).norm()/Cr.norm()) < 1e-11
print('Multiplication time: ',time)

print("Band A, Band B")
time = datetime.datetime.now()
C = tntt.amen_mm(A, B, bandsA=bands_A, bandsB=bands_B)
time = datetime.datetime.now() - time
print('Multiplication time: ',time)
assert ((C-Cr).norm()/Cr.norm()) < 1e-11