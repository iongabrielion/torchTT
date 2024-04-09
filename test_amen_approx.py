import torchtt as tntt 
import torch as tn 

A = tntt.randn([(10,11), (12,13), (14,15)], [1,4,2,1])
x = tntt.randn([11, 13, 15], [1,3,5,1])
x2 = tntt.randn([11, 13, 15], [1,3,5,1])
yy = A @ x

y = tntt.amen_mv(A, x, verbose=True, use_cpp=False)

print((y-yy).norm()/yy.norm())

yy2 = A @ (x+x2)
y2 = tntt.amen_mv(A, [x, x2], verbose=False)

print((y2-yy2).norm()/yy2.norm())

xs = []
Ao = tntt.randn([(101,100), (102,100), (104,100)], [1,4,2,1])
yy3 = 0
rndd = tntt.randn([100, 100, 100], [1,3,5,1])
for i in range(9):
    rnd = rndd + i
    yy3 = yy3 + 64*Ao@rnd
    rnd = rnd + rnd 
    rnd = rnd + rnd 
    rnd = rnd + rnd 
    xs.append(rnd)
yy3 = yy3.round(0)

A = Ao+ Ao
A = A + A 
A = A + A 

y3 = tntt.amen_mv(A, xs, kickrank=8, verbose=False)

print(y3)
print(yy3)

print((y3-yy3).norm()/yy3.norm())


A = tntt.randn([(2,2)]*20, [1] + [8]*19 + [1])
x = tntt.randn([2]*20, [1] + [6]*19 + [1])
A = A - A + A - A + A
x = x - x + x - x + x
yy = (A @ x).round(1e-14)

y = tntt.amen_mv(A, x, verbose=True, use_cpp=True)

print((y-yy).norm()/yy.norm())