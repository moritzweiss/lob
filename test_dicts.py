import time 
from collections import OrderedDict
from sortedcontainers import SortedDict


class MyClass:
    def __init__(self, x):
        self.x = x

a = [1, 2, 3]

C = MyClass(a)

C.x = a
print(C.x)

a.append(4)

print(C.x)



d = OrderedDict()

d[1] = 2 
d[2] = 3
d[3] = 4
d[2] = 5

print(d.popitem(last=False))
print(d.popitem(last=False))
print(d.popitem(last=False))

d[1] = 2 
d[2] = 3
d[3] = 4
d[2] = 5


for k in reversed(d): 
    print(k)

# remove order from dict 
del d[1]

# add order to the back of the dict queue 
d[4] = 10
print(d)
# find queue position 
# linear time 
q = 0
for k in d:
    if k == 4:
        print(f'queue position is {q}')
        break
    q += 1

# keys should be order_id if increasing 
# otherwise creation time might be a good idea
# after any event in the order book, update the tick time by one 
# any incoming order: cancel, limit, market receives the current tick time as a time stamp 


s = SortedDict()

s[1] = 2
s[2] = 3
s[3] = 4
s[2] = 5

q=s.index(3)
print(f'queue position is {q}')

for n in range(2000):
    s[n] = n
    d[n] = n

start = time.time()
q = s.index(1000)
end = time.time()
print(end-start)
print(f'queue position is {q}')

start = time.time()
q = 0 
for k in d:
    if k == 1000:
        print(f'queue position is {q}')
        break
    q += 1
end = time.time()
print(end-start)

























































