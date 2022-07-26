# import subprocess
# kam = [1,2,3,[1,2,3]]
# # print(str(kam))
# # str. replace()
# out = subprocess.run('python tester.py '+str(kam).replace(' ',''), shell=True)
# print(out)

import time
sa = time.time()
initial_list = "[11, 21, 29, 46, [19 ,18],12]"

print("initial string", initial_list)
print(type(initial_list))

op = initial_list.strip('][').split(', ')

print("final list", op)
print(type(op))
print(time.time() - sa)