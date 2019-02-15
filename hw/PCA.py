import numpy as np
cov = np.array([[0.61655555, 0.615444444 ],[0.615444444, 0.716555556 ]])
print(cov)

w, v = np.linalg.eig(cov)

print(w,"\n",v, "\n")


cov = np.array([[0.715, -1.39],[-1.39, 2.72]])
print(cov)

w, v = np.linalg.eig(cov)

print(w,"\n",v, "\n")

print(v[:, 1])
