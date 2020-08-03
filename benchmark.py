#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
get_ipython().system('{sys.executable} -m pip install progressbar')


# In[ ]:


import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from scipy.linalg import *
from scipy.fft import fft
import time
import progressbar


# In[ ]:


# preperation
rnd.seed(1)
t_list = []
inv_list = []
fft_list = []
eig_list = []


# In[ ]:


bar = progressbar.ProgressBar(maxval=100,     widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
bar.start() 
# testing loop starts.
for i in range(100):
    start = time.perf_counter()
    A = rnd.randn(500,5000)
    # eigenvalue 
    largest_eig = np.flip(np.sort(eigvals(A.dot(A.T))))[0]
    eig_list.append(largest_eig)
    # inversion
    invA = inv(A.dot(A.T))
    inv_list.append(invA)
    # fft
    fft_list.append(fft(A.dot(A.T)))
    
    # timing
    timer = time.perf_counter()-start
    t_list.append(timer)
    bar.update(i+1)

bar.finish()
print('average loop time: %1.3f' % np.mean(t_list))


# In[ ]:


# write results
import socket
host = socket.gethostname()
f=open('benchmark.csv','a')
f.write("%s, %1.3f\n" % (host,np.mean(t_list)))
f.close()

