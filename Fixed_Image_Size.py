
# coding: utf-8

# # Fixed Image Size

# In[1]:


import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
# %matplotlib inline


# In[2]:


def makeLOGfilter(size,sigma):
    kernel = np.zeros((size,size))
    for i in range(size):
        for j in range(size):
            x = i - int(size/2)
            y = j - int(size/2)
            kernel[i,j] = -(1- ((x**2+y**2)/(2*sigma**2)))*(np.exp(-(x**2+y**2)/(2*sigma**2)))/(3.14*sigma**4)
    return (sigma**2)*(kernel)
    return (sigma**2)*(kernel - np.mean(kernel))


# In[3]:


import sys
def makeLOGfilter2(size,sigma):
    siz=size//2
    # This creates a LoG filter
    x = y = np.linspace(-siz, siz, 2*siz+1)
    x, y = np.meshgrid(x, y)
    arg = -(x**2 + y**2) / (2*sigma**2)
    h = np.exp(arg)
    h[h < sys.float_info.epsilon * h.max()] = 0
    h = h/h.sum() if h.sum() != 0 else h
    h1 = h*(x**2 + y**2 - 2*sigma**2) / (sigma**2)
    return h1 - h1.mean()


# In[4]:


def conv(img,kernel):
    s = kernel.shape[0] //2
    m,n = img.shape
    temp = (np.zeros((m+2*s,n+2*s)))
    temp[s:-s,s:-s] = img
    for i in range(s,m+s):
        for j in range(s,n+s):
            img[i-s,j-s] = np.sum(temp[i-s:i+s+1,j-s:j+s+1]*kernel)
    return (img)


# In[5]:


def scalespace(img,scale,sigma,name,num):
    m,n = img.shape
    octave = np.zeros((scale,m,n))
    for i in range(0,scale):
        if(np.ceil(6*sigma[i])%2):
            size = np.ceil(6*sigma[i])
        else:
            size = np.ceil(6*sigma[i])+1
        kernel = makeLOGfilter2(size.astype(int),sigma[i])
        octave[i] = np.square(conv(np.copy(img),kernel))        
    return octave


# In[6]:


def nonmaximasuppression(img,octave,scale,sigma):
    m,n = octave[0].shape
    temp = np.zeros((scale,m,n))    
    th = 0.01
    rad = [np.int(np.ceil(np.sqrt(2)*sigma[i])) for i in range(scale)]
    for curr in range(0,scale):
        for x in range(rad[curr],m-rad[curr]):
            for y in range(rad[curr],n-rad[curr]):
                if (curr == 0 and octave[curr,x,y]>th and octave[curr,x,y] == np.max([octave[curr,x-1:x+2,y-1:y+2], octave[curr+1,x-1:x+2,y-1:y+2]])):
                    temp[curr,x,y] = 1
                    img = cv2.circle(img, (y,x), rad[curr], (0,0,255), 1)
                elif (curr == scale-1 and octave[curr,x,y]>th and octave[curr,x,y] == np.max([octave[curr,x-1:x+2,y-1:y+2], octave[curr-1,x-1:x+2,y-1:y+2]])):
                    temp[curr,x,y] = 1
                    img = cv2.circle(img, (y,x), rad[curr], (0,0,255), 1)
                elif(0<curr<scale-1 and octave[curr,x,y]>th and octave[curr,x,y] == np.max([octave[curr,x-1:x+2,y-1:y+2], octave[curr-1,x-1:x+2,y-1:y+2], octave[curr+1,x-1:x+2,y-1:y+2]])):
                    temp[curr,x,y] = 1
                    img = cv2.circle(img, (y,x), rad[curr], (0,0,255), 1)
    return img


# # Main Code Starts Here:
# * Reading the image
# * Calling the functions conv and nonmaximasuppression

# In[7]:


while 1:
    try:
        path = input("Write the image file name in the folder TestImages4Project with extension (eg. fishes.jpg): ")
        start_time = time.time()
        img = cv2.imread("TestImages4Project\{}".format(path),0)
        img = img/np.max(img)
        break
    except:
        print("Wrong file name given. Enter again.")
        continue


# In[8]:


scale = 15
k = 1.24
sigma = np.sqrt(0.5)
sigma = [sigma*(k**i) for i in range(scale)]
img = img/np.max(img)

octave = scalespace(img,scale,sigma,path,0)
img = cv2.imread("TestImages4Project\{}".format(path))
img[:,:,0] = img[:,:,2]
img[:,:,1] = img[:,:,2]
img = nonmaximasuppression(img,octave,scale,sigma)


# * Displaying and saving the image

# In[9]:


print("Time Taken:", time.time()-start_time)
cv2.imshow('Blob Circles',img)
cv2.imwrite('blob_{}.png'.format(path),img)
cv2.waitKey(0)
cv2.destroyAllWindows()

