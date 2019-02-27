
# coding: utf-8

# # Varying Image Size

# In[1]:


import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
# %matplotlib inline


# In[2]:


import sys
def makeLOGfilter(size,sigma):
    siz=size//2
    # This creates a LoG filter
    x = y = np.linspace(-siz, siz, 2*siz+1)
    x, y = np.meshgrid(x, y)
    arg = -(x**2 + y**2) / (2*sigma**2)
    h = np.exp(arg)
    h[h < sys.float_info.epsilon * h.max()] = 0
    h = h/h.sum() if h.sum() != 0 else h
    h1 = h*(x**2 + y**2 - 2*sigma**2) / (sigma**4)
    return h1 - h1.mean()


# In[3]:


def conv(img,kernel):
    s = kernel.shape[0] //2
    m,n = img.shape
    temp = (np.zeros((m+2*s,n+2*s)))
    temp[s:-s,s:-s] = img
    for i in range(s,m+s):
        for j in range(s,n+s):
            img[i-s,j-s] = np.sum(temp[i-s:i+s+1,j-s:j+s+1]*kernel)
    return img


# In[4]:


def subsample(img): 
    return img[::2,::2] 
def upsample(image):
    m,n = image.shape
    upsample = np.ones((2*m,2*n))
    upsample[::2,::2] = image
    upsample[1::2,1::2] = image
    upsample[::2,1::2] = image
    upsample[1::2,::2] = image
    return upsample


# In[5]:


def nonmaximasuppression(img,octave,scale,sigma):
    m,n = octave[0].shape
    temp = np.zeros((scale,m,n))    
    th = 0.08
    sigma = [sigma*(k**i) for i in range(scale)]
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
#                     print(curr)
    return img


# In[6]:


def showimg(octave,scale,num):
    for i in range(0,scale):
        plt.figure(num*(scale+1)+i+1)
        plt.imshow(octave[i],cmap='gray')
        plt.title('Octave{}, Scale{}'.format(num,i))
        plt.savefig('Non-Maxima Suppression: Sigma = {}.png'.format(1.4**(num+i)*1.4))
        plt.show()


# # Main Code Starts Here:
# * Reading the image
# * Calling the functions conv and nonmaximasuppression

# In[ ]:


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
scale = 15
k = 1.24
sigma = np.sqrt(0.5)
size = np.round(6*sigma)
if(size%2 != 0):
    size = size +1
m,n = img.shape
octave = np.zeros((scale,m,n))
kernel = makeLOGfilter(size,sigma)
for i in range(scale):
    m,n = np.int(img.shape[0]/(k**i)), np.int(img.shape[1]/(k**i))
    octave[i] = cv2.resize(np.square(conv(cv2.resize(img,(m,n)),kernel)), (img.shape[1],img.shape[0]))


# In[ ]:


img = cv2.imread("TestImages4Project\{}".format(path))
img[:,:,0] = img[:,:,2]
img[:,:,1] = img[:,:,2]
img = nonmaximasuppression(img,octave,scale,sigma)
print("Time Taken:", time.time()-start_time)
# plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
# plt.show()
cv2.imshow('Blob Circles',img)
cv2.imwrite('blob_{}.png'.format(path),img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


def updateoctave(img,scale,num):
    size = 5 
    sigma = np.sqrt(2)*(num+1)
    k = np.sqrt(0.5)
    m,n = img.shape
    octave = np.zeros((scale,m,n))
    for i in range(0,scale):
        kernel = makeLOGfilter(size,(k**(i-1))*sigma)
        octave[i] = conv(img,kernel)
        plt.figure(num*(scale+1)+i)
        plt.imshow(octave[i],cmap='gray')
        plt.title('Octave{}, Scale{}'.format(num,i))
    plt.show()
    return octave

