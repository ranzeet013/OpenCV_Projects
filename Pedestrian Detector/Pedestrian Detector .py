#!/usr/bin/env python
# coding: utf-8

# In[21]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[22]:


import imutils
from imutils.object_detection import non_max_suppression


# In[62]:


image = cv2.imread(r'C:\Users\DELL\Desktop\python project\open cv2\New folder (16)\people.jpg')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Orginal Image')
plt.show()


# # HOG Descriptor :

# The Histogram of Oriented Gradients (HOG) is a feature descriptor used primarily for object detection in computer vision tasks. It's particularly effective at detecting objects in images based on their shapes and edge orientations. The HOG descriptor has been widely used in pedestrian detection, but it can also be applied to other object detection tasks.

# In[90]:


hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
(r, w) = hog.detectMultiScale(image, winStride = (8, 8), 
                              padding = (8, 8), 
                              scale = 1.01)


# In[91]:


rect = np.array([[x, y, x+w, y+h] for (x, y, w, h) in r])


# In[92]:


rect_NMS = non_max_suppression(rect, 
                               probs = None, 
                               overlapThresh = 0.6)


# In[93]:


for (x1, y1, x2, y2) in rect_NMS:
    cv2.rectangle(image, (x1, y1), (x2, y2), (0 ,255, 0), 2)


# In[97]:


plt.imshow(image)


# In[ ]:




