#!/usr/bin/env python
# coding: utf-8

# # Finding Waldo ( mini project )

# Creating a "Find Waldo" project using the OpenCV (cv2) library involves building a computer vision application that can locate the character "Waldo" in a given image. This project can be a fun way to showcase object detection techniques using OpenCV. 

# # Importing Libraries :

# cv2 (OpenCV):
# OpenCV (Open Source Computer Vision Library) is a popular open-source computer vision and machine learning software library. It provides a wide range of tools and functions for image and video processing, object detection, facial recognition, and more. OpenCV is widely used in applications related to computer vision, robotics, augmented reality, and medical imaging.
# 
# numpy:
# NumPy is a fundamental package for scientific computing in Python. It provides support for large, multi-dimensional arrays and matrices, along with a variety of mathematical functions to operate on these arrays. NumPy is a cornerstone library for numerical computations in Python and is used extensively in data analysis, machine learning, and scientific research.
# 
# matplotlib:
# Matplotlib is a widely-used Python plotting library that helps create static, interactive, and animated visualizations in Python. It provides a flexible interface for creating various types of plots, charts, and graphs, making it valuable for data visualization and analysis tasks. Matplotlib is often used in conjunction with other libraries like NumPy for creating informative and visually appealing visualizations.

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# # Beach Image :

# In[4]:


beach_image = cv2.imread(r'C:\Users\DELL\Desktop\python project\open cv2\New folder (14)\wheres_waldo.jpg')
plt.imshow(cv2.cvtColor(beach_image, cv2.COLOR_BGR2RGB))
plt.title('Beach Image')
plt.show()


# # Convering Orginal Image To Gary Image :

# In[6]:


gray_image = cv2.cvtColor(beach_image, cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.cvtColor(gray_image, cv2.COLOR_BGR2RGB))
plt.show()
cv2.imwrite('gray_beach_image.jpg', gray_image)


# # Waldo Image :

# In[7]:


waldo_image = cv2.imread(r'C:\Users\DELL\Desktop\python project\open cv2\New folder (14)\color_waldo_template.jpg')
plt.imshow(cv2.cvtColor(waldo_image, cv2.COLOR_BGR2RGB))
plt.title('Waldo Image')
cv2.imwrite('Waldo_Image.png', waldo_image)


# # Converting Waldo Image into Gray Image :

# In[10]:


gray_waldo = cv2.cvtColor(waldo_image, cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.cvtColor(gray_waldo, cv2.COLOR_BGR2RGB))
plt.title('Gray Waldo')
plt.show()
cv2.imwrite('Gray_waldo_image.png', gray_waldo)


# # Finding Waldo :

# In[11]:


result = cv2.matchTemplate(gray_image, gray_waldo, cv2.TM_CCOEFF)
min_value, max_value, min_lock, max_lock = cv2.minMaxLoc(result)

top_left = max_lock
bottom_right = (top_left[0] + 50, top_left[1] + 50)
cv2.rectangle(beach_image, top_left, bottom_right, (0, 0, 255), 5)

plt.imshow(cv2.cvtColor(beach_image, cv2.COLOR_BGR2RGB))
plt.title('Where Is Waldo')
plt.show()
cv2.imwrite('Where_Is_Waldo.png', beach_image)

