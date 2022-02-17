# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 16:43:01 2019

@author: b03881
"""
#https://fr.wikipedia.org/wiki/Th%C3%A9or%C3%A8me_de_Bayes#/media/File:Bayes%27_Theorem_MMB_01.jpg
import numpy as np
from skimage import io

photo = io.imread(r"U:\Nouveau dossier\data_scentist\data\mldata\Bayes_Theorem_MMB_01_1024_px.jpg")
type(photo)

import matplotlib.pyplot as plt
plt.imshow(photo)

plt.imshow(photo[::-1])

plt.imshow(photo[:,::-1])

#plt.imshow(photo[50:150,150:280])
plt.imshow(photo[330:450,50:350])

plt.imshow(photo[::2,::2])

photo_sin = np.sin(photo)
print(photo_sin)

photo_masked =np.where(photo>100,255,0)
plt.imshow(photo_masked)

plt.imshow(photo[:,:,0])
plt.imshow(photo[:,:,0].T)