# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 16:58:47 2019

@author: Administrator
"""

import os
image_list=[]
num_files=0
path='C:\\Users\\Administrator\\Desktop\\Try\\data\\test\\'
for fn in os.listdir(path):
        num_files += 1
        


for i in range (num_files):
    k=i+1
    image_path=path+str(k)+'.png'
    image_list.append(image_path)
    