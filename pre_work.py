# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 14:48:36 2019

@author: Administrator
"""
from PIL import Image
import numpy as np

#############################补充图片和标签的完整地址############################
imagedir='C:\\Users\\Administrator\\Desktop\\Try\\data\\rgb\\image\\2.png'
labeldir='C:\\Users\\Administrator\\Desktop\\Try\\data\\rgb\\label\\2.png'
savedir='C:\\Users\\Administrator\\Desktop\\Try\\data\\train\\'
###############################################################################

############################打开图片并创建存储空间###############################
img = Image.open(imagedir)
label = Image.open(labeldir)

#img = img.crop((10,0,1014,784))
label = label.crop((32,32,991,751))

#img = np.asarray(img, dtype='float64')
label = np.asarray(label,dtype='float64')

#mark0_min = np.zeros(len(label[0]))
#mark1_min = np.zeros(len(label[0]))
#mark2_min = np.zeros(len(label[0]))
#mark3_min = np.zeros(len(label[0]))
#mark4_min = np.zeros(len(label[0]))
#mark5_min = np.zeros(len(label[0]))
#mark6_min = np.zeros(len(label[0]))
#mark7_min = np.zeros(len(label[0]))
#mark8_min = np.zeros(len(label[0]))

label_dict = ['background.','one.','two.','three.','four.','five.','six.','seven.','eight.']
#for i in range(len(label[0])):
#
#    mark1 = np.where(label[:,i]==1)   
#    mark1_min[i] = np.min(mark1)
#
#    mark2 = np.where(label[:,i]==2)   
#    mark2_min[i] = np.min(mark2)
#    
#    mark3 = np.where(label[:,i]==3)   
#    mark3_min[i] = np.min(mark3)
#    
#    mark4 = np.where(label[:,i]==4)   
#    mark4_min[i] = np.min(mark4)
#
#    mark5 = np.where(label[:,i]==5)   
#    mark5_min[i] = np.min(mark5)
#
#    mark6 = np.where(label[:,i]==6)   
#    mark6_min[i] = np.min(mark6)
#
#    mark7 = np.where(label[:,i]==7)   
#    mark7_min[i] = np.min(mark7)
#
#    mark8 = np.where(label[:,i]==8)   
#    mark8_min[i] = np.min(mark8)

mark = np.zeros([len(label[0]),9])

for i in range(len(label[0])):
    for k in range(8):
        j = k+1
        mark[i,j] = np.min(np.where(label[:,i]==j))

for col in range(len(label[0])):
    if col<200:
        mark[col,0] = mark[col,8]+10
    elif col<400:
        mark[col,0] = mark[col,8]+50
    elif col<600:
        mark[col,0] = mark[col,1]-5
    else :
        if mark[col,5]+10 < mark[col,6]:
            mark[col,0] = mark[col,5]+10
        else:
            mark[col,0] = mark[col,5]+3
##################################剪图片并保存#################################
#k=0   
#for col in range(0,959,5):
#    
#    row = mark1_min[col]
#    label1 = img.crop((col,row-32,col+65,row+33))
#    label1_65 = Image.fromarray(np.uint8(label1))
#    label1_name = 'one.'+str(k)+'.png'
#    save1 = savedir+label1_name 
#    label1_65.save(save1)
#    
#    row = mark2_min[col]
#    label2 = img.crop((col,row-32,col+65,row+33))
#    label2_65 = Image.fromarray(np.uint8(label2))
#    label2_name = 'two.'+str(k)+'.png'
#    save2 = savedir+label2_name 
#    label2_65.save(save2)
#    
#    row = mark3_min[col]
#    label3 = img.crop((col,row-32,col+65,row+33))
#    label3_65 = Image.fromarray(np.uint8(label3))
#    label3_name = 'three.'+str(k)+'.png'
#    save3 = savedir+label3_name 
#    label3_65.save(save3)
#    
#    row = mark2_min[col]
#    label4 = img.crop((col,row-32,col+65,row+33))
#    label4_65 = Image.fromarray(np.uint8(label4))
#    label4_name = 'four.'+str(k)+'.png'
#    save4 = savedir+label4_name 
#    label4_65.save(save4)    
#
#    row = mark5_min[col]
#    label5 = img.crop((col,row-32,col+65,row+33))
#    label5_65 = Image.fromarray(np.uint8(label5))
#    label5_name = 'five.'+str(k)+'.png'
#    save5 = savedir+label5_name 
#    label5_65.save(save5) 
#    
#    row = mark6_min[col]
#    label6 = img.crop((col,row-32,col+65,row+33))
#    label6_65 = Image.fromarray(np.uint8(label6))
#    label6_name = 'six.'+str(k)+'.png'
#    save6 = savedir+label6_name 
#    label6_65.save(save6)
#    
#    row = mark7_min[col]
#    label7 = img.crop((col,row-32,col+65,row+33))
#    label7_65 = Image.fromarray(np.uint8(label7))
#    label7_name = 'seven.'+str(k)+'.png'
#    save7 = savedir+label7_name 
#    label7_65.save(save7)
#    
#    row = mark8_min[col]
#    label8 = img.crop((col,row-32,col+65,row+33))
#    label8_65 = Image.fromarray(np.uint8(label8))
#    label8_name = 'eight.'+str(k)+'.png'
#    save8 = savedir+label8_name 
#    label8_65.save(save8)
#    
#    k+=1
k=0
for col in range(0,len(label[0]),5):
    k+=1
    for i in range(8):
        row = mark[col,i]
        label = img.crop((col,row-32,col+65,row+33))
        lebel = Image.fromarray(np.uint8(label))
        name = label_dict[i]+str(k)+'.png'
        save = savedir+name
        label.save(save)
    


















