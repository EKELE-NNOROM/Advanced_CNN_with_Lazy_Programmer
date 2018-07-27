#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 22:47:07 2018

@author: ekele
"""

import os

# fruits-360 classes
#classes = [
#  'Apple Golden 1',
#  'Avocado',
#  'Lemon',
#  'Mango',
#  'Kiwi',
#  'Banana',
#  'Strawberry',
#  'Raspberry'
#]


# blood-cell classes
classes = ['LYMPHOCYTE', 'MONOCYTE', 'EOSINOPHIL', 'NEUTROPHIL']    
    
def mkdir(p):
    if not os.path.exists(p):
        os.mkdir(p)
        
def link(src, dst):
    if not os.path.exists(dst):
        os.symlink(src, dst, target_is_directory=True)
        
#mkdir('../large_files/fruits-360-small')
mkdir('../large_files/blood-cells-small')

#train_path_from = os.path.abspath('../large_files/fruits-360/Training')
#valid_path_from = os.path.abspath('../large_files/fruits-360/Test')
train_path_from = os.path.abspath('../large_files/blood-cells/dataset2-master/dataset2-master/images/TRAIN')
valid_path_from = os.path.abspath('../large_files/blood-cells/dataset2-master/dataset2-master/images/TEST')

#train_path_to = os.path.abspath('../large_files/fruits-360-small/Training')
#valid_path_to = os.path.abspath('../large_files/fruits-360-small/Test')
train_path_to = os.path.abspath('../large_files/blood-cells-small/TRAIN')
valid_path_to = os.path.abspath('../large_files/blood-cells-small/TEST')

mkdir(train_path_to)
mkdir(valid_path_to)

for c in classes:
    link(train_path_from + '/' + c, train_path_to + '/' + c)
    link(valid_path_from + '/' + c, valid_path_to + '/' + c)
    

# list the classes available for blood-cell data    
#print(os.listdir(train_path_from))
    
    
    
    
    