#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 14:10:35 2019

@author: andrewbuttery
"""

import pandas as pd

from PIL import Image, ImageDraw, ImageFont
import numpy as np

chase_seq = pd.read_csv('Chase - human - 20190707 - 1748.csv')

for row in range(chase_seq.shape[0]):
    arena = chase_seq.iloc[row, 5:406].values.reshape(20, 20)
    e = str(chase_seq.iloc[row, 0])
    s = str(chase_seq.iloc[row, 1])
    a = str(chase_seq.iloc[row, 2])
    r = str(chase_seq.iloc[row, 3])
    arena = np.array2string(arena)
    
    arena = ' ' + arena.replace('0',' ').replace('1','X').replace('2','X').replace('3','R').replace('4','A').replace('[','').replace(']','')
    
    img = Image.new('RGB', (1920, 1080), color = 'white')
    d = ImageDraw.Draw(img)
    fnt = ImageFont.truetype('/Library/Fonts/Arial.ttf', 100)
    d.text((24, 10), 'CHASE', font=fnt, fill=(0,0,0))
    fnt = ImageFont.truetype('/Library/Fonts/Arial.ttf', 40)
    d.text((32, 108), 'A toy-text reinforcement learning environment', font=fnt, fill=(0,0,0))    
    fnt = ImageFont.truetype('/Library/Fonts/Courier.dfont', 40)
    d.text((10,180), arena, font=fnt, fill=(0,0,0))

    
    fnt = ImageFont.truetype('/Library/Fonts/Arial.ttf', 40)
    d.text((32, 900), 'Episode: ' + e 
                      + '        Step: ' + s
                      + '        Action: ' + a
                      + '        Reward: ' + r, font=fnt, fill=(0,0,0))    
    # d.text((10,10), "Hello World", font=fnt, fill=(255,255,0))

    img.save('test' + str(row).zfill(4) + '.png')    
