import pandas as pd

from PIL import Image, ImageDraw, ImageFont
import numpy as np

# PC font path
font_path = 'C:/Windows/Fonts/'
courier = 'courbd.ttf'

# Mac font path
# font_path = 'Library/Fonts/'
# courier = 'cour.dfont'

fnt_a_40 = ImageFont.truetype(font_path + 'Arial.ttf', 40)
fnt_a_100 = ImageFont.truetype(font_path + 'Arial.ttf', 100)
fnt_c_40 = ImageFont.truetype(font_path + courier, 40)

try:
    chase_seq = pd.read_csv('Chase - Random - 20190708 - 0751.csv')
except:
    print('Can not open log file.')

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
    
    d.text((24, 10), 'CHASE', font=fnt_a_100, fill=(0,0,0))
    d.text((32, 108), 'A toy-text reinforcement learning environment', font=fnt_a_40, fill=(0,0,0))    
    d.text((10,170), arena, font=fnt_c_40, fill=(0,0,0))
    d.text((32, 940), 'Episode: ' + e 
                      + '        Step: ' + s
                      + '        Action: ' + a
                      + '        Reward: ' + r, font=fnt_a_40, fill=(0,0,0))    
    
    if row%10 == 0:
        print('Completed frame:', row)

    img.save('test' + str(row).zfill(4) + '.png')    
