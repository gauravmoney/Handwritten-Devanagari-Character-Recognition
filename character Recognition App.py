# -*- coding: utf-8 -*-
"""
Created on Tue may 15 21:14:49 2020

@author: Gaurav
"""

from tkinter import Label,Tk,Canvas,LEFT
from tkinter.ttk import Button
from PIL import Image,ImageDraw
from keras.models import load_model
import numpy as np

main_window=Tk()
main_window.title("Handwritten Devanagari Character Recognition - Gaurav")
main_window.iconbitmap(r'ai.ico')

index2label={0: "क_ka", 1: "ख_kha", 2:"ग_ga", 3:"घ_gha", 4:"ङ_kna", 5:"च_cha", 6:"छ_chha", 7:"ज_ja",
           8:"झ_jha", 9:"ञ_yna", 10:"ट_ta", 11:"ठ_tha", 12:"ड_daa", 13:"ढ_dhaa", 14:"ण_adha", 15:"त_ta",
           16:"थ_tha", 17:"द_da", 18:"ध_dha", 19:"न_na", 20:"प_pa", 21:"फ_pha", 22:"ब_ba", 23:"भ_bha",
           24:"म_ma", 25:"य_yaw", 26:"र_ra", 27:"ल_la", 28:"व_waw", 29:"श_saw", 30:"ष_sa", 31:"स_sa",
           32:"ह_ha", 33:"क्ष_chhya", 34:"त्र_tra", 35:"ज्ञ_gya", 36:"०_0", 37:"१_1", 38:"२_2", 39:"३_3",
           40:"४_4", 41:"५_5", 42:"६_6", 43:"७_7", 44:"८_8", 45:"९_9"}

#Canvas Code Section   
canvas_width=600
canvas_height=500
cnvs_title=Label(main_window,height=1,text="Canvas Area",highlightthickness=1,borderwidth=0,background='#98AFC7')
cnvs_title.grid(row=0,column=0,sticky='EW')
cnvs = Canvas(main_window,width=canvas_width,height=canvas_height,background='#F0F8FF')
cnvs.grid(row=1,column=0,sticky='NEWS')

image = Image.new("L", (canvas_width, canvas_height), 0)
draw = ImageDraw.Draw(image)



import tensorflow as tf
import keras.backend.tensorflow_backend as tfback
print("tf.__version__ is", tf.__version__)
print("tf.keras.__version__ is:", tf.keras.__version__)

def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

tfback._get_available_gpus = _get_available_gpus
                    





def getImage():
    img=np.array(image)
    x0,x1=(0,0)
    y0,y1=(0,0)
    x_flag=True
    y_flag=True
    
    for i in range(0,img.shape[1]):
        if img[:,i].any():
            if x_flag:
                x0=i
                x_flag=False
            else:
                x1=i
                
        if i<img.shape[0]:
            if img[i,:].any():
                if y_flag:
                    y0=i
                    y_flag=False
                else:
                    y1=i
                    
    pad=10
    img2=img[y0-pad:y1+pad,x0-pad:x1+pad]
    pil_image=Image.fromarray(img2)
    pil_image=np.array(pil_image.resize((32,32)))
    pil_image_reshaped=pil_image.reshape(1, 32, 32, 1).astype('float32')/255
    probab=model.predict([pil_image_reshaped])[0]
    probab_dict=dict(zip(index2label.values(),probab))
    sorted_probab_dict=sorted(probab_dict.items(), key=lambda kv: kv[1],reverse=True)
    to_display="Predictions:-\n\n\t"+"\n\t".join([ key+' : '+str(round(value*100,2))+'%' for key,
                                                  value in sorted_probab_dict[0:10]])
    result_area.configure(text=to_display)
 
def paint( event ):
   black_color = "#000000"
   r=3
   x1, y1 = ( event.x - r ), ( event.y - r )
   x2, y2 = ( event.x + r ), ( event.y + r )
   cnvs.create_oval( x1, y1, x2, y2, fill = black_color)
   draw.ellipse([x1,y1,x2,y2], fill=255)

def clear( event ):
    cnvs.delete("all")
    draw.rectangle([0,0,canvas_width,canvas_height],0)
    result_area.configure(text="Predictions\n\n")
    
cnvs.bind( "<B1-Motion>", paint )

##Canvas Controls
clear_btn=Button(text='Clear')
clear_btn.grid(row=2,column=0,sticky='EW')
clear_btn.bind("<1>",clear)
run_btn=Button(text='Run',command=getImage)
run_btn.grid(row=2,column=1,sticky='EW')
#Result Code Section
text_title=Label(main_window,height=1,text='Predictions',highlightthickness=1,borderwidth=0,background='#98AFC7')
text_title.grid(row=0,column=1,sticky='EW')
result_area=Label(main_window,width=50,text='Predictions\n\n',foreground='#00FF00',
                  highlightthickness=1,borderwidth=0,background='#2C3539')
result_area.config(justify=LEFT,pady=4,padx=10,anchor='w',font=("Courier", 15))
result_area.grid(row=1,column=1,sticky='NS')                  

model=load_model("New_Model.h5")

main_window.mainloop()