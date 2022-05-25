# -*- coding: utf-8 -*-
"""
<vol_02_show_dataset>
@author: oze_tech

this cord is function for the prediction

===============
1. show data 
2. show label
3. show predoct label
==============="""

import matplotlib.pyplot as plt

    
#==========================================================================
# 1. show spectrogram 
#==========================================================================
  
def show_spec_data(spec_data,title_="",new_=True):
    
    if new_:
        plt.figure()
        
    plt.imshow(spec_data,origin='lower',interpolation='none')  
    plt.title(title_)
    plt.tight_layout()
        
    

#==========================================================================
# 2. show event label
#==========================================================================

def show_event_label(event_label, title_="", new_=True):
    
    def rescale(im, nR, nC):
        number_rows = len(im)     # source number of rows 
        number_columns = len(im[0])  # source number of columns 
        return [[ im[int(number_rows * r / nR)][int(number_columns * c / nC)]  
                     for c in range(nC)] for r in range(nR)]
                
    
    if new_:
        plt.figure()
        

    res_label = rescale(event_label, 64, 64)
    y_resize  = len(res_label) -1
    y_size    = len(event_label)
    plt.yticks([0,y_resize//2,y_resize], [0,y_size//2,y_size])
    plt.imshow(res_label, origin='lower',interpolation='none')  
    plt.title(title_)
    plt.tight_layout()
    
    
    
#==========================================================================
# 3.show data and label and pred 
#==========================================================================
def pred_label(model, X):
     return (model.predict(X) > 0.5 ) * 1
 

    

def show_predict_data(x_test, y_test, z_test, i = 0):
    
    plt.figure()
    plt.subplot(3, 1, 1)
    show_spec_data(x_test[i,:,:,0].T,'input spectrogram No.'+str(i),False)
    
    plt.subplot(3, 1, 2)
    show_event_label(y_test[i].T,'ideal event label No.'+str(i), False)
    
    plt.subplot(3, 1, 3)
    show_event_label(z_test[i].T,'predected event label No.'+str(i), False)
    
    plt.rcParams["font.size"] = 7
    plt.show()
    
    