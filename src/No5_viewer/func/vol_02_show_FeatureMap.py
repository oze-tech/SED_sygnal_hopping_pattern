# -*- coding: utf-8 -*-
"""
<vol_03_show_FeatureMap>
@author: oze_tech

this cord is function for machine Learning

===============
1. get layer info
2. show curve
===============

"""


from tensorflow.keras.models import Model
import matplotlib.pyplot as plt



# =============================================================================
# 1. showing feature map
# =============================================================================

def show_feature_map(model, test_img, show_idx = 0):
    
    """    
    input:
        model    : model with weight
        test_x   : want to show feature map
        show_idx : show only test_x of "show_idx" 
    """ 

    # 1. get conv2d layer    
    Lc = get_layer_list(model,"max_pooling2d")
    x0 = model.get_layer(index=0)
    
    
    # 2. show filter by each layer
    for i, Lci in enumerate(Lc):
        Lci_model = Model(inputs = x0.output, outputs = Lci.output)
        pred_map  = Lci_model.predict(test_img)[show_idx]              # (show_idx, 64, Y, ch)
        
        # show 2x2 filter
        show_2x2_img(pred_map, "map ", i)
        
    
    
# =============================================================================
# 2. showing filter parameter
# =============================================================================
    
def show_filter(model):
    
    """    
    input:
        model    : model with weight
    """ 
    
    # 1. get conv2d layer
    Lc = get_layer_list(model, "conv2d")
    
    # 2. show filter by each layer
    for i, Lci in enumerate(Lc):
        
        target_layer = Lci.get_weights()[0]                            # (3, 3, y, ch)
        # show 2x2 filter (batch,X,y,ch)
        show_2x2_img(target_layer[:, :, 0, :], "filter ", i)
        
        
        
       
# =============================================================================
# 0. get_layer_list
# =============================================================================
             
def get_layer_list(model, layer_name = 'conv2d'):
    """
    return : get_layer array
    """
    show_Lc = []

    names_conv2d = [l.name for l in model.layers if layer_name in l.name]
    for Lc_name in names_conv2d:
        show_Lc.append(model.get_layer(Lc_name))
        
    return show_Lc


       
# =============================================================================
# 0. show 2x2 img
# =============================================================================
             
def show_2x2_img(target, title, i):
    
    plt.figure()
    plt.suptitle("Lc_" + str(i))
    
    for j in range(4):
        plt.subplot(2, 2, j + 1)
        plt.imshow(target[:, :, j].T, cmap="gray")
        plt.title(title + str(j))
        plt.axis("off")
        
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.show()
