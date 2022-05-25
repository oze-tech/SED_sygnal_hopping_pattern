# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 03:12:06 2021

@author: oze_tech
音響イベントからF-Score,ERを計算するプログラム
理想イベントと対象のイベントを入力とする
"""

import numpy as np

# ER値の計算
def cal_Error_rate(ideal_event,pred_event):
    """
    input : T_event,F_event,イベントは2D
    S : 置換エラー数（イベントを誤った数）
    D : 削除エラー数（存在するイベントを誤り検出した数）
    I : 挿入エラー数（存在しないイベントを誤り検出した数）
    N : 存在するイベント総数
    """
    ideal_event = (ideal_event>0.5)*1
    pred_event  = (pred_event >0.5)*1
    
    T_event = ideal_event.flatten()
    P_event = pred_event.flatten()

    N = T_event.size
    S = np.count_nonzero(T_event != P_event)
    
    # イベントが存在するか否か
    T_index = T_event == 1
    F_index = T_event == 0

    D = np.count_nonzero(T_event[T_index] != P_event[T_index])
    I = np.count_nonzero(T_event[F_index] != P_event[F_index])

    return (S+D+I)/N



# F値の計算
def cal_F_score(ideal_event,pred_event):
    """
    input : T_event,F_event,イベントは2D
    TP : イベントが実在すると予測して実際に実在する数
    EP : イベントが存在すると予測したが実際には実在しない数
    FN : イベントが実在しないと予測したが実際には実在する数
    """

    ideal_event = (ideal_event>0.5)*1
    pred_event  = (pred_event >0.5)*1
    
    T_event = ideal_event.flatten()
    P_event = pred_event.flatten()

    # イベントを存在すると予測したか否か
    Pred_T_index = P_event == 1
    Pred_F_index = P_event == 0

    TP = np.count_nonzero(T_event[Pred_T_index] == P_event[Pred_T_index])
    EP = np.count_nonzero(T_event[Pred_T_index] != P_event[Pred_T_index])
    FN = np.count_nonzero(T_event[Pred_F_index] != P_event[Pred_F_index])
    
    return 2*TP/(2*TP+EP+FN)



if __name__ == '__main__':
    
    a = np.arange(20).reshape(4,5)
    b = a.copy()
    b[-1] = 0
    print(a,b)
    F_score = cal_F_score(a,b)
    ER = cal_Error_rate(a,b)
    print(F_score,ER)
