# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 02:39:48 2023

@author: fnazr
"""
import tensorflow as tf
import numpy as np
from pyDOE import lhs

DTYPE='float32'
m_level = 3
m_dir = [2,0,2,2,0,2]
#-----------------------------------------------------------------------------#
def random_collection_np(lb, ub, N =[], Bound = False, C_band=[], Dir = []):
    # Dir : lb | ub | IC : lb: z = z0 | ub:z = zl | IC :initial condition
    # Bound =   1: 'x' ; 2: 'y' ; 3: 'z' ; 4: 't' ; List
    # Bound = False | 0: all directions (used for X_f)    
    # lb : is lb[key] = [x0,y0,z0,t0]  # ub: ub [key] = [xl,yl,zl,t_end]
    # N: Number of random selection points
    # X is a N x lb.shape[1] with dict lb.shape[0]
    VarList = list(range(lb.shape[1]))
    b = Bound    
    if b: VarList.remove(b-1)
    LB = lb[0,VarList]
    UB = ub[0,VarList]
    temp = LB + (UB-LB)*lhs(LB.size,N)
    #temp=  LB + (UB-LB)*np.random.uniform(size=(N,LB.size))
    #-----------------------------------------------------------------------#
    # Check if inside the constraint band
    # C_band = {'lb' : lb_c_band ,'ub' : ub_c_band}
    if C_band:
        Nt_new = 10
        while Nt_new > 0:
            for key_band in C_band.keys():
                temp = remove_constraint_points_np(temp,C_band[key_band],VarList)  #c3
            Nt_new = N - temp.shape[0]   
            if Nt_new>0:
                Nt_new=Nt_new*(temp.shape[1])*50
            if Nt_new<=0:
                temp=temp[:N,:]
                break
            temp = np.concatenate((temp,LB + (UB-LB)*lhs(LB.size,Nt_new)),axis =0)
            #temp=np.concatenate((temp,LB +(UB-LB)*\
            #                    np.random.uniform(size=(Nt_new,LB.size))),axis =0) #c4
    #-------------------------------------------------------------------------#
    if Dir: # Add boundary and initial conditions
        if Dir == 'ub': b_val =  ub[0,b-1] 
        elif Dir == 'lb': b_val =  lb[0,b-1]
        return  np.insert(temp, b-1, b_val,axis =1)  #C5
        #return tf.concat((temp[:,:b-1], b_val* tf.ones((temp.shape[0],1),dtype=DTYPE), temp[:,b-1:]), axis=1)         
    else: return temp  
#%%----------------------------------------------------------------------------#
def remove_constraint_points_np(X, C_bandk, VarList =[]):
    # Check if random points are inside the constraint band, if so remove them and return
    # C_band = {'lb' : lb_c_band ,'ub' : ub_c_band}
    if len(VarList)==0: VarList = list(range(C_bandk['lb'].shape[1]))
        
    a = C_bandk['lb'].shape[0]
    ii1 = X.shape[1]
    for i in range(a):
        # points inside the constraints
        lim1 = X >= C_bandk['lb'][i,VarList]
        lim2 = X <= C_bandk['ub'][i,VarList]
        S = np.where((np.sum(lim1[:,:ii1],axis =1) +np.sum(lim2[:,:ii1],axis =1))==2*ii1) 
        if S[0].size>0: X = np.delete(X,S[0],axis =0)       
    return X
#%%----------------------------------------------------------------------------#
def new_prop_Wallpoints_tf(lb, ub, N, D_shape):
    a = len(m_dir)  # 6 Blood vessels
    Xwall_new = ()
    n = lb.shape[1]
    for i in range(a):
        Dd = list(range(3))
        Dd.remove(m_dir[i])
        dim_= [max(m_dir[i]-1,0),'lb'] 
        dim_[1] ='ub' if i<3 else 'lb'#key=='Arterial' else 'lb'
        Nw = [N[i][j] for j in range(len(N[i]))]
        if i==0 or i==3: Nw.insert(1,Nw[0])
        
        xw1 = random_collection_np(lb=lb[i:i+1,:], ub=ub[i:i+1,:], N=Nw[0], Bound=Dd[0]+1, Dir='lb')
        xw2 = random_collection_np(lb=lb[i:i+1,:], ub=ub[i:i+1,:], N=Nw[1], Bound=Dd[0]+1, Dir='ub')  
        if i==1 or i==4:
            LBw, UBw = lb[i:i+1,:].copy(), ub[i:i+1,:].copy()
            UBw[0,0] = lb[i-1,0]
            #UBw = tf.concat((lb[i-1:i,0:1], ub[i:i+1,1:]),axis=1)
            xw4 = random_collection_np(lb=LBw, ub=UBw, N=Nw[3], Bound=Dd[1]+1, Dir='ub')
            #xw4 = random_collection_tf(lb[i:i+1,:], UBw, N=Nw[3], Bound=Dd[1]+1, Dir='ub')
            
            c_bandz1 = {'Arterial': {'lb':lb[i+1:i+2,:],'ub':ub[i+1:i+2,:]}
                        ,'Venus':{'lb':lb[i+1:i+2,:],'ub':ub[i+1:i+2,:]}}
            xw3 = random_collection_np(lb[i:i+1,:], ub[i:i+1,:], Nw[2], Dd[1]+1, c_bandz1,'lb')
        else:
            xw3 = random_collection_np(lb=lb[i:i+1,:], ub=ub[i:i+1,:], N=Nw[2], Bound=Dd[1]+1, Dir='lb')
            xw4 = random_collection_np(lb=lb[i:i+1,:], ub=ub[i:i+1,:], N=Nw[3], Bound=Dd[1]+1, Dir='ub')
            
        if i==0 or i==3:XX =(xw1,xw3,xw4)
        else: XX = (xw1,xw2,xw3,xw4)
        #---------------------------------------------------------------------#     
        if i==0 or i==3:
            c_bandz={'Arterial': {'lb':lb[i+1:i+2],'ub':ub[i+1:i+2]}
                     ,'Venus':{'lb':lb[i+1:i+2],'ub':ub[i+1:i+2]}}
            XX +=(random_collection_np(lb[i:i+1], ub[i:i+1], N = Nw[4], Bound =m_dir[i]+1,
                                          C_band=c_bandz, Dir='lb'),)        
        elif i==1 or i==4:
            XX += (random_collection_np(lb[i:i+1], ub[i:i+1], Nw[4], Bound=m_dir[i]+1, Dir='lb'),)
            
        XX = tuple(tf.reshape(i,(D_shape, -1,n)) for i in XX)
        #---------------------------------------------------------------------#
        Xwall_new += (tuple(tf.cast(xk,dtype = DTYPE) for xk in XX),) 
        
    return Xwall_new
#%%----------------------------------------------------------------------------#
def new_prop_points_tf(LB, UB, Nf, C_band_keys, Bound = False, Dir = 'lb',
                       Key_order=['Skin_1st','Skin_2nd','Skin_3rd']):
    x_new = ()
    for ii,keys in enumerate(Key_order):
        if ii<len(Nf):
            #---------------Edit C_band for different BC ---------------------#
            if len(C_band_keys[keys])==0:C_band = C_band_keys[keys]
            elif Bound==2 or (Bound==1 and Dir =='lb'):C_band = []
            elif isinstance(C_band_keys[keys],dict):
                C_band = {k1: {k2: C_band_keys[keys][k1][k2].copy() for k2 in C_band_keys[keys][k1].keys()}
                           for k1 in C_band_keys[keys].keys()}
                if Bound==3: # lower bound of Z
                    if 'Tumor2' in C_band.keys() and Dir == 'lb':
                        C_band = {'Tumor2': C_band['Tumor2']}
                        
                    elif  keys=='Tumor2' and  Dir == 'lb':
                        C_band = {'Gold_Shell': C_band['Gold_Shell']}      
                    elif 'Arterial' in C_band.keys() and Dir == 'ub':
                        C_band = {'Arterial': {'lb':C_band['Arterial']['lb'][0:1,:],
                                                'ub':C_band['Arterial']['ub'][0:1,:]}
                                    ,'Venous':{'lb':C_band['Venous']['lb'][0:1,:],
                                               'ub':C_band['Venous']['ub'][0:1,:]}}
                    else: C_band = []   
                        
                if Bound==1 and Dir =='ub' and 'Arterial' in C_band.keys():
                    C_band ['Arterial']= {'lb':C_band['Arterial']['lb'][0:2,:],
                                          'ub':C_band['Arterial']['ub'][0:2,:]}
                    C_band ['Venous'] = {'lb':C_band['Venous']['lb'][0:2,:],
                                         'ub':C_band['Venous']['ub'][0:2,:]} 
                #-------------------------------------------------------------#
            if Nf[ii]>0:    
                x_new += (random_collection_np(LB[keys], UB[keys], Nf[ii], Bound=Bound,\
                           C_band = C_band).astype(DTYPE),) 
            else:x_new +=((),)
                
    x_new_tf = tuple(tf.convert_to_tensor(jj) for jj in x_new)
    return x_new_tf
#-----------------------------------------------------------------------------#    

