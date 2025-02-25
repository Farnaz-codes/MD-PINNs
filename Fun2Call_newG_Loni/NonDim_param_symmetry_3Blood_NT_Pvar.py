# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 14:10:28 2021
Edited on Thurseday Sep 09

@author: fnazr
Non_dimensional coefficients and scales
Converting dimensional T, x,y,z,t to nondimensional and inverse

Checked on April 2023
Updated to new on Friday Dec 8th 2023
"""
import numpy as np
import tensorflow as tf
from Fun2Call_newG_Loni import parameters_symmetry_newG as param

P_range = [1.15,1.4]
dP = P_range[1] -P_range[0]
Pmin = P_range[0]
Dtype = np.float32
#%%
class NonDim_param:

    def __init__(self , Skin_layer = 'Skin_1st',tn = 1, dt_tissue = 8, dt_blood = 2, remove_d =[]):

        self.d3_n = remove_d  # dimension I wantto remove e.g remove p and t [4,3]
        self.Skin_l = Skin_layer
        self.tn = tn
        if Skin_layer in ['Arterial', 'Venous']:
            self.Skin_l='Skin_3rd'
            
        self.param = param.param(self.Skin_l)
        self.cx , self.cy,self.cz = self.param.cxyz()
        self.dt_tissue,self.dt_blood = dt_tissue, dt_blood
       
        self.Bi1 = self.param.h_inf*self.Char_Len()[2]/self.param.kl()
        self.T_alpha = self.param.kl()/(self.param.rhol()*self.param.Cpl())
        
        self.A = self.ts()*self.param.Cblood()*self.param.Wbl()/(self.param.rhol()*self.param.Cpl())
        
        self.F0 = tuple(self.T_alpha*self.ts()/(L**2) for L in self.Char_Len())
        self.Rt = self.param.taul()/self.ts()
    
    def Char_Len(self):
        # if X = (x-Lx)/Lsx, 
        # Y = (Y-Ly)/Lsy, 
        # Z= (z-Lz)/Lsz with X, Y , and Z dimensionless parameters, 
        # and Ls = Li being charactersitic lenght
        Char_Len_x = self.param.Len_t2()[3]-self.param.Len_t2()[2]
        Char_Len_y = self.param.Len_t2()[5]-self.param.Len_t2()[4]
        Char_Len_z = self.param.Len_t2()[1]-self.param.Len_t2()[0]
                          
        return Char_Len_x/self.cx, Char_Len_y/self.cy, Char_Len_z/self.cz
    
    def Char_T(self,blood=False):
        # if T = (T-Tr)/(Tr-Ts) be a dimensionless temperatue then for skin layers
        # the Tr and Ts are
        Tr = self.param.T0
        Ts = Tr - self.dt_tissue #self.param.T_inf
        if blood:
            Ts = Tr - self.dt_blood
        return Tr, Ts
    
    def ts(self): 
        return self.tn
    
    def ND_T(self,T, blood = False):
        # if T = (T-Tr)/(Tr-Ts) be a dimensionless temperatue then for skin layers
        # The tetha is
        Tr, Ts = self.Char_T(blood = blood)
        tetha = (T-Tr)/(Tr-Ts)
        return tetha
    
    def Q_Coeff(self):
        # if Qhat = Q*Qs     be a dimensionless Q 
        # then Qs is 
        Tr, Ts = self.Char_T()
        Qs = self.param.rhol()*self.param.Cpl()/self.ts()*(Tr-Ts)
        return Qs
    #------------------------------------------------#
    # determine the format and size of variable
    # input to the Forward or inverse object
    def a_cond(self,Var):
        if isinstance(Var,(float,int)):a = 1
        elif isinstance(Var,dict): a=10
        elif tf.is_tensor(Var): a = Var.shape[1]
        elif Var.shape[0] == Var.size: a = 1 
        else: a = Var.shape[-1]
        return a
    #------------------------------------------------#
    def Forward_Var(self,Var, d='T', t0=0, blood = False):
        # convert to nondmensional variables
        a = self.a_cond(Var)
        Lsx, Lsy, Lsz = self.Char_Len()
        Tr, Ts = self.Char_T(blood = blood)
        Lz, _, Lx, _ , Ly, _ = self.param.Len_t2()
        coeffx = [Lsx, Lsy, Lsz, self.ts(), dP]
        coeff0 = [Lx, Ly, Lz, t0, Pmin]
        for i in self.d3_n: 
                 coeffx.pop(i)
                 coeff0.pop(i)

        if (a>1 and a<10):
            if tf.is_tensor(Var):
                Coeff_X = tf.constant([coeffx],dtype = tf.float32)
                Coeff_0 = tf.constant([coeff0],dtype = tf.float32)
                Var = (Var - Coeff_0)/ Coeff_X
            else:
                Coeff_X = np.array([coeffx])
                Coeff_0 = np.array([coeff0])
                Var = (Var - Coeff_0)/ Coeff_X
                Var = Var.astype(Dtype)
        #--------------------------------------------------------------------#        
        elif a==1:
            if d=='T': Var = (Var-Tr)/(Tr-Ts)
            elif d == 'z': Var =(Var-Lz)/Lsz
            elif d=='time': Var = (Var-t0)/self.ts()
            elif d=='x': Var =(Var-Lx)/Lsx
            elif d=='y': Var =(Var-Ly)/Lsy
                
        elif a==10:
            Var ={0:(Var[0]-Lx)/Lsx,1:(Var[1]-Ly)/Lsy,2:(Var[2]-Lz)/Lsz,
                  3:(Var[3]-t0)/self.ts()}
                    
        return Var
    
    def Inverse_Var(self,Var,d = 'T', t0 = 0, blood=False):
        # return to our real geometry
        Lsx,Lsy,Lsz = self.Char_Len()
        Tr, Ts = self.Char_T(blood=blood)
        Lz,_,Lx,_,Ly,_ = self.param.Len_t2()
        a = self.a_cond(Var)
        coeffx = [Lsx, Lsy, Lsz, self.ts(), dP]
        coeff0 = [Lx, Ly, Lz, t0, Pmin]
        for i in self.d3_n: 
                 coeffx.pop(i)
                 coeff0.pop(i)
        if (a>1 and a<10) :
           if tf.is_tensor(Var):
               Coeff_X = tf.constant([coeffx],dtype = tf.float32)
               Coeff_0 = tf.constant([coeff0],dtype = tf.float32)
               Var = Var* Coeff_X + Coeff_0
           else:
               Coeff_X = np.array([coeffx])
               Coeff_0 = np.array([coeff0])
               Var = Var* Coeff_X + Coeff_0
               Var = Var.astype(Dtype)
        #--------------------------------------------------------------------#        
        elif a==1:
            if d =='T':  Var = Var*(Tr-Ts)+Tr
            elif d =='z': Var =Var*Lsz + Lz
            elif d=='time': Var = Var*self.ts()+t0
            elif d=='x': Var = Var*Lsx+Lx 
            elif d=='y': Var = Var*Lsy+Ly
        elif a==10:
            Var ={0:Var[0]*Lsx+Lx,1:Var[1]*Lsy+Ly,2:Var[2]*Lsz+Lz,3:Var[3]*self.ts()+t0}

        return Var
    #------------------------------------------------#
    def L0_blood_basedOn3rd(self, cell_num):
        Label = ['z','x','z','z','x','z']
        Lz1, Lz2, Lx1, Lx2, _, _ = self.param.Len_t2()
        # L0 = [L_entry,L_out]
        if Label[cell_num]=='z': L0 = [Lz1,(Lz2-Lz1)/self.cz]
        else:                    L0 = [Lx1,(Lx2-Lx1)/self.cx]
        return L0   
    #-------------------------------------------------------------------------#
    def Bi_blood_wall2(self, endx=False):
        Lsx, Lsy, Lsz = self.Char_Len()
        Bi = 95.23

        Dim_wall =((0,1,1),(1,1,2,2,0),(0,0,1,1))
        wall_label = ((Lsx,Lsy,-Lsy),(Lsy,-Lsy,Lsz,-Lsz,Lsx),(Lsx,-Lsx,Lsy,-Lsy))
        Band =()
        if endx:
            Dim_wall=((0,1,1,2),(1,1,2,2,0),(0,0,1,1))
            wall_label= ((Lsx,Lsy,-Lsy,Lsz),(Lsy,-Lsy,Lsz,-Lsz,Lsx),(Lsx,-Lsx,Lsy,-Lsy))
            Band = (('lb','lb','ub','lb'),('lb','ub','lb','ub','lb'),('lb','ub','lb','ub'))

        Dim_wall +=Dim_wall
        Band +=Band

        wall_label +=wall_label   
        return Dim_wall , tuple(tuple(Bi*wall_label[i][j] for j in range(len(wall_label[i])))
                               for i in range(len(wall_label))), Band
    #-------------------------------------------------------------------------#
    #Blood vessels
    def Forward_Var_blood(self,Var,L0,t0=0,d='L'):
        # convert data in blood vessels to nondmensional variables
        # We just have 2D variables [x(or y or z), t]
        # Temperature is similar to the other tissues, so:
        #     ND.Forward_Var(Var,d='T')
        # L0 =[Lentry, Lout]
        Coeff_X = np.array([L0[1],self.ts()])
        
        a = self.a_cond(Var)
   
        if (a>1 and a<10):
            Var = (Var-np.array([L0[0],t0]))/ Coeff_X.flatten()[None,:]
        elif a==1:
            if d == 'L':
                Var =(Var-L0[0])/Coeff_X[0]
            elif d=='time':
                Var = (Var-t0)/self.ts()
        elif a==10:
            Var ={0:(Var[0]-L0[0])/Coeff_X[0],1:(Var[1]-t0)/self.ts()}              
        return Var
