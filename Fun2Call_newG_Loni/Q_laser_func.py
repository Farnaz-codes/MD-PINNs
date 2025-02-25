# -*- coding: utf-8 -*-
"""
Created on Mon Feb 29 02:19:02 2024
@author: fnazr
GNR of R = 4.6, lambda = 863 nm, reff = 11.43 nm, fv = 1e-5
"""
import numpy as np
import tensorflow as tf

a = 10  # fv = 1-6*a and 863 nm (GNR (reff = 11.43 nm, R = 4.6))
tetha = 0
miu = np.cos(tetha)

class Q_laser:
   def __init__(self, Is_tumor = True):#, Is_blood =False):
       
       self.Is_tumor = Is_tumor
       self.KEYS = ['Skin_1st','Skin_2nd','Skin_3rd']
       if Is_tumor: self.KEYS = ['Skin_1st','Skin_2nd','Tumor1','Gold_Shell','Tumor2','Skin_3rd']
       
       self.Btot, self.W_landa, self.Ubz = () ,() ,[]
       for i, Skin_layer in enumerate(self.KEYS):
           self.Btot += (self.absorption(i) + self.scattering(i),)
           self.W_landa += (self.scattering(i)/self.Btot[i],)
           self.Ubz.append(self.Char_Len(i))

       self.C_star, self.Sai = self.sai_estimate()
   #--------------------------------------------------------------------------#
   # Related physical properties for our current Geometry
   def Char_Len(self,index):  
       if self.Is_tumor: Lz = [0.008, 0.05, 0.1, 0.08, 0.02, 0.95]
       else: Lz = [0.008, 0.20, 1.0]
       return Lz[index]

   def absorption(self,index):
        ag = 10.0387*a # 10.24 *a  # fv =1-6*a and 863 nm (GNR (reff = 11.43nm, R =4.6))
        if self.Is_tumor: 
            alpha = [1.037, 0.921, 0.72, 0.72+ag, 0.72, 0.94, 0.94] # 863 nm (wavelength)
        else: 
            alpha = [1.037, 0.921, 0.94, 0.94]
        return alpha[index]
    
   def scattering(self,index) :
        sg = 1.0205*a   # fv =1-6*a and 863 nm ( GNR (reff = 11.43nm, R =4.6))      
        if self.Is_tumor: 
            S = [33.52, 20.92, 18.39, 18.39+sg,18.39,19.24,19.24]  # 863 nm
        else:  
            S = [33.52, 20.92,19.24]
        return S[index]  
   #--------------------------------------------------------------------------#     
   def C_star_approx(self, Sai_):
      K_num = len(self.KEYS)
      A = np.zeros((K_num*2,K_num*2))
      B = np.zeros((K_num*2,1))
      
      for li in range(K_num):   
        Btot, w_landa, UBz = self.Btot[li] ,self.W_landa[li] ,self.Ubz[li]
        sai = Sai_[li]
        tau_L_landa = Btot*UBz
        ksi = (3*(1-w_landa))**0.5
        eta = -3*w_landa/(1/miu**2-ksi**2)*sai  #B1 = eta*sai * Iin
        Exp , nExp = np.exp(tau_L_landa*ksi), np.exp(-tau_L_landa*ksi)
        miuExp = np.exp(-tau_L_landa/miu)
  
        if li==0:  #BC :  a grad(Gd) = b G - const
            a, b , const=2/3,1, 0#4*np.pi*sai
            A[li,li] , A[li,li+K_num] = b-a*ksi,b+a*ksi
            B[li] = -eta *(b+a/miu)+ const
            
        if li==K_num-1: B[-1] = eta/miu*miuExp            
        A[2*li+1, li],A[2*li+1, li+K_num] = ksi*Exp, -ksi*nExp   # constant dG 1stand 2nd col (C1,C2)
        if li<K_num-1:  # L condition
            A[2*li+2, li],A[2*li+2, li+K_num] = Exp, nExp   # constant G 1stand 2nd col (C1,C2)
            B[2*li+1] += eta/miu*miuExp
            B[2*li+2] -= eta*miuExp 
            
        if li>0:   # 0 condition
            A[2*li-1, li], A[2*li-1, li+K_num] = -ksi,ksi  # constant dG  (C1 and C2)
            A[2*li, li], A[2*li, li+K_num] = -1,-1         # constant G  (C1 and C2)
            B[2*li-1] -=eta/miu
            B[2*li] += eta
           
      C_star = np.matmul(np.linalg.inv(A),B)
       
      return np.reshape(C_star,(2,K_num))
   #--------------------------------------------------------------------#
   def sai_estimate(self):
        sai = [1]
        for li in range(len(self.KEYS)-1):
           Btot, UBz = self.Btot[li] ,self.Ubz[li]
           tau_L_landa = Btot*UBz
           sai.append(np.exp(-tau_L_landa/miu)*sai[li])  #qc
               
        C_star= self.C_star_approx(sai) 
        return C_star, sai  
   #--------------------------------------------------------------------# 
   def Qz(self,z, slayer):   
       li = self.KEYS.index(slayer)
       Gc,Gd = self.Normalized_Q_per_layers(z, li)
       Qz = (Gc+Gd)*(1-self.W_landa[li])*self.Btot[li] # NQ = Gc+Gd      
       return Qz        
   #------------------------------------------------------------------------#
   def Normalized_Q_per_layers(self, z ,li):
        # ['Skin_1st','Skin_2nd','Tumor1','Gold_Shell','Tumor2','Skin_3rd']
        # z is a nondimensionalized array
        ksi = tf.sqrt(3*(1-self.W_landa[li]))
        eta = -3*self.W_landa[li]/(1/miu**2-ksi**2)*self.Sai[li]
        if not(tf.is_tensor(z)): z = tf.convert_to_tensor(z,tf.float32)
        if self.Is_tumor:
            if li==1 or li==2: z = z*0.2  # Skin_2nd or Tumor1 
            elif li==4: z = z*0.2 - 0.18  # Tumor2
            elif li==5: z = z*1 -0.05   # Skin_3rd
            elif li==3: z = z*0.2 -0.1   # Gold_Shell
            else: z = z * self.Ubz[li]   # Skin_1st
        else:
            z = z*self.Ubz[li]

        tau_z_landa = self.Btot[li] *z  # z starts from 0 -->[z-z0,z-z0+Lz] in cm
        #tau_in_landa =  0.00 #Btot[ii]*LBz[ii]
        #--------------------------------------------------------------------#
        # normalized by Btot*Iin
        # negative of derivative of collimated irradiance wrt (z)
        Gc = tf.exp(-tau_z_landa/miu)*self.Sai[li]
        # negative of of derivative of diffusive irradiance wrt z
        Gd = self.C_star[0,li]*tf.exp(ksi*tau_z_landa)+\
            self.C_star[1,li]*tf.exp(-ksi*tau_z_landa)+eta*tf.exp(-tau_z_landa/miu)
        return Gc,Gd  #*Gaussian_*Iin_scattering_Norm*I_in*Btot
   #--------------------------------------------------------------------------#

           
           
           