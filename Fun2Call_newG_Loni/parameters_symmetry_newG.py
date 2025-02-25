# -*- coding: utf-8 -*-
"""
Updated on December 2023 --> Last version

Created on Mon May 21 17:10:28 2022
Symmetry update
@author: fnazr
Physical properties
Parameters of the problem ( based 0n paper 2006, tables)
"""
import numpy as np

class param:
    T0 = 34 + 273.15        # deg.C  34 +
    h_inf = 0.001           # W/cm^2

    def __init__(self , Skin_layer = 'Skin_1st', type_ = []):
        # type_ :  if "Same", then I will use the same properties for all layer which
        # is the property of the second layer
        # it will be used for testinf the code
        # otherwise type_=[]
        self.Skin_l = Skin_layer
        KEYS = ['Skin_1st','Skin_2nd','Skin_3rd','Tumor','Gold_Shell']
        
        if Skin_layer=='Skin_3rd_blood': self.Skin_l='Skin_3rd'
        if Skin_layer=='Tumor1' or Skin_layer=='Tumor2': self.Skin_l='Tumor'
            
        self.index = KEYS.index(self.Skin_l)
        self.index_L=5 if Skin_layer=='Tumor2' else 6 if Skin_layer=='Skin_3rd_blood' else self.index
        
        if type_=='Same': self.index = 1    
        
    def kl(self): # Conduction coefficient (W/cm.deg. C)
        k = [0.0026,0.0052,0.0021,0.00642,0.00642]  # W/cm. deg.C        
        return k[self.index]
        
    def taul(self): # Time delay (Second)
    # The time delay is assumed to be zero in 2006 paper
        tau=[20,20,20,6.825,6.825]                 # second
        return tau[self.index]
        
    def rhol(self):#  density
        rho =[1.2,1.2,1,1,1]            # gr/(cm^3 )
        return rho[self.index]
            
    def Cpl(self): # heat capacity 
        Cp = [3.6,3.4,3.06,3.75,3.75]    # J/(gr * deg.C)
        return Cp[self.index]
        
    def Wbl(self):    
        Wb = [0, 0.0005, 0.0005, 0.0005, 0.0005]    # gr/(cm^3*second)
        return Wb[self.index]
        
    def Cblood(self): 
        Cbl = [4.2, 4.2, 4.2, 4.2, 4.2]     # J/(g*deg.C)
        return Cbl[self.index]   
    
    def cxyz(self):
        cx = [2, 2, 2, 1, 1]
        cy = [4, 4, 4, 2, 2]
        cz = [1 ,1 ,1, 1, 1]
        if self.index_L==6: cy[2] = 2
        return cx[self.index] ,cy[self.index], cz[self.index]
    #%%-----------------------------------------------------------------------#
    def Len_(self):
       # LX = 1                       # Skin width (cm)
       LX = 1.00/2                    # Skin width (cm)
       Ly = 1.00
       L1 = [0   , 0.008, 0.208, 0.108, 0.158]
       L2 =[0.008, 0.208, 1.208, 0.308, 0.258]
       Lx1 = [0, 0, 0, 0.34, 0.4]
       Lx2 = [LX, LX, LX , LX, LX]
       Ly1 = [0, 0, 0, 0.34, 0.4]
       Ly2 = [Ly, Ly, Ly, 0.66, 0.6]
       return L1[self.index],L2[self.index],Lx1[self.index],\
                 Lx2[self.index],Ly1[self.index],Ly2[self.index]
    #-------------------------------------------------------------------------#
    def Len_t2(self): # nonddimensionalize parameters
       # LX = 1                       # Skin width (cm)
       LX = 1.00/2                    # Skin width (cm)
       Ly = 1.00
       L1 = [0   , 0.008, 0.208, 0.058,0.058,0.058, 0.438 ]#0.158, 0.208, 0.438]
       L2 =[0.008, 0.208, 1.208, 0.258,0.258,0.258, 1.208 ]#0.258, 0.308, 1.208]
       Lx1 = [0,   0,  0, 0.34, 0.34, 0.34, 0 ]
       Lx2 = [LX, LX, LX, LX,    LX,  LX ,  LX]
       Ly1 = [0,  0,   0, 0.34, 0.34, 0.34, 0.23]
       Ly2 = [Ly, Ly, Ly, 0.66, 0.66, 0.66, 0.68]
       #Lz = L1
       return L1[self.index_L],L2[self.index_L],Lx1[self.index_L],\
                 Lx2[self.index_L],Ly1[self.index_L],Ly2[self.index_L]
    #-------------------------------------------------------------------------#
    # Blood vessels properties
    def factor_blood(self,cell_num,m_level=3):
        # m is a number between [0,6]
        if m_level ==3:
            m, sign_ = self.cell_num_3levels(cell_num)
        # alpha: Heat transfer Coefficient between blood and tissue
        alpha = 0.2  # W/cm^2.C
        # CB: Heat capacity of blood
        CB = 4.134   # J/cm^3.C
        # vm: Velocity of blood flow 
        NLb,NWb,Lb,vm =self.Vessel_dim()
        # Pm: Vessel perimeter (cm)
        Pm = 2*(NWb[m]+NLb[m])
        if m==0:
            Pm=2*NWb[m]+NLb[m]
        # Fm: Area of Cross section
        Fm = NWb[m]*NLb[m]
        # Mm : Mass flow of blood vessels
        # Mm = vm[m]*Fm
        factorm = alpha*Pm/(CB*vm[m]*Fm) #* Lb[m]       
        # pdot: decreased blood flow rate
        Pdot = 0.5e-3 # 1/s
        g_m = Pdot/vm[m]#*Lb[m]
        return sign_*factorm, sign_*g_m          
    """
     Vessel coordinates for system
     7 layers of blood vessels
    """
    def Vessel_dim(self):
        NLb,NWb,Lb,vm=np.zeros(3),np.zeros(3),np.zeros(3),np.zeros(3)     
        NLb[0], NWb[0], Lb[0], vm[0] = 0.1, 0.1, 0.4, 8  # cm , cm, cm, cm/s
        M1 = vm[0]*NWb[0]*NLb[0]           # correct M1: not symmetry
        
        for i in range(1,3):
            NLb[i] = 2**(-1/3)*NLb[i-1]
            NWb[i] = 2**(-1/3)*NWb[i-1]
            Lb[i] = 2**(-1/2)*Lb[i-1]
            vm[i] = M1/(2*NWb[i]*NLb[i])
        NWb[0] /=2    
        return NLb,NWb,Lb,vm
    #-------------------------------------------------------------------------#
    def cell_num_3levels(self, cell_num): # in symmetry
        # cell_num : between 0-9 : 0-4 for Arteries | 5-9 for veins
        m_level_i = 0 if cell_num in [0,3] else 1 if cell_num in [1,4] else 2
        sign_ = 1 if cell_num in [3,4,5] else -1
        return m_level_i, sign_
    #-------------------------------------------------------------------------#
#p1 = param(Skin_layer = 'Skin_1st')
#p2 = param(Skin_layer = 'Skin_2nd')
#p3 = param(Skin_layer = 'Skin_3rd')
# pa = param(Skin_layer = 'Arterial')
# pg = param(Skin_layer = 'Gold_Shell')
# pt = param(Skin_layer = 'Tumor')