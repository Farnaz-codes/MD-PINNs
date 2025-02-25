# -*- coding: utf-8 -*-
"""
Editted on November 22nd 2023 -- > Added gold nanoshells + Tumor
                                   Updated the Geometry for my case
#------------------------------------------------------#
Editted on June 15: Small geometry: like 2006
Created on Fri May 20 14:24:14 2022
Symmetry Geometry of the system,
   Without tumor and Gold shell parts -  with 3 levels blood vessels 
  (according to Dr. Dai paper 2006):
  The Geometry is editted to follow the geometry is defined 2006 paper:
m_level = 3
m = 1 in z-direction
m = 2 in x- direction
m = 3 , 2 Vessels i z-direction

  Output: a dictionary with the keys: 
  Skin_1st', 'Skin_2nd', 'Skin_3rd', 'Arterial', 'Venous'
  Containing : 'x', 'y' ,'z' for each section
 
@author: fnazr
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

XY_tissue = 1.0    # It was 1 --> changed to 1.2 to satisfy the infinite BC conditions
#%%
def get_cube(X,Y,Z):   
    x1 = np.array([X[0],X[1],X[1],X[0],X[0]])
    x2 = np.array([X[1],X[0],X[0],X[1],X[1]])
    x = np.stack((x1,x1,x2,x2,x1))
    
    y1 = np.array([Y[0],Y[0],Y[1],Y[1],Y[0]])
    y2 = np.array([Y[1],Y[1],Y[0],Y[0],Y[1]])
    y = np.stack((y1,y1,y2,y2,y1))
    
    z1 = np.tile(Z[0],(1,5))
    z2 = np.tile(Z[1],(1,5))
    z = np.stack((z1[0,:],z2[0,:],z2[0,:],z1[0,:],z1[0,:]))
    
    return x,y,z
#%%

def Geometry(dY = [-0.1  ,0.06] ,dX= [0,0] ):
    # Separate the blood vessels
    # index  0 for artery and 1 for vein
    # dY = [0  ,38/100]     # offset of  [ arteries ,veins]
    # dZ= 150/1000          # offset
    Vessels = {}
    """
    Dimension of triple-layered skin
    """  
    LX = XY_tissue/2      # Skin width (cm)
    LY = XY_tissue        # Skin length (cm)
    #LZ = 1.3               # Skin depth  (cm)
    L1 = 0.008              # first layer (cm)
    L2 = 0.2 + L1           # Summation of the first two layers (cm)
    L3 = 1+L2#1.092 + L2         # Summation of all three layers  (cm) --> LZ is fixed to 1.3
    # Tumor Region
    Lt1 = 0.058 #0.208#0.24   # Location of top of the tumor1 (cm)
    Lt2 = 0.208 #0.208#0.24   # Location of top of the tumor2 (cm)
    LT1 = 0.15   # Length of tumor (cm) 
    LT2 = 0.05   # Length of tumor (cm) 
    LG = 0.08#0.1    # Golden nano shells part (cm) 
    Lg = Lt1 + 0.1 #0.258  # Location of top of the Golden nano shells part (cm)

    Vessels ['Skin_1st'] = {'x': np.array([0,LX]).flatten()[:,None].T,
                            'y' : np.array([0,LY]).flatten()[:,None].T, 
                            'z' : np.array([-L1,0]).flatten()[:,None].T}
    
    Vessels ['Skin_2nd'] = {'x': np.array([0,LX]).flatten()[:,None].T,
                            'y' : np.array([0,LY]).flatten()[:,None].T, 
                            'z' : np.array([-L2,-L1]).flatten()[:,None].T}
    
    Vessels ['Skin_3rd'] = {'x': np.array([0,LX]).flatten()[:,None].T,
                            'y' : np.array([0,LY]).flatten()[:,None].T, 
                            'z' : np.array([-L3,-L2]).flatten()[:,None].T}
    
    Vessels ['Skin_3rd_blood'] = {'x': np.array([0,LX]).flatten()[:,None].T,
                            'y' : np.array([0.23,0.68]).flatten()[:,None].T, 
                            'z' : np.array([-L3,-0.438]).flatten()[:,None].T}
    #-------------------------------------------------------------------------#
    # 2*LX/3 = 0.34 , LY/3 = 0.34 - 0.66
    Vessels ['Gold_Shell'] = {'x': np.array([0.4,LX]).flatten()[:,None].T,
                              'y' : np.array([0.4, 0.6]).flatten()[:,None].T, 
                              'z' : np.array([-(Lg+LG),-Lg]).flatten()[:,None].T}
    Vessels ['Tumor1'] = {'x': np.array([0.34,LX]).flatten()[:,None].T,
                         'y' : np.array([0.34,0.66]).flatten()[:,None].T, 
                         'z' : np.array([-(Lt1+LT1),-Lt1]).flatten()[:,None].T}
    
    Vessels ['Tumor2'] = {'x': np.array([0.34,LX]).flatten()[:,None].T,
                         'y' : np.array([0.34,0.66]).flatten()[:,None].T, 
                         'z' : np.array([-(Lt2+LT2),-Lt2]).flatten()[:,None].T}
    #-------------------------------------------------------------------------#
    """
    Vessel coordinates for system: 3 layers of blood vessels (2006)
     """
    NLb = np.zeros(3)
    NWb = np.zeros(3)
    Lb = np.zeros(3)
    
    NLb[0] = 0.1
    NWb[0] = 0.1
    Lb[0] = 0.4
    
    for i in range(1,3):
        NLb[i] = 2**(-1/3)*NLb[i-1]
        NWb[i] = 2**(-1/3)*NWb[i-1]
        Lb[i] = 2**(-1/2)*Lb[i-1]   
    """
    3 branche: Arterial system [0], & venous system [1]
    """
    for index in range(2):  # Arterial and venous systems
        if index == 0:
            V_name = 'Arterial'
        elif index==1:
            V_name = 'Venous'
            
        y_art0 = LY/2 + dY[index]
        x_art0 = LX + dX[index]
        z_art0 = L3

        # First level in z-direction
        X1 = x_art0 + np.array([[-NWb[0]/2,0]])
        Y1 = y_art0 + np.array([[-NLb[0]/2,NLb[0]/2]])
        Z1 = -z_art0+np.array([[0, Lb[0]]])
        
        # Second levelin x-direction
        X2_l = x_art0+np.array([[-Lb[1],0]])        
        # X2_r = x_art0+np.array([[0,Lb[1]]])
        Z2 = -z_art0 +Lb[0]+np.array([[0,NWb[1]]])
        Y2 = y_art0+np.array([[-NLb[1]/2,NLb[1]/2]])
        
        # Third level (2 branches) in z-direction
        Z3 = Z2[0,1] + np.array([[0,Lb[2]]])        
        X3 = X2_l[0,0]+np.array([[0,NWb[2]]])       
        Y3 = y_art0+np.array([[-NLb[2]/2,NLb[2]/2]])

        #--------------------------------------------------------------------#
        BloodVX = np.concatenate((X1,X2_l,X3))
        BloodVY = np.concatenate((Y1,Y2,Y3))
        BloodVZ = np.concatenate((Z1,Z2,Z3))
        BloodV = {'x' : BloodVX ,'y' : BloodVY, 'z' : BloodVZ}
        Vessels[V_name] = BloodV
        
    return Vessels
#----------------------------------------------------------------------------#
"""
Plot the 2D Skin :
""" 
def plot_skin_2D(V,  In_domain = {},xlabel = 'x', ylabel = 'z' ,alpha= 0.2,figsize=(4.2, 4.2),cc='k',ax=0):#,view1 =30,view2 = 50):
     if not(ax):    
         fig, ax = plt.subplots(figsize =figsize)
     my_cmap = plt.cm.jet([0,7,36,68,92,120,150,203,240],alpha = alpha)

     for iter_,key in enumerate(V.keys()):         
         X, Y = V[key][xlabel] ,V[key][ylabel]
         if ylabel =='z':
             if  X.shape[0]==3:
                 Y = np.array([[-Y[0,1],-Y[0,0]],[-Y[1,1],-Y[1,0]],[-Y[2,1],-Y[2,0]]])
             else: Y = np.array([[-Y[0,1],-Y[0,0]]])
             if xlabel =='x':
                 if X.shape[0]==3:
                     X = np.array([[X[0,0],X[0,0]+2*(X[0,1]-X[0,0])],
                                   [X[1,0],X[1,0]+2*(X[1,1]-X[1,0])],
                                   [X[2,0],X[2,1]],[X[2,0]+0.5,X[2,1]+0.5]])
                 else: X = np.array([[X[0,0],X[0,0]+2*(X[0,1]-X[0,0])]])    
         h=''
         lw=1
         if key =='Gold_Shell':
             Col ,alpha = 'y' , 0.8
             h = 'OOX'
         elif key =='Tumor':
             Col = my_cmap[iter_+2,:]
             h = 'X'
         elif key == 'Arterial':
             Col ,alpha = 'r' , 0.3
             alpha=0.3
         elif key =='Venous':
             Col , alpha ='b' ,0.3
         else :
             Col = my_cmap[iter_+2,:]
         if cc!='k':
             Col,lw ,h  = "None" ,0.28,''
         if (key=='Skin_1st' or key=='Skin_2nd' or key=='Skin_3rd') and ylabel=='z' and cc!='k':
                if key=='Skin_2nd' and 'Tumor' in In_domain.keys():
                    if In_domain['Tumor1'] or In_domain['Tumor2']:
                       ax.hlines(Y[0,1], X[0,0], 0.34,ls="--",color=cc,lw=lw)
                       ax.hlines(Y[0,1], 0.66, X[0,1],ls="--",color=cc,lw=lw)
                    else:
                       ax.hlines(Y[0,1], X[0,0], X[0,1],ls="--",color=cc,lw=lw)
                else:
                    ax.hlines(Y[0,1], X[0,0], X[0,1],ls="--",color=cc,lw=lw)
         else: 
             for i in range(Y.shape[0]):
                if In_domain: 
                   if key in In_domain.keys(): pp = In_domain[key]
                   elif (key == 'Arterial' and i in In_domain.keys()) or (key == 'Venous' and i+3 in In_domain.keys()):
                       pp =In_domain[i] if key == 'Arterial' else In_domain[i+3]
                   else: pp==False
                else: pp=True
                if pp:
                   W, L =  X[i,1]-X[i,0],Y[i,1]-Y[i,0] 
                   ax.add_patch(Rectangle((X[i,0], Y[i,0]),W, L,
                             ec=cc,fc = Col,ls="--",lw=lw,hatch = h))
                   if xlabel =='x' and i==2:
                       W, L =  X[3,1]-X[3,0],Y[i,1]-Y[i,0] 
                       ax.add_patch(Rectangle((X[3,0], Y[i,0]),W, L,
                             ec=cc,fc = Col,ls="--",lw=lw,hatch = h))
                   #plt.gca().add_patch(Rectangle((X[i,0], Y[i,0]),W, L,
                   #          ec=cc,fc = Col,ls="--",lw=lw,hatch = h))
         
         ax.set_xlabel(xlabel,fontsize = 12)
         ax.set_ylabel(ylabel,fontsize = 12)
             
     if xlabel =='x':ax.set_xlim(0,1)
     else: ax.set_xlim(0,V['Skin_3rd'][xlabel][0,1])
     Ylim_ = 1.208 if ylabel=='z' else 1 
     ax.set_ylim(0, Ylim_)#V['Skin_3rd'][ylabel][0,1])
     plt.savefig('Fig2D'+xlabel+'_'+ylabel+'.png', bbox_inches="tight",dpi=600)
 #---------------------------------------------------------------------------
"""
Plot the Skin :
""" 
def plot_skin(V, In_domain=[],alpha= 0.08,figsize=(14, 15)):#,view1 =30,view2 = 50):
     #V=Geometry()
     fig =plt.figure(figsize =figsize)
     ax = fig.add_subplot(111, projection='3d')
     my_cmap = plt.cm.jet([0,7,36,68,92,120,150,203,240],alpha = alpha)

     for iter,key in enumerate(V.keys()):
         
         X = V[key]['x']
         Y = V[key]['y']
         Z = V[key]['z']
         if key =='Gold_Shell':
             Col = 'y'
             alpha = 0.3
         elif key == 'Arterial':
             Col = 'r'
             alpha=0.3
         elif key =='Venous':
             Col ='b'
             alpha=0.3
         else :
             Col = my_cmap[iter+2,:]
             print( my_cmap[iter+2,:])
         a,b = X.shape

         for i in range(a):
             if In_domain: 
                if (key in In_domain) or (key in V.keys[:3]): pp =True
                elif (key == 'Arterial' and i in In_domain) or (key == 'Venous' and i+3 in In_domain):pp =True
                else: pp==False
             else: pp=True 
             if pp:
                 x,y,z = get_cube(X[i,:],Y[i,:],Z[i,:])
                 ax.plot_surface(x,y,-z,color = Col,alpha = alpha)
             # plt.pause(3)
         ax.set_xlabel('x',fontsize = 25)
         ax.set_ylabel('y',fontsize = 25)
         ax.set_zlabel('z',fontsize = 25)
             
     plt.rc('xtick',labelsize =18 )
     plt.rc('ytick',labelsize = 25)
     plt.rc('axes',titlesize =15)       
     #x.set_xlim(0,V['Skin_3rd']['x'][0,1])
     plt.axis('off')

     ax.set_ylim(0,V['Skin_3rd']['y'][0,1])
     #ax.set_zlim(V['Skin_3rd']['z'][0,0],0)
     #ax.view_init(view1,view2)

     #plt.show()
     plt.savefig('Fig3.png', bbox_inches="tight",dpi=600)
     #%%
Vessels = Geometry()
#plot_skin(Vessels)
plot_skin_2D(Vessels, xlabel = 'x', ylabel = 'z')
plot_skin_2D(Vessels, xlabel = 'y', ylabel = 'z')
#plot_skin_2D(Vessels, xlabel = 'x', ylabel = 'y')


#lb,ub= Mid_geometry_partition_chi(Is_blood=1)