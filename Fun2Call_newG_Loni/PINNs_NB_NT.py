# -*- coding: utf-8 -*-
"""
Skin-3rd ---> tanh
Others LAAF

December 11th
Loss of upperband boundary Conditions of skin_3rd is removed
December 5th
1) net_skinMerged is updated and related functions
2)Loss of wall is separted
Edits on November 18th:
    Horovod is applied to parallel the data
Edits on October 25th, 2023:
Edits and 80% completely changes on Feb, 2024
@author: fnazr
"""
import os
import tensorflow as tf
from tensorflow import keras as keras
import numpy as np
import time
import pickle as pkl
from Fun2Call_newG_Loni import NonDim_param_symmetry_3Blood_NT_Pvar as NDparam
from Fun2Call_newG_Loni import PINN_model
from Fun2Call_newG_Loni import MCMH_data_selection_horovod_wall_tumor_DOE as mcmh_fun
from Fun2Call_newG_Loni import MyLRSchedule_Cycle2
#-----------------------------------------------------------------------------#
# Initialize Horovod --> Step2 " 3- Configure CPU-GPU Mapping
import horovod.tensorflow as hvd    # Step 1
hvd.init()
# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: # Memory growth must be set before GPUs have been initialized
   tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
print(len(gpus), "Physical GPUs with hvd size: ",hvd.size(), " & hvd rank: ", hvd.rank())
#------------------------------------------------------------------------#
P_variable = True
SGD = 0
DTYPE='float32'
DTYPE_np = np.float32
keras.backend.set_floatx('float32')
P_range = [1.15, 1.4]  
dP = P_range[1] -P_range[0]
Pmin = P_range[0]  # W/cm
er_est = 1
#-----------------------------------------------------------------------------#
class PINNs_tf2_Power():
  # Initialize the class
  # 1 - input_args = {'args_tissue':args_tissue,'args_blood': args_blood}
  #     args_tissue = dict(['X0', 'Xf', 'X_lb', 'X_ub', 'Y_lb_ub', 'Z_lb', 'Z_ub'])
  # 2 - Tissue name: ['Skin_1st','Skin_2nd','Skin_3rd']
  # 3 - layers : {'Skin': [5, 70, 70, 70, 70, 70, 1]}
  # 4,5 - lb, ub: dict containing non-dim lb and ub of size 5 [x, y, z, t, P]
  # 10,11 - path_ , path_T0 :The pathes the models were saved and should be loaded now 
  # remove_d =[4] for removing p and [4,3] for ignoring time and p
  def __init__(self, input_args, Tissues_name, layers, lb, ub, blood_band,
               Is_segments = (False,False), new_case = False, load_w_data = (False,False),
               path_ = [], path_T0 = [], ntime = 1, T0 = [], minpoints = 504, C_band = [],
               it_Stop=100000 ,lr_cycle_start = 0, P0 = 1.345, act_func = 'tanh', 
               opt='adam', data_path_err =[],path_1 =[],tn = 0):
    # 1) inputs
    args_tissue = input_args['args_tissue']
    self.act_func =  act_func
    self.lr_cycle_start = lr_cycle_start
    self.Tissues_name = Tissues_name
    self.T0, self.P0 = T0 ,P0
    self.path_T0, self.path_  = path_T0, path_
    self.ntime =ntime
    self.load_w_t, self.load_data = load_w_data    
    self.Is_tumor, self.Is_blood,  = Is_segments
    self.Time = blood_band['time']
    self.LB_t, self.UB_t = lb, ub  # dict
    self.C_band = C_band
    
    self.D1_shape = int(minpoints/2)  #Size of datasets in each batch
    self.d3 = layers['Skin'][0]
    self.layers = layers
    self.it_Stop = it_Stop
    #-----------------Tissues properties and initialization-------------------#
    # Parameters and new conditions for PDE, Q of laser and initial Temp
    # self.Time[1]-self.Time[0] = 10 but tn =5s
    Tn = self.Time[1]-self.Time[0] if tn==0 else tn
    self.ND = {key: NDparam.NonDim_param(Skin_layer=key,tn = Tn,
                                         remove_d =[]) for key in self.Tissues_name}   
    
    self.sigma = 0.01 
    self.T_inf = self.ND['Skin_1st'].ND_T(34 + 273.15)  
    self.T_in = self.ND['Skin_1st'].ND_T(35 + 273.15, blood = self.Is_blood)
    # in qs: N = 1 means fv = 7e9 (1/cm3) density of Gold-nanoRods N=10 -->7e10
    #-------------------------------------------------------------------------#
    self.er_est1 =False
    if er_est and self.Time[1] in [10,20,50,100]:
        self.er_est1 = True
        with open(data_path_err+'/Comsol_center_yz_plane'+str(self.Time[1])+'.pkl','rb') as f:
              C1 = pkl.load(f)
        self.Uyz_comsol = {'U_exact': (C1['U_exact'][2], C1['U_exact'][1], C1['U_exact'][0]),
                           'X_estimate': (C1['X_estimate'][2], C1['X_estimate'][1], C1['X_estimate'][0])}
                           
        with open(data_path_err+'/Comsol_Skin Surface_xy_plane'+str(self.Time[1])+'.pkl','rb') as f:
             C1 = pkl.load(f)   
        self.Usurf_comsol = {'U_exact': C1['U_exact'],
                            'X_estimate': (C1['X_estimate'][2], C1['X_estimate'][1], C1['X_estimate'][0])}     
    #-------------------------------------------------------------------------#
    # 3) load the Model of previoys time step as initial conditions for the next time step
    keras.backend.clear_session()
    if path_T0:
        self.merged_model_T0 = keras.models.load_model(path_T0+'/tissue',compile=False)
    #-------------------------------------------------------------------------#
    # Load/create the tissue's and blood vessel's models   
    if not self.load_data: self.Tissues_DataSet(args_tissue)
    #-------------Blood properties and initialization-------------------------#
    # 2) initialize the global weights | load the global weights
    if self.load_w_t==False:
        Loss_num_t =  24 # "0" mcmh and '1' is rand
        self.w_t = [tf.Variable(1.0, dtype = DTYPE) for ii in range(Loss_num_t)]# dWPINN
    #-------------------------------------------------------------------------#            
    #Combine and merge the models
    INPUTS_, OUTPUTS_= () ,()
    for key in self.Tissues_name: #self.Model_keys:
        print('key is: ' ,key)                       
        input_ = tf.keras.Input(shape = (layers['Skin'][0],),dtype = DTYPE)
        # Initialize models for triple layers if we don't have saved_model  
        #Act = 'tanh' #if key=='Skin_3rd' else 'swish'
        model_T = PINN_model.PINN_model(act_func, layers['Skin'], LBUB=(lb[key],ub[key]))
        output_ = model_T(input_)
        INPUTS_ += (input_ ,)
        OUTPUTS_ += (output_ ,)
    self.merged_model = tf.keras.Model(inputs = INPUTS_ , outputs = OUTPUTS_)
    
    #%%------------------------------------------------------------------------# 
    if path_1:
          W_load1 = tf.keras.models.load_model(path_1+'/tissue', compile=False).get_weights()[:24]
          print('*************** Skin 1st-2nd Model is loaded*******************')    
    if path_:
       W_load3 = tf.keras.models.load_model(path_+'/tissue', compile=False).get_weights()
       print('*************** Tissues Model is loaded*******************')
       
    if path_1 and path_: W_load = W_load1 + W_load3[-12:]
    elif path_1: W_load = W_load1 + self.merged_model.get_weights()[-12:]
    elif path_: W_load = W_load3 
    else: W_load = self.merged_model.get_weights()
    
    self.merged_model.set_weights(W_load)

    #%%-----------------------------------------------------------------------#
    # Define the cyclic learning rate schedule
    # https://github.com/bckenstler/CLR?ref=jeremyjordan.me
    self.step0 = tf.Variable(0, trainable=False) 
    # lr_cycle_start = 0 if len(self.path_)==0 else 2 if self.load_w_t else 1
    lr_cycle = [(0.004, 8e-4),(8e-4, 1e-4),(2e-4, 6e-5),(5e-5, 2e-5),(8e-6, 9e-6)]
    self.max_lr, self.base_lr = lr_cycle[self.lr_cycle_start]
    self.learning_rate_schedule = MyLRSchedule_Cycle2.MyLRSchedule_Cycle()
    #-------------------------------------------------------------------------#
    # 5) Optimizers  : learning_rate = 0.0001, beta1 = 0.99, beta2 = 0.99 
    if SGD:
        self.optimizer_method = tf.keras.optimizers.SGD(learning_rate = 1e-4, momentum=0.9)
    if opt=='adamw':
        print('***************Adam WWWWWWWWWW*******************')
        self.optimizer_method = tf.keras.optimizers.AdamW(learning_rate=
                                        self.learning_rate_schedule(0, self.max_lr, self.base_lr),
                                        amsgrad=True, weight_decay = 1e-4)
    elif opt=='adadelta':
       print('***************Adam Delta*******************')
       self.optimizer_method = tf.keras.optimizers.Adadelta(learning_rate=0.0002)
       
    else:
        self.optimizer_method = tf.keras.optimizers.Adam(learning_rate=
                                        self.learning_rate_schedule(0, self.max_lr, self.base_lr))
                                                         
    #self.optimizer_Wt = tf.keras.optimizers.Adam(learning_rate = 0.005)#self.lr_wt)  
    self.optimizer_Wt = tf.keras.optimizers.SGD(learning_rate = 0.95)
    print('***************SGD optimizer for weights*******************')
  #---------------------------------------------------------------------------#                
  def Tissues_DataSet(self,args_tissue = [], rank_ = 0,data_path=[]):       # Tissues
     #  args_tissue = dict(['X0', 'Xf', 'X_lb', 'X_ub', 'Y_lb_ub', 'Z_lb', 'Z_ub'])
     # Tissue name: ['Skin_1st','Skin_2nd','Tumor1','Tumor2',
     #                   'Gold_Shell', 'Skin_3rd','Skin_3rd_blood']
     if self.load_data:
         with open(data_path+'/newData_MCMH'+str(rank_)+'_Rank.pkl' , 'rb') as f:
             new_data = pkl.load(f)
         T0 = self.create_T0(new_data['X_tissue'][0], self.ntime)
         self.train_dataset = tf.data.Dataset.from_tensors(new_data['X_tissue'][:-1]+(T0,))
             
     else:
        # X0, ,Xf, X_ub, X_lb, Y_lb, Z_lb [all tissues (7)], 
        # Z_ub [skin_3rd, tumor2, gold, skin_3rd_blood]
        X0_p = self.fd(args_tissue['X0'])
        T0 = self.create_T0(X0_p, self.ntime)
        dataset = (X0_p, self.fd(args_tissue['Xf']),  self.fd(args_tissue['X_lb']),
                        self.fd(args_tissue['X_ub']), self.fd(args_tissue['Y_lb_ub']),
                        self.fd(args_tissue['Z_lb']), self.fd(args_tissue['Z_ub']),T0)
        
        train_dataset_ = tf.data.Dataset.from_tensor_slices(dataset)
        self.train_dataset = train_dataset_.shuffle(args_tissue['X0'][0].shape[0]).\
                                 batch(self.D1_shape,drop_remainder=True)
     self.train_dataset.prefetch(tf.data.AUTOTUNE)
  #%%------------------------ Utilities functions ----------------------------#
  def fd(self, x):   
     P_shape =len(x[-1].shape)-1
     if P_variable:
         x_out = tuple(tf.concat([xx, tf.random.uniform(xx.shape[:P_shape]+(1,),dtype =DTYPE)],\
                                axis=P_shape) if len(xx)>0 else xx for xx in x)
     else:
         if self.d3==5:
             x_out = tuple(tf.concat([xx, self.P0 *tf.ones(xx.shape[:P_shape]+(1,),dtype =DTYPE)], 
                                 axis = P_shape) if len(xx)>0 else xx for xx in x)
         else: 
             x_out = x
                 
     return x_out
  #---------------------------------------------------------------------------#
  def create_T0(self,x0, ntime):
     if self.Time[0]>0: 
        input_all =self.add_ConstColumn(self.ft(x0), C = ntime, NCol = 3)
        u_pred = self.merged_model_T0(input_all)

        u_0 = tuple(tf.reshape(u_pred[i], x0[i].shape[:-1]+(1,)) for i in range(len(u_pred)))
     else:
        u_0 = tuple(self.T0['Skin']*tf.ones(xi.shape[:-1]+(1,),dtype=DTYPE) for xi in x0)
     return  u_0
  #----------------------------------------------------------------------------#
  def add_ConstColumn(self, X, C = [1,1,1,1,1], NCol = 2):
      if not(isinstance(C,list)): C=[C]
      if isinstance(X,tuple):
         if len(C)!=len(X): C = [C[0] for _ in range(len(X))]
         return tuple(tf.concat((X[j][:,:NCol], C[j]* tf.ones((X[j].shape[0],1),\
                       dtype=DTYPE), X[j][:,NCol:]), axis=1) for j in range(len(X)))
      else:
         if isinstance(C, list) or isinstance(C, tuple):
            return tuple(tf.concat((X[:,:NCol], C[j]* tf.ones((X.shape[0],1),\
                       dtype=DTYPE), X[:,NCol:]), axis=1) for j in range(len(C)))  
         else:
            return tf.concat((X[:,:NCol], C* tf.ones((X.shape[0],1),\
                            dtype=DTYPE), X[:,NCol:]), axis=1)  
  
  #%%-------------------------------------------------------------------------#
  def net_skinMerged(self,X, mixed = tf.constant(0), T_num = 2):  
      # T_num : the number of tissues in which X is defined when len(X)!=len(self.Tissues_name)
      if len(X)!=len(self.Tissues_name):
          temp = tf.ones([3,self.d3], dtype=DTYPE)
          X = tuple(X[0] if i==T_num else temp for i in range(len(self.Tissues_name)))
      if mixed==2: # For blood ends
          T = self.merged_model(inputs = X)
          return T[T_num]
      else: return self.net_skinMerged_models(X, mixed = mixed)

  @tf.function
  def net_skinMerged_models(self, X , mixed = 0):#tf.constant(0)):
    # mixed = 1 for PDE    # mixed = 3 for IC
    # mixed = 0 for BCs   # mixed = 2 for Power
    # inputs and outputs are tuples: X is a tuple
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(X)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(X)
            T = self.merged_model(inputs = X) # T is a tuple with size of 3                
        # inputs is also a list of size of 3
        if mixed < 3 :
           T_grad = tape1.gradient(T, X)
           Tx = [T_grad[j][:,0:1] for j in range(len(T_grad))]
           Ty = [T_grad[j][:,1:2] for j in range(len(T_grad))]
           Tz = [T_grad[j][:,2:3] for j in range(len(T_grad))]
           Tt = [T_grad[j][:,3:4] for j in range(len(T_grad))]
           
    if mixed == 1:  # for PDE
        T2_gradx = tape2.gradient(Tx, X)
        Txx = [T2_gradx[j][:,0:1] for j in range(len(T2_gradx))]
        T2_grady = tape2.gradient(Ty, X)
        Tyy = [T2_grady[j][:,1:2] for j in range(len(T2_grady))]
        T2_gradz = tape2.gradient(Tz, X)
        Tzz = [T2_gradz[j][:,2:3] for j in range(len(T2_gradz))]

        del tape2, tape1
        return T,Tt,Txx,Tyy,Tzz,()
        
    elif mixed==0: return T,Tx,Ty,Tz,Tt # otherwise mixed == 0
    else: return T, ()
  
  #%% *********************** Skin Tissues Losses ****************************#
  #***************************************************************************#
  def ft(self, x, n = 0):
     if n==0: n=self.d3-1
     return tuple(tf.reshape(i,(-1,n)) for i in x)
  #%%--------------------------------------------------------------------------#
  def Loss_Skin_tissues(self, X_d): 
     # Nt = len(self.Tissues_name) 
     X_dist_batch = tuple(tuple([] if (j==6 and i<2) else tf.concat([X_d[0][j][i],X_d[1][j][i]],axis=1)
                           for i in range(len(X_d[0][0]))) for j in range(len(X_d[0])))
     
     x0, x_f, x_lbx, x_ubx, x_lb_uby , x_lbz, x_ubz, T0 = X_dist_batch
     #------------------------------------------------------------------------#     
     T00 = tuple(tf.reshape(i,(-1,1)) for i in T0)
     
     Loss_f = self.Skin_Loss_PDE(self.ft(x_f,self.d3)) #--> Nt losses
     # Losses on BC x-y-z-dir, IC, PDE, and IFC 
     Loss_0 = self.Skin_Loss_IC(self.ft(x0), T00) #--> Nt losses

     Loss_xub = self.Skin_Loss_symmetry_Xub(self.ft(x_ubx))  # -> len(Tissues) losses
     #------------------------#
     # Loss_x for ['Skin-1st - Skin-3rd] -->3 (3)
     Loss_xlb = self.Skin_Loss_BCXY(self.ft(x_lbx), 'lb', col = 0)

     Y = self.ft(x_lb_uby)
     Loss_ylb = self.Skin_Loss_BCXY(Y, 'lb', col = 1)
     Loss_yub = self.Skin_Loss_BCXY(Y, 'ub', col = 1)
     #------------------------#
     # loss0 (1) + IFC (2)|(3)|(7)|(8) + UBz skin_3rd (2 with blood) or (1))-->(4)|(6)|(9)|(11)
     Loss_z = self.Skin_Loss_BCZ(self.ft(x_lbz), self.ft(x_ubz))
     
     #----------# Loss_f + Loss_0 + Symmetery + LossX_Y insulation -> 3*Nt + 6
     # Nt(Lf) + Nt(L0)+ Nt(L_ubx) + 3[L_lbx] + 3[Ly] + 1 [L_z0]
     Loss_T = Loss_f + Loss_0 + Loss_xub + Loss_xlb + Loss_ylb+ Loss_yub   
          
     return Loss_T + Loss_z

  #%% Initial Conditions ( All self.Tissues_name members) 
  def Skin_Loss_IC(self,x0, T00, plt_loss=0): 
      #  X0 is 4D --> 0  should be added to column 3
      # IC  : U0 , dU0 
      U0_pred, U0t_pred = self.net_skinMerged(self.add_ConstColumn(x0, C=0, NCol=3), mixed=3)
      
      Loss_0_data = [tf.square(U0_pred[ii]-T00[ii]) for ii in range(len(U0_pred))]
          
      if plt_loss: return Loss_0_data
      else:       #  MSE0 of key 
         Loss_0 = [tf.reduce_mean(Loss_0_data[ii]) for ii in range(len(Loss_0_data))]
         return  Loss_0   
      
  #%% Symmetry Boundary conditions on X --> X_ub  ( All self.Tissues_name members)
  def Skin_Loss_symmetry_Xub(self,x_ub, plt_loss = 0): 
      #  Xub is 4D -->cx should be added to column 0
      _, ux_ubx, _,_,_ = self.net_skinMerged(self.add_ConstColumn(x_ub,
                  [self.UB_t[ky][0,0] for ky in self.Tissues_name] ,NCol=0), mixed = 0)
          
      if plt_loss==0: return [tf.reduce_mean(tf.square(ux)) for ux in ux_ubx]             
      else: return [tf.square(ux) for ux in ux_ubx] 
        
  #%% BC on XY-dirs: insulation BC + IFC (for tumors, Gold_nanshell and Skin_3rd_blood) 
  def transform_fun(self,X,Transform_from =[], Transform_to =[]):
      # Transform_to should not contain identical members
      # the length of X should be the same as self.Tissues_name
      # X shoul be created for Transform-from
      temp = tf.ones([3,self.d3], dtype=DTYPE)
      Xnew = ()
      #-----------------------------------------------------------------------#
      for ii, key_frwd in enumerate(self.Tissues_name):
          if key_frwd in Transform_to:
              jj = Transform_to.index(key_frwd)
              key_inv = Transform_from[jj]
              if key_frwd==key_inv:
                  Xnew +=(X[jj],)
              else:
                  Xnew += (self.ND[key_frwd].Forward_Var(self.ND[key_inv].Inverse_Var(
                          X[jj],t0 =self.Time[0]),t0 = self.Time[0]),)
          else:Xnew +=(temp,)
      return Xnew
  #---------------------------------------------------------------------------#    
  def Skin_Loss_BCXY(self,  X , Bound ='lb', col = 0, plt_loss = 0): 
      #  X is 4D --> C_band[key][0,col]  should be added to column col 
      Loss_bc = [] 
      C_band = self.LB_t if Bound =='lb' else self.UB_t     
      XD = self.add_ConstColumn(X, C=[C_band[ky][0,col] for ky in self.Tissues_name] ,NCol = col)

      u_pred, *ud_pred = self.net_skinMerged(XD, mixed = 0)
      
      for ii , key in enumerate(self.Tissues_name):
         Loss_bc.append(tf.reduce_mean(tf.square(ud_pred[col][ii])))
          
      return Loss_bc
                                                             
  #%%Boundery and interfacial Conditions on z-dir and their losses
  def Skin_Loss_BCZ(self, z_lb, z_ub, plt_loss=0):  # Z is 4D (x,y,z,t,p) 
 
      Zlb =self.add_ConstColumn(z_lb, C=[self.LB_t[ky][0,2] for ky in self.Tissues_name] ,NCol = 2)
      
      Zub = tuple(self.add_ConstColumn(zi, C= self.UB_t[self.Tissues_name[ii]][0,2] ,NCol = 2)[0]\
                          if len(zi)>0 else zi for ii,zi in enumerate(z_ub))
      
      # to make the code runs faster, lb and ub are combined first and 
      # after estimation they will separted vs their neighbors
      Ci = [self.ND[ky].param.kl()/self.ND[ky].Char_Len()[2] for ky in self.Tissues_name]
      
      i3, i2 = self.Tissues_name.index('Skin_3rd'), self.Tissues_name.index('Skin_2nd')
      # function estimation for LB
      U_lb,_, _, Ud_lb, Ulbt = self.net_skinMerged(Zlb, mixed = 0) 
      # function estimation for UB
      Zub_tissue = self.transform_fun([Zlb[i2], Zlb[i3]],['Skin_2nd','Skin_3rd'], 
                                                         ['Skin_1st','Skin_2nd'])
      
      Zub_new = (Zub_tissue[0],Zub_tissue[1],Zub[2])
          
      U_ub,_, _, Ud_ub,_ = self.net_skinMerged(Zub_new, mixed = 0) 
      #---------------------------------------------------------------------#     
      U_IFC = (((U_ub[0],Ci[0]*Ud_ub[0]) ,(U_lb[1],Ci[1]*Ud_lb[1])),  #'Skin_2nd & 1st
               ((U_ub[1],Ci[1]*Ud_ub[1]) ,(U_lb[2],Ci[2]*Ud_lb[2]))) #'Skin_2nd& 3rd
                                       
      #---------------------------Loss_estimations-----------------------------# 
      LossT_IFC, LossdT_IFC, Lossz =[], [], []   

      if plt_loss==0:Lossz0 = [60*tf.reduce_mean(tf.square(Ud_lb[0]-self.ND['Skin_1st'].Bi1*(U_lb[0]- self.T_inf)))]
      else: LossT_IFC.append(tf.square(Ud_lb[0]-self.ND['Skin_1st'].Bi1*(U_lb[0]- self.T_inf)))
      #-----------------------------------------------------------------------# 
      w = [7,7,4,1]
      for j,i in enumerate(range(len(U_IFC))):  #--> 8 (7) for case with blood --> 16(14)
          if plt_loss==0:
             LossT_IFC.append(w[j]*tf.reduce_mean(tf.square(U_IFC[i][0][0]-U_IFC[i][1][0])))
             LossdT_IFC.append(w[j]*tf.reduce_mean(tf.square(U_IFC[i][0][1]-U_IFC[i][1][1])))
          else: 
              LossT_IFC.append(tf.square(U_IFC[i][0][0]-U_IFC[i][1][0])+\
                                tf.square(U_IFC[i][0][1]-U_IFC[i][1][1]))
      #-----------------------------------------------------------------------#       
      if plt_loss==0: 
          Lossz.append(tf.reduce_mean(tf.square(Ud_ub[i3])))     
      # total loss has loss0 (1) + IFC (total 8) + UBz skin_3rd (max=2 or (1))           
      if plt_loss==0:return Lossz0 + LossT_IFC + LossdT_IFC + Lossz #(6)
      else: return LossT_IFC  # Loss of LBZ + UBZ (the last 2 losses for ub_tumor2, ub_gold)     
  #-------------------------------------------------------------------------#
  #%% Loss of Skin PDE 
  def Skin_Loss_PDE(self, x  ,plt_loss = 0): 
    # mixed = 1  # For PDE in net_skinMerged function
    # x = (x_f for: 'Skin_3rd' ,'Skin_2nd','Skin_1st') : tuple  
    Loss_f , Loss_f_data = [] ,[]
    T, Tt, Txx, Tyy, Tzz, Ttt = self.net_skinMerged(x, mixed = 1) 
    wf =  [20, 20, 10]
    
    for j,key in enumerate(self.Tissues_name):
        ND = self.ND[key]
        Q = self.Q_estimation(x[j], ND, key)
        # physics informed
        f_u = Tt[j] -ND.F0[0]*Txx[j]- ND.F0[1]*Tyy[j]- ND.F0[2]*Tzz[j]-\
                              ND.A*(self.T_in-T[j])- Q/ND.Q_Coeff()                              
        # 1 - MSE PDEs for 3 skin_layers
        Loss_f_data.append(tf.square(f_u))
        if plt_loss==0:
            Loss_f.append(wf[j]*tf.reduce_mean(Loss_f_data[j]))
                
    if plt_loss: return Loss_f_data
    else: return Loss_f    # it is a list

  # ------------------------------------------------------------------------#
  def Q_estimation(self, x, ND, Skin_layer =[]):
      alpha , Reff = 1.8, 0.1     
      Gauusian_xy = tf.exp((tf.square(ND.Inverse_Var(x[:,0:1],d='x')-0.5)+
                    tf.square(ND.Inverse_Var(x[:,1:2],d='y')-0.5))/\
                           (-2*self.sigma**2))/tf.sqrt(2*np.pi*self.sigma**2)
  
      Q = alpha *tf.exp(-alpha*ND.Inverse_Var(x[:,2:3],d='z'))
      
      if P_variable:return Q*(x[:,4:5]*dP + Pmin)*(1-Reff)*Gauusian_xy  #
      else:return Q*self.P0*(1-Reff)*Gauusian_xy  # Iin = P0 *(1 - Reff)     

  #***************************************************************************#      
  def train_vars(self):
       return self.merged_model.trainable_weights 
  #%% ------------------------------------------------------------------------#
  ##**************************************************************************#
  @tf.function#(jit_compile = True)  
  def loss_fn_opt(self, X_dist_Tissue, first_batch = False ):  
     with tf.GradientTape(persistent=True) as tape:
         Losses = self.Loss_Skin_tissues(X_dist_Tissue)
         W_Losses = [self.w_t[i]*Losses[i] for i in range(len(self.w_t))]
         wLoss = tf.reduce_sum(W_Losses)
     #-----------------------------------------------------------------------# 
     # Horovod: add Horovod Distributed GradientTape. --> Step 5
     tape = hvd.DistributedGradientTape(tape)
     grads = tape.gradient(wLoss , self.train_vars())

     self.optimizer_method.apply_gradients(zip(grads, self.train_vars()))

     # Horovod: broadcast initial variable states from rank 0 to all other processes.
     # This is necessary to ensure consistent initialization of all workers when
     # training is started with random weights or restored from a checkpoint.
     # Note: broadcast 
     if first_batch:
         hvd.broadcast_variables(self.train_vars(), root_rank=0)
         hvd.broadcast_variables(self.optimizer_method.variables(), root_rank=0) 
     #-----------------------------------------------------------------------#
     return  Losses, W_Losses
  #----------------------------------------------------------------------------#
  @tf.function
  def dwPINN_update(self, Losses , weight_ ):
      with tf.GradientTape(persistent=True) as tapew:
        tapew.watch(weight_)
        WL= [-Losses[i]*weight_[i] for i in range(len(weight_))]
        
      grads_wt = tapew.gradient(WL , weight_)
      self.optimizer_Wt.apply_gradients(zip(grads_wt, weight_))
  #----------------------------------------------------------------------------#
  def main_Metropolis_Hasting_Alg(self, Xt ): 
      x0, x_f, x_lbx, x_ubx, x_lby , x_lbz, x_ubz, T0 = Xt
      
      x_f = self.Metropolis_Hasting_Alg_PDE(x_f)
      x0, T0 = self.Metropolis_Hasting_Alg_IC(x0,T0)
      x_ubx = self.Metropolis_Hasting_Alg_ubX(x_ubx)
      x_lbz, x_ubz = self.Metropolis_Hasting_Alg_z_IFC( x_lbz, x_ubz)      

      return (x0, x_f, x_lbx, x_ubx, x_lby , x_lbz, x_ubz, T0)

  #%%-------------------------------------------------------------------------#    
  # train loop    
  def train(self, save_path=[], max_Stop_err = 2.00e-5, 
            mcmh = False, data_path=[], LL_Loss = 650 , alpha_constant = 'False'): 
    #-------------------------------------------------------------------------#    
    # initializations
    if not save_path: save_path = self.path_
    if not data_path: data_path = self.path_T0
    it_Stop = self.it_Stop
    it , stopping_run ,Loss_overal = 0, 0, 2
    self.LL_Loss = LL_Loss
    #--------------------------- indices if Losses ---------------------------#
    Nt = len(self.Tissues_name)
    # f |0 | ubx | lbx | lby |uby
    Loss_ind = {'Skin_1st':[0, Nt, 2*Nt, 3*Nt, 3*Nt+3, 3*Nt+6]}
    Loss_ind['Skin_2nd'] = [i+1 for i in Loss_ind['Skin_1st']]
    Loss_ind['Skin_3rd'] = [i+1 for i in Loss_ind['Skin_2nd']]
    ind_n , Ifc_n = 3*Nt+8 ,6
    Loss_ind['IFC'] = list(range(ind_n, Ifc_n + ind_n))
    #-------------------------------------------------------------------------#
    self.writer_train = tf.summary.create_file_writer(save_path + 'train')
    os.makedirs(save_path,exist_ok=True)
    start_time = time.time()
    #-------------------------------------------------------------------------#
    # 4- Distribute Data and initialize NN
    S , XT_tf = 2 ,{}
    if self.load_w_t:#self.load_data: 
       with open(data_path+'/Power0_Rank.pkl','rb') as f:
              P_W_Data = pkl.load(f)         
       self.w_t = P_W_Data['wL']['wt'] 
    weight_ = self.w_t 
    Loss_num = len(weight_)
    #-----------------------------------------------------------------#
    for i in range(S):#hvd.size())
       if self.load_data:
          self.Tissues_DataSet(rank_ = i, data_path = data_path)
          XT_tf[i] = next(iter(self.train_dataset)) 
       else:
          XT_tf[i] = next(iter(self.train_dataset.skip(i).take(1)))    
    #-----------------------------------------------------------------#
    self.step_LR = int(2*(self.lr_cycle_start + 3))
    it_stop = min(60000,int(it_Stop*0.45))
    #---------------------------------------------------------------------#
    while stopping_run<=10002 and it<it_Stop :
   
      if it==0: 
          Loss, W_Loss = self.loss_fn_opt(XT_tf, first_batch = True) 
      else:
         Loss, W_Loss = self.loss_fn_opt(XT_tf, first_batch = False) 
      #-----------------------------------------------------------------#
      if it<it_stop and Loss_overal<0.2:
         if mcmh and it%200==0: 
             XT_tf[1] = self.main_Metropolis_Hasting_Alg(XT_tf[1])
      #----------------------------------------------------------------#
      Losses = np.reshape(Loss,(Loss_num,1))
      W_Loss_t = np.reshape(W_Loss,(Loss_num,))
      Loss_overal = hvd.allreduce(np.sum(Losses), average=True)              
      #%%--------------------------------------------------------------------#
      # weight optimizations 
      if (it>self.it_Stop*0.1 and Loss_overal<0.1) or self.load_w_t: 
         if it%1==0:# and it<self.it_Stop*0.8: 
             self.dwPINN_update(tf.convert_to_tensor(Losses), weight_) 
      #%%--------------------------------------------------------------------#
      self.step0.assign_add(1)
      if it>self.it_Stop*0.92 or stopping_run>0: self.max_lr, self.base_lr = 1, 1
      else:
         if it%10000==0 and (self.load_w_t or it>0):
             if it<=self.it_Stop*0.06:# or it>=self.it_Stop*0.72: 
                 self.alpha_max_min(W_Loss_t, it)
         if it%1000==0 and (self.load_w_t or it>0):
             if it>self.it_Stop*0.06: 
                 self.alpha_mine(W_Loss_t, it, stopping_run)
                 
         if it<self.it_Stop*0.06: self.max_lr = 0
         elif it>self.it_Stop*0.83: self.max_lr = 0.5  # 8e-6
         elif it>self.it_Stop*0.76: self.max_lr = 0.25 # 1e-5

      if alpha_constant: 
          if it>self.it_Stop*0.6: self.max_lr, self.base_lr = 1, 1
          else: self.max_lr = 0.5
      learning_rate = self.learning_rate_schedule(self.step0 , self.max_lr, self.base_lr)
      self.optimizer_method.learning_rate = learning_rate 
      if Loss_overal<max_Stop_err and it>self.it_Stop*0.25: stopping_run+=1
      #%%--------------------------------------------------------------------#
      if it>0:
         if Loss_overal<max_Stop_err and it>self.it_Stop*0.25: stopping_run+=1
         if  it % 500 == 0:
            elapsed = time.time() - start_time
            tf.print('rank0: Skin-Only,- %d s- ,It: %d, Loss: %.3e, Loss overal:%.3e, w-Loss: %.3e, Time: %.2f, lr: %.3e' 
                      %(self.Time[1], it, np.sum(Losses),Loss_overal,np.sum(W_Loss_t),
                            elapsed, learning_rate.numpy()))
            if  it % 5000 == 0:
               for iw in [2,5]:
                   tf.print( f' weight {iw} is {weight_[iw]}****')
            start_time = time.time()
         #--------------------------Print Losses----------------------------#
            for index in Loss_ind.keys():
               tf.print('loss_ %s: %.3e'% (index,sum([Losses[i,0] for i in Loss_ind[index]])))                    
            #--------------------Save the summary of Results----------------#      
            with self.writer_train.as_default():  
               for i in range(len(Losses)):
                   tf.summary.scalar(name='Loss_' + str(i), data=Losses[i,0], step=it)
                   tf.summary.scalar(name='Weight_'+ str(i), data=weight_[i], step=it)
         if  it % 10 == 0: 
             with self.writer_train.as_default():
                tf.summary.scalar(name ='iter', data = it, step=it)
                tf.summary.scalar(name ='Loss_overal', data = Loss_overal, step=it)
                tf.summary.scalar(name ='learning_rate', data = learning_rate, step=it)
         if it%100==0 and self.er_est1: self.Error_estimate(it)       
         #---------------------------------------------------------------#
         # Save Models
         if it %2500 == 0 or it==it_Stop-1 or stopping_run==10002:
            self.merged_model.save(save_path+'/tissue',save_traces=True) #save_format='keras_v3')#, 
         #------------------------ Save for  all ranks--------------------------
         if (it%2000==0 and it>3000) or  it==it_Stop-1 or stopping_run==10002:
             with open(save_path+'/Power0_Rank.pkl','wb') as f:
                pkl.dump({'wL': {'wt': self.w_t},'Loss_far':Losses,'LL_Loss':self.LL_Loss},f)
             for si in range(2):
                  with open(save_path+'/newData_MCMH'+str(si)+'_Rank.pkl','wb') as f:
                     pkl.dump({'X_tissue': XT_tf[si]},f)
         #----------------------------------------------------------------------#        
      it+=1
    return Loss_overal   
   
  #***************************************************************************#
  #                       Metropolis-Hasting Algorithm                        #
  #***************************************************************************#
  def ft_inv(self, x, n = 0, D =0):
      D1 = D if D else self.D1_shape 
      n1 = n if n else self.d3-1 
      return tuple(tf.reshape(i,(D1,-1, n1)) for i in x)
  #---------------------------------------------------------------------------#
  def Metropolis_Hasting_Alg_PDE(self, Xf):
      Xf = self.ft(Xf,self.d3)
      Nf = [xii.shape[0] for xii in Xf]
      Xf_prop = mcmh_fun.new_prop_points_tf(self.LB_t, self.UB_t, Nf,
                                            self.C_band, Key_order = self.Tissues_name)
      Xf_prop = self.fd(Xf_prop)
      #-----------------------------------------------------------------------#
      Loss_f_data = self.Skin_Loss_PDE(Xf, plt_loss = 1)
      Loss_f_prop = self.Skin_Loss_PDE(Xf_prop ,plt_loss = 1)
      alphaf = [Loss_f_prop[i]/Loss_f_data[i] for i in range(len(Loss_f_data))]
      del Loss_f_data, Loss_f_prop
      return self.ft_inv(self.Metropolis_Hasting_Alg_loss(alphaf ,Xf, Xf_prop),self.d3)
      #return self.Metropolis_Hasting_Alg_loss(alphaf ,Xf, Xf_prop)
  #---------------------------------------------------------------------------#
  def Metropolis_Hasting_Alg_IC(self, X0, T00):
      X0, T00 = self.ft(X0),self.ft(T00,1)
      Loss_0_data = self.Skin_Loss_IC(X0, T00 , plt_loss = 1)
      N0 = [xii.shape[0] for xii in X0]

      X0_prop=mcmh_fun.new_prop_points_tf(self.LB_t,self.UB_t, N0,
                                 self.C_band, Bound = 4,Key_order = self.Tissues_name)
      X0_prop = self.fd(X0_prop)
      T0p = self.create_T0(X0_prop, self.ntime)
      #-----------------------------------------------------------------------#
      Loss_0_prop = self.Skin_Loss_IC(X0_prop, T0p, plt_loss = 1)
      alpha0 = [Loss_0_prop[i]/Loss_0_data[i] for i in range(len(Loss_0_data))]
      del Loss_0_data, Loss_0_prop
      X0new = self.ft_inv(self.Metropolis_Hasting_Alg_loss(alpha0 ,X0, X0_prop))
      #X0new = self.Metropolis_Hasting_Alg_loss(alpha0 ,X0, X0_prop,X0_more)
      T0_new = self.create_T0(X0new, self.ntime)
      return X0new, T0_new
  #---------------------------------------------------------------------------#
  def Metropolis_Hasting_Alg_ubX(self, Xub_x):
      Xub_x = self.ft(Xub_x)
      Loss_ubx_data = self.Skin_Loss_symmetry_Xub(Xub_x, plt_loss = 1)
      
      Xub_x_prop=mcmh_fun.new_prop_points_tf(self.LB_t,self.UB_t,[xi.shape[0] for xi in Xub_x]
                ,self.C_band, Bound = 1, Dir = 'ub', Key_order = self.Tissues_name)
      Xub_x_prop = self.fd(Xub_x_prop)
      
      Loss_ubx_prop = self.Skin_Loss_symmetry_Xub(Xub_x_prop, plt_loss = 1)
     
      alphax = [Loss_ubx_prop[i]/Loss_ubx_data[i] for i in range(len(Loss_ubx_data))]
      del Loss_ubx_prop,Loss_ubx_data

      return self.ft_inv(self.Metropolis_Hasting_Alg_loss(alphax ,Xub_x, Xub_x_prop))
  #---------------------------------------------------------------------------#
  def Metropolis_Hasting_Alg_z_IFC(self, x_lbz, x_ubz):
      Xlbz, Xubz = self.ft(x_lbz), self.ft(x_ubz)     
      Nz = [xii.shape[0] for xii in Xlbz]
      Xlbz_prop = mcmh_fun.new_prop_points_tf(self.LB_t, self.UB_t, Nz,
                                self.C_band, Bound = 3, Dir = 'lb',Key_order = self.Tissues_name)
      Xlbz_prop = self.fd(Xlbz_prop)
      if self.Is_tumor:
           #Nt = [Xubz[i].shape[0] if i>=3 else 0 for i in range(len(Xubz))]
           Nt = [Xubz[i].shape[0] for i in range(len(Xubz))]
           Xubz_prop = mcmh_fun.new_prop_points_tf(self.LB_t,self.UB_t,Nt,
                        self.C_band, Bound = 3, Dir = 'ub',Key_order = self.Tissues_name)
           Xubz_prop = self.fd(Xubz_prop)
      else:Xubz_prop = Xubz
      # lossz = [lb of all tissues] if Is_tumor : [ub: tumor2, and Gold]
      Loss_z_data= self.Skin_Loss_BCZ(Xlbz, Xubz, plt_loss = 1) 
      Loss_z_prop= self.Skin_Loss_BCZ(Xlbz_prop, Xubz_prop, plt_loss = 1)
      
      alphaz = [Loss_z_prop[i]/Loss_z_data[i] for i in range(len(Loss_z_data))]
      #----------------------------------------------------------------------#
      Xlbznew = self.Metropolis_Hasting_Alg_loss(alphaz[:len(Xlbz)], Xlbz, Xlbz_prop)#, Xlbz_more)
      
      if self.Is_tumor:
          alphaub = [0, 0, 0, 0, alphaz[-2], alphaz[-1]]
          Xubznew = self.Metropolis_Hasting_Alg_loss(alphaub, Xubz, Xubz_prop)
      else: Xubznew = Xubz
             
      return self.ft_inv(Xlbznew), self.ft_inv(Xubznew)
      #except: return Xlbznew, self.ft_inv(Xubznew)
  #---------------------------------------------------------------------------#
  def Metropolis_Hasting_Alg_loss(self,alpha, X, Xp, Xmore = []):
      N = [alpha[i].shape[0] if tf.is_tensor(alpha[i]) else 0  for i in range(len(alpha))]    
      C = tuple(tf.random.uniform([N[i],1], minval=0.01, maxval=1.0,dtype =DTYPE) 
                for i in range(len(N)))
      if len(Xmore): 
         return tuple(tf.concat([tf.where(alpha[i]>=C[i], Xp[i], X[i]),Xmore[i]],axis=0) 
                      if N[i]>0 else X[i] for i in range(len(N)))
      
      else: 
         return tuple(tf.where(alpha[i]>=C[i], Xp[i], X[i]) if N[i]>0 else X[i]
                                                         for i in range(len(N)))   
  #---------------------------------------------------------------------------#
  # alpha
  def alpha_max_min(self, W_Loss_t, it):
      # Every 20K iterations --> max and min will change--> e.g.LL_Loss = 200
      # starting 20 K itearion (start is based on our definitions in cycles)
      self.step0.assign(0)
      # m2 = 2000
      ER = np.sum(W_Loss_t)/ self.LL_Loss
      m1 = 1.1 if ER>1e-4 else 1.04 if ER>2e-5 else 1.01
      if it<=20000 and self.load_w_t == False: m1 = 1.13
      if it>0: self.LL_Loss = self.LL_Loss * m1**5  # each 20K iterations
      R_L = np.sum(W_Loss_t)/ self.LL_Loss
      self.max_lr, self.base_lr = tf.reduce_min([R_L,6e-4]) , tf.reduce_min([R_L/10,8e-5])
      tf.print(f'****it is {it} |  max_lr is {self.max_lr} and min_lr is {self.base_lr}********')
      
  #%%-------------------------------------------------------------------------#    
  def alpha_mine(self, W_Loss_t, it, stopping_run):
      #%% Define constants
      max_opt = 22000 if self.Is_blood else 24000
      ER = np.sum(W_Loss_t)/self.LL_Loss
      m1 = 1.1 if ER>1e-4 else 1.04 if ER>1e-5 else 1.01 
      m2 = 2000 if ER>8.5e-6 else 4000
      m3 = 1.13

      decay_rates = [1e-3, 9e-4, 8e-4, 7e-4, 6e-4, 5e-4, 4e-4, 3e-4, 2e-4, 
                     1e-4, 9e-5 ,8e-5, 7e-5, 6e-5, 5e-5, 4e-5, 3e-5, 2e-5, 
                     1e-5, 9e-6, 8e-6, 7e-6, 6e-6, 5e-6, 4e-6, 3e-6]
      #%%----------------------------initialization--------------------------#
      self.step_LR_pre = self.step_LR
      here=False
      #%%---------------------------------------------------------------------#
      if stopping_run: 
         self.step_LR = 20
      else:   
         if it%2000==0 and it<max_opt and self.load_w_t==False:
            self.LL_Loss = self.LL_Loss*m3
         elif it%m2==0:
            self.LL_Loss = self.LL_Loss*m1
         #-----------------------------------------------------#   
         R_L = np.sum(W_Loss_t)/ self.LL_Loss
         LR_prpos=np.abs(np.array(decay_rates)-R_L)
         if it%10000==0 and it>30000 and self.optimizer_method.learning_rate<1e-4:
               LR_prpos=np.abs(np.array(decay_rates)-2.5*R_L)
               here=True
         #-----------------------------------------------------------#
         try: self.step_LR = np.where(LR_prpos==np.min(LR_prpos))[0][0] 
         except: self.step_LR = 20 # 8e-6
   
         if abs(self.step_LR_pre-self.step_LR)>2 and here==False:#3100:#2100: # more than two step
             if self.step_LR_pre-self.step_LR<0: self.step_LR = self.step_LR_pre+2 #1500
             else: self.step_LR = self.step_LR_pre-2#1500                  
             if it>20000 and self.step_LR < 3:self.step_LR = 3
         self.max_lr, self.base_lr = decay_rates[self.step_LR],decay_rates[self.step_LR] 
         self.step0.assign(0)
  #%%-------------------------------------------------------------------------#    
  def Error_estimate(self,it):
     
      uyz_pred = self.merged_model(inputs = self.Uyz_comsol['X_estimate'])
      usurf_pred = self.merged_model(inputs = self.Usurf_comsol['X_estimate'])
      #-----------------------------------------------------------------------%
      MeanSE_yz = tuple(tf.reduce_mean(tf.square(uyz_pred[i]-self.Uyz_comsol['U_exact'][i])) 
                                    for i in range(3))
      
      MeanSE_surf = tf.reduce_mean(tf.abs(usurf_pred[0]-self.Usurf_comsol['U_exact']))

      with self.writer_train.as_default():
          tf.summary.scalar(name ='E_surfmean', data = MeanSE_surf, step=it)

          for i in range(3):
             tf.summary.scalar(name ='E_yzmean'+str(i), data=MeanSE_yz[i], step=it)
#-----------------------------------------------------------------------------#

