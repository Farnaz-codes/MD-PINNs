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
#import tensorflow_addons as tfa
import numpy as np
import time
import pickle as pkl
from Fun2Call_newG_Loni import NonDim_param_symmetry_3Blood_NT_Pvar as NDparam
from Fun2Call_newG_Loni import PINN_model 
from Fun2Call_newG_Loni import MCMH_data_selection_horovod_wall_tumor_DOE as mcmh_fun
from Fun2Call_newG_Loni import Q_laser_func as Qs
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
loss_set = {'1_2': [0,1,3,4,6,7,9,10,12,13,15,16,18,19,21],
            '3':[2,5,8,11,14,17,23],'2_3':[20,22]}

Is_local_w = False
P_variable = True
SGD = 0
DTYPE='float32'
DTYPE_np = np.float32
keras.backend.set_floatx('float32')

dTb_Tt = 2/8  
P_range = [1.15,1.4]  
#P_range = [1, 1.5]
dP = P_range[1] -P_range[0]
Pmin = P_range[0]  # W/cm
m_dir = [2, 0, 2, 2, 0, 2]
#-----------------------------------------------------------------------------#
class PINNs_tf2_Power():
  # Initialize the class
  # 1 - input_args = {'args_tissue':args_tissue,'args_blood': args_blood}
  #     args_tissue = dict(['X0', 'Xf', 'X_lb', 'X_ub', 'Y_lb_ub', 'Z_lb', 'Z_ub'])
  #     args_blood = (Xend, xwall)
  # 2 - Tissue name: ['Skin_1st','Skin_2nd','Tumor1','Tumor2',
  #                   'Gold_Shell', 'Skin_3rd','Skin_3rd_blood']
  # 3 - layers : {'Skin': [5, 80, 80, 80, 80, 80, 1], 'Blood': [3, 20, 20, 20, 20, 1]}
  # 4,5 - lb, ub: dict containing non-dim lb and ub of size 5 [x, y, z, t, P]
  # 10,11 - path_ , path_T0 :The pathes the models were saved and should be loaded now 
  # remove_d =[4] for removing p and [4,3] for ignoring time and p
  def __init__(self, input_args, Tissues_name, layers, lb, ub, blood_band,
               Is_segments = (False,False), new_case = False, load_w_data = (False,False),
               path_ = [], path_T0 = [], ntime = 1, T0 = [], minpoints = 504, C_band = [],
               it_Stop=100000 ,lr_cycle_start = 0, P0 = 0.75, LR_method ='cycle',
               act_func = 'tanh', opt='adam', path_blood=[]):
    # 1) inputs
    args_tissue, args_blood = input_args['args_tissue'], input_args['args_blood']
    self.LR_method, self.act_func = LR_method, act_func
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
    
    self.D1_shape = int(minpoints/hvd.size())  #Size of datasets in each batch
    self.d3 = layers['Skin'][0]
    self.layers = layers
    self.it_Stop = it_Stop
    #-----------------Tissues properties and initialization-------------------#
    # Parameters and new conditions for PDE, Q of laser and initial Temp
    self.ND = {key: NDparam.NonDim_param(Skin_layer=key,tn =self.Time[1]-self.Time[0],
                                         remove_d =[]) for key in self.Tissues_name}   
    if new_case:
        self.Is_dTsquare, self.Is_Scattering = True, True
        self.sigma = 0.04
    else:
        self.Is_dTsquare, self.Is_Scattering = False, False
        self.sigma = 0.01
    
    self.T_inf = self.ND['Skin_1st'].ND_T(34 + 273.15)  
    self.T_in = self.ND['Skin_1st'].ND_T(35 + 273.15, blood = self.Is_blood)
    if self.Is_Scattering:  self.qs = Qs.Q_laser(Is_tumor = self.Is_tumor)
    # in qs: N = 1 means fv = 7e9 (1/cm3) density of Gold-nanoRods N=10 -->7e10
    #-------------------------------------------------------------------------#
    # 3) load the Model of previoys time step as initial conditions for the next time step
    keras.backend.clear_session()
    if path_T0:
        self.merged_model_T0 = keras.models.load_model(path_T0+'/tissue',compile=False)
    #-------------------------------------------------------------------------#
    # Load/create the tissue's and blood vessel's models   
    if not self.load_data: self.Tissues_DataSet(args_tissue)
    #-------------Blood properties and initialization-------------------------#
    if self.Is_blood:
        self.blood_name = [i for i in range(len(args_blood[1]))]
        self.model_blood ={}   # Blood vessels and model        
        self.Wall_band = {'ub': blood_band['ub'],'lb': blood_band['lb']} 
        Mid_point = (blood_band['lb'][2,0] + blood_band['ub'][2,0])/2
        self.in_b = [blood_band['ub'][i ,m_dir[i]] for i in range(3)]+\
                               [blood_band['lb'][i ,m_dir[i]] for i in range(3,6)]
                               
        self.out_b = [blood_band['lb'][i,m_dir[i]] for i in range(3)]+\
                               [blood_band['ub'][i ,m_dir[i]] for i in range(3,6)]
        self.out_b[1] , self.in_b[4] = Mid_point, Mid_point  
                     
        if not self.load_data: self.Blood_dataset(args_blood)
    #%%-------------------------------------------------------------------------#
    self.wall_group = [[0,1,2],[3],[4,5,6,7],[8],[9,10,11,12],[13,14,15],
                       [16],[17,18,19,20],[21],[22,23,24,25]]
    # 2) initialize the global weights | load the global weights
    if self.load_w_t==False:
        Loss_num_b =  13 #+ 6  # 6 for sloperecovery
        Loss_num_t =  24 + 31  if self.Is_tumor else 24 #
        Loss_num_wall = len(self.wall_group)  # group walls together 
        
        #self.w_t = [tf.constant(1.0, dtype= DTYPE) for _ in range(Loss_num_t)]
        self.w_t = [ tf.Variable(1.0, dtype = DTYPE) for ii in range(Loss_num_t)]   # dWPINN
        if self.Is_blood:
            #self.w_b = [tf.constant(1.0, dtype= DTYPE) for ii in range(Loss_num_b)]
            self.w_b = [tf.Variable(1.00, dtype = DTYPE) for ii in range(Loss_num_b)]   # dWPINN
            #self.w_wall = [tf.constant(1.0, dtype= DTYPE) for ii in range(Loss_num_wall)]
            self.w_wall = [tf.Variable(1.00, dtype = DTYPE) for ii in range(Loss_num_wall)]
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
    if path_:#['3rd']:  
       #W_load1 = tf.keras.models.load_model(path_['1st']+'/tissue', compile=False).get_weights()[:24] 
       #W_load2 = tf.keras.models.load_model(path_['1st']+'/tissue', compile=False).get_weights()[12:24] 
       W_load = tf.keras.models.load_model(path_+'/tissue', compile=False).get_weights()#[24:]
       #W_load = W_load1 + W_load3#+ W_load3 
       print('*************** Tissues Model is loaded*******************')
       self.merged_model.set_weights(W_load)
    #-----------------------------------------------------------------#
    if self.Is_blood:
       INPUTS_, OUTPUTS_ = () ,()
       for ii in self.blood_name:
          print('key is: ' ,ii)
          # Initialize models for blood vessels   
          #layers['Blood'] = [3, 45, 45, 45, 45, 1]
          input_ = tf.keras.Input(shape = (layers['Blood'][0],), dtype = DTYPE)                
          Bands_ =(blood_band['lb'][ii:ii+1 ,[m_dir[ii],3]], 
                      blood_band['ub'][ii:ii+1 ,[m_dir[ii],3]])
          model_blood = PINN_model.PINN_model(act_func, layers['Blood'], LBUB = Bands_)
             
          INPUTS_ += (input_ ,)
          output_ = model_blood(input_)
          OUTPUTS_ += (output_ ,)
       self.merged_model_blood = tf.keras.Model(inputs = INPUTS_ , outputs = OUTPUTS_ )
       
       try:
          print('***************Blood Model is loaded*******************')
          self.merged_model_blood.set_weights(tf.keras.models.load_model(path_blood+'/blood', \
                                                      compile=False).get_weights())

       except: tf.print('********The blood models could not be loaded*******')
     
    #%%-----------------------------------------------------------------------#
    # Define the cyclic learning rate schedule
    # https://github.com/bckenstler/CLR?ref=jeremyjordan.me
    self.step0 = tf.Variable(0, trainable=False) 
    # lr_cycle_start = 0 if len(self.path_)==0 else 2 if self.load_w_t else 1
    lr_cycle = [(0.004, 8e-4),(8e-4, 1e-4),(2e-4, 6e-5),(5e-5, 2e-5),(8e-6, 9e-6)]
    self.max_lr, self.base_lr = lr_cycle[self.lr_cycle_start]
    self.learning_rate_schedule = MyLRSchedule_Cycle2.MyLRSchedule_Cycle(LR_method =self.LR_method)
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
                                                         
    # Global weights optimizer  lr_wt = init_lr *0.97**(it/1000)

    #self.optimizer_method.build(self.train_vars())
    #init_lr = 0.001 if self.load_w_t else 0.01
    #self.lr_wt = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = init_lr, 
    #                                                       decay_steps=1000, decay_rate=0.99)
    
    self.optimizer_Wt = tf.keras.optimizers.Adam(learning_rate = 0.005)#self.lr_wt) 
    #%%-----------------------------------------------------------------------#
    # 6) Conceret Functions and graphs
    if self.Is_blood:   
        X_blood_sign = (tf.TensorSpec([None,self.d3-2], DTYPE), tf.TensorSpec([None,self.d3-2], DTYPE),
                        tf.TensorSpec([None,self.d3-2], DTYPE), tf.TensorSpec([None,self.d3-2], DTYPE),
                        tf.TensorSpec([None,self.d3-2], DTYPE), tf.TensorSpec([None,self.d3-2], DTYPE))
    
        self.net_BloodMerged_conceret =self.net_BloodMerged.get_concrete_function(X = X_blood_sign)
  #%%------------------------------------------------------------------------#
  # Create Data Sets
  def Blood_dataset(self, args_blood=[], rank_ = 0,data_path=[]): 
    if self.load_data:
        with open(data_path+'/newBloodData_MCMH'+str(rank_)+'_Rank.pkl' , 'rb') as f:
            new_data = pkl.load(f)
        self.train_dataset_blood = tf.data.Dataset.from_tensors(new_data['X_blood'])  #from_tensors
    else:
        Xend, Xwall = args_blood
        #Dw_shape = int(Xwall[0][0].shape[0])
        xwall_out = tuple(self.fd(xwall_i) for xwall_i in Xwall)
        self.train_dataset_blood = tf.data.Dataset.from_tensors((self.fd(Xend), xwall_out))
        #self.train_dataset_blood =  train_dataset_.batch(Dw_shape, drop_remainder=True)
    self.train_dataset_blood.prefetch(tf.data.AUTOTUNE)
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
        
        if 'Xfmoreb' in args_tissue.keys(): dataset+=(self.fd(args_tissue['Xfmoreb']),)
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
        try:
           u_pred = self.merged_model_T0(input_all)
        except:
           u_pred = self.merged_model_T0(tuple(xi[:,:-1] for xi in input_all))

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
        if mixed < 3 or self.Is_dTsquare:
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
        if self.Is_dTsquare:
           T2_gradt = tape2.gradient(Tt, X)
           Ttt = [T2_gradt[j][:,3:4] for j in range(len(T2_gradt))]
        del tape2, tape1
        if self.Is_dTsquare: return T,Tt,Txx,Tyy,Tzz,Ttt 
        else: return T,Tt,Txx,Tyy,Tzz,()
        
    elif mixed==0: return T,Tx,Ty,Tz,Tt # otherwise mixed == 0
    else: 
        if self.Is_dTsquare: return T, Tt  # for IC mixed = 3
        else: return T, ()
  #--------------------------------------------------------------------------#
  @tf.function
  def net_BloodMerged(self, X):#, mixed = tf.constant(0)):
    # inputs and outputs are tuples
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(X)
        # we have 3 inputs [lenght , time, Power] and one output [T]
        T = self.merged_model_blood(inputs = X) # u_ is a list with size of 6
    T_grad = tape.gradient(T, X)
    T_d = [T_grad[j][:,0:1] for j in range(len(X))]
    return T, T_d
  
  #%% *********************** Skin Tissues Losses ****************************#
  #***************************************************************************#
  def ft(self, x, n = 0):
     if n==0: n=self.d3-1
     return tuple(tf.reshape(i,(-1,n)) for i in x)
  #%%--------------------------------------------------------------------------#
  def Loss_Skin_tissues(self, X_dist_batch): 
     # Nt = len(self.Tissues_name) 
     if len(X_dist_batch)==8:
         x0, x_f, x_lbx, x_ubx, x_lb_uby , x_lbz, x_ubz, T0 = X_dist_batch
     if len(X_dist_batch)==9:
         x0, x_f, x_lbx, x_ubx, x_lb_uby , x_lbz, x_ubz, T0,xmore = X_dist_batch
         x_f = tuple(tf.concat([x_f[i],xmore[i]],axis=1) for i in range(len(xmore)))
     #------------------------------------------------------------------------#     
     T00 = tuple(tf.reshape(i,(-1,1)) for i in T0)
     
     Loss_f = self.Skin_Loss_PDE(self.ft(x_f,self.d3)) #--> Nt losses
     # Losses on BC x-y-z-dir, IC, PDE, and IFC 
     Loss_0 = self.Skin_Loss_IC(self.ft(x0), T00) #--> Nt losses

     Loss_xub = self.Skin_Loss_symmetry_Xub(self.ft(x_ubx))  # -> len(Tissues) losses
     #------------------------#
     # Loss_x for ['Skin-1st - Skin-3rd] -->3 (3)
     # LossT_x and LossdT_x for IFC [tumor1-2nd, tumor2-3rd, gold-tumor] --> 6
     Loss_xlb, LossT_xlb, LossdT_xlb = self.Skin_Loss_BCXY(self.ft(x_lbx), 'lb', col = 0)

     Y = self.ft(x_lb_uby)
     # Loss_y for ['Skin-1st - Skin-3rd ]--> 3
     # LossT_y and LossdT_y for IFC [tumor1-2nd, tumor2-3rd, gold-tumor] --> (6) 
     Loss_ylb, LossT_ylb, LossdT_ylb = self.Skin_Loss_BCXY(Y, 'lb', col = 1)
     Loss_yub, LossT_yub, LossdT_yub = self.Skin_Loss_BCXY(Y, 'ub', col = 1)
     #Loss_y = [Loss_ylb[i] + Loss_yub[i] for i in range(len(Loss_yub))]
     LossT_y = [LossT_ylb[i] + LossT_yub[i] for i in range(len(LossT_ylb))]
     LossdT_y = [LossdT_ylb[i] + LossdT_yub[i] for i in range(len(LossdT_ylb))]
     #------------------------#
     # loss0 (1) + IFC (2)|(3)|(7)|(8) + UBz skin_3rd (2 with blood) or (1))-->(4)|(6)|(9)|(11)
     Loss_z = self.Skin_Loss_BCZ(self.ft(x_lbz), self.ft(x_ubz))
     
     #----------# Loss_f + Loss_0 + Symmetery + LossX_Y insulation -> 3*Nt + 6
     # Nt(Lf) + Nt(L0)+ Nt(L_ubx) + 3[L_lbx] + 3[Ly] + 1 [L_z0]
     Loss_T = Loss_f + Loss_0 + Loss_xub + Loss_xlb[:3] + Loss_ylb[:3]   + Loss_yub[:3]    # Do the same for tumor ????
     if self.Is_tumor: Loss_T += LossT_xlb[:3] + LossdT_xlb[:3] + LossT_y[:3] + LossdT_y[:3] # 12   
          
     return Loss_T + Loss_z

  #%% Initial Conditions ( All self.Tissues_name members) 
  def Skin_Loss_IC(self,x0, T00, plt_loss=0): 
      #  X0 is 4D --> 0  should be added to column 3
      # IC  : U0 , dU0 
      U0_pred, U0t_pred = self.net_skinMerged(self.add_ConstColumn(x0, C=0, NCol=3), mixed=3)
      
      if self.Is_dTsquare: 
          Loss_0_data = [tf.square(U0_pred[ii]-T00[ii])+tf.square(U0t_pred[ii])
                         for ii in range(len(U0_pred))]
      else:
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
      Same_Domain = ['Tumor1','Tumor2','Gold_Shell']
      temp = tf.ones([3,self.d3], dtype=DTYPE)
      Xnew = ()
      #-----------------------------------------------------------------------#
      if len(Transform_to)>0:
          for ii, key_frwd in enumerate(self.Tissues_name):
              if key_frwd in Transform_to:
                  jj = Transform_to.index(key_frwd)
                  key_inv = Transform_from[jj]
                  if key_frwd==key_inv or (key_frwd in Same_Domain and key_inv in Same_Domain):
                      Xnew +=(X[jj],)
                  else:
                      Xnew += (self.ND[key_frwd].Forward_Var(self.ND[key_inv].Inverse_Var(
                              X[jj],t0 =self.Time[0]),t0 = self.Time[0]),)
              else:Xnew +=(temp,)
      return Xnew
  #---------------------------------------------------------------------------#    
  def transformation_IFC_Xy(self, X, col = 0):  # X is 5D (x,y,z,t,p) --> for BC on X, col=0, etc.
      # inputs X transform into their neighbors for IFC
      Transform_from = ['Tumor1','Tumor2','Gold_Shell','Gold_Shell']
      Transform_to = ['Skin_2nd','Skin_3rd','Tumor1','Tumor2']

      x_from = tuple(X[self.Tissues_name.index(ii)] for ii in Transform_from)
      Xtransform = self.transform_fun(x_from, Transform_from, Transform_to)
     
      U,*Ugrads = self.net_skinMerged(Xtransform, mixed = 0)

      U_neighbor = {Transform_from[ii]: (kk, U[self.Tissues_name.index(kk)],
                      Ugrads[col][self.Tissues_name.index(kk)]) for ii,kk in enumerate(Transform_to[:2])}
      #---------------------------------------------------------------#
      ## concat the gold-Shells between Tumor1 and Tumor2
      g_ind, t1_ind = self.Tissues_name.index('Gold_Shell'),self.Tissues_name.index('Tumor1'),
      t2_ind = self.Tissues_name.index('Tumor2')
      ug = tf.where(X[g_ind][:,2:3]<0.5, U[t1_ind], U[t2_ind])
      ug_grad = tf.where(X[g_ind][:,2:3]<0.5, Ugrads[col][t1_ind], Ugrads[col][t2_ind])
      U_neighbor['Gold_Shell'] = ('Tumor1', ug, ug_grad)
                                    
      return U_neighbor  
  #--------------------------------------------------------------------------#
  def Skin_Loss_BCXY(self,  X , Bound ='lb', col = 0, plt_loss = 0): 
      #  X is 4D --> C_band[key][0,col]  should be added to column col 
      Loss_bc, LossT_IFC, LossdT_IFC = [] ,[] ,[]
      C_band = self.LB_t if Bound =='lb' else self.UB_t     
      if plt_loss==0:
          XD = self.add_ConstColumn(X, C=[C_band[ky][0,col] for ky in self.Tissues_name] ,NCol = col)
      else: 
          XD = tuple(self.add_ConstColumn(xi, C= C_band[self.Tissues_name[ii]][0,col] ,NCol = col)[0]\
                 if len(xi)>0 else tf.ones([2,self.d3], dtype=DTYPE) for ii,xi in enumerate(X))

      u_pred, *ud_pred = self.net_skinMerged(XD, mixed = 0)
      
      if self.Is_tumor:  # if Is_tumor
            u_neghbor = self.transformation_IFC_Xy(XD, col = col) # {key:(u, gradu) ,...}
      else: u_neghbor = {}
      #----------------------------------------------------------------------#
      for ii , key in enumerate(self.Tissues_name):
          if key in u_neghbor.keys():  # IFC Boundary conditions
              Ci = self.ND[key].param.kl()/self.ND[key].Char_Len()[col],
              Cj = self.ND[u_neghbor[key][0]].param.kl()/self.ND[u_neghbor[key][0]].Char_Len()[col]
              if plt_loss==0:        
                  LossT_IFC.append(tf.reduce_mean(tf.square(u_pred[ii]-u_neghbor[key][1])))
              
                  LossdT_IFC.append(tf.reduce_mean(tf.square(Ci*ud_pred[col][ii]-Cj*u_neghbor[key][2]))) 
              else:
                  LossT_IFC.append(tf.square(Ci*ud_pred[col][ii]-Cj*u_neghbor[key][2])+
                                          tf.square(u_pred[ii]-u_neghbor[key][1]))
          #----------------------------------------------        
          elif plt_loss==0:  # insulation conditions
              Loss_bc.append(tf.reduce_mean(tf.square(ud_pred[col][ii])))
          
      if plt_loss==0: return Loss_bc, LossT_IFC, LossdT_IFC
      else: return LossT_IFC
                                                             
  #%%Boundery and interfacial Conditions on z-dir and their losses
  def Skin_Loss_BCZ(self, z_lb, z_ub, plt_loss=0):  # Z is 4D (x,y,z,t,p) 
 
      Zlb =self.add_ConstColumn(z_lb, C=[self.LB_t[ky][0,2] for ky in self.Tissues_name] ,NCol = 2)
      
      Zub = tuple(self.add_ConstColumn(zi, C= self.UB_t[self.Tissues_name[ii]][0,2] ,NCol = 2)[0]\
                          if len(zi)>0 else zi for ii,zi in enumerate(z_ub))
      
      # to make the code runs faster, lb and ub are combined first and 
      # after estimation they will separted vs their neighbors
      Ci = [self.ND[ky].param.kl()/self.ND[ky].Char_Len()[2] for ky in self.Tissues_name]
      i3, i2 = self.Tissues_name.index('Skin_3rd'), self.Tissues_name.index('Skin_2nd')
      if self.Is_tumor:
         it2, it1 = self.Tissues_name.index('Tumor2'), self.Tissues_name.index('Tumor1')
         ig = self.Tissues_name.index('Gold_Shell')
      # function estimation for LB
      U_lb,_, _, Ud_lb, Ulbt = self.net_skinMerged(Zlb, mixed = 0) 
      # function estimation for UB
      Zub_tissue = self.transform_fun([Zlb[i2], Zlb[i3]],['Skin_2nd','Skin_3rd'], 
                                                         ['Skin_1st','Skin_2nd'])
      Zub_new = (Zub_tissue[0],)
      for ii in range(len(Zub)-1):
          if self.Tissues_name[ii+1]=='Tumor1': Zub_new +=(Zlb[it2],)
          elif self.Tissues_name[ii+1]=='Skin_2nd': Zub_new +=(Zub_tissue[1],)
          else: Zub_new +=(Zub[ii+1],)
          
      U_ub,_, _, Ud_ub,_ = self.net_skinMerged(Zub_new, mixed = 0) 
      #---------------------------------------------------------------------#     
      U_IFC = (((U_ub[0],Ci[0]*Ud_ub[0]) ,(U_lb[i2],Ci[i2]*Ud_lb[i2])),  #'Skin_2nd & 1st
               ((U_ub[i2],Ci[i2]*Ud_ub[i2]) ,(U_lb[i3],Ci[i3]*Ud_lb[i3]))) #'Skin_2nd& 3rd
      
      if self.Is_tumor:
          # function estimation for neighbors
          Transform_from = ['Tumor1','Tumor2','Gold_Shell','Gold_Shell']
          Transform_to = ['Skin_2nd','Skin_3rd','Tumor1','Tumor2']
          Z_neighbor = (Zlb[it1], Zub[it2], Zlb[ig], Zub[ig])
          Z_neighbors = self.transform_fun(Z_neighbor,Transform_from, Transform_to)
          U_n,_, _, Ud_n,_ = self.net_skinMerged(Z_neighbors, mixed = 0)
          #-------------------------------------------------------------------#
          U_IFC += (((U_n[i2],Ci[i2]*Ud_n[i2]),(U_lb[it1],Ci[it1]*Ud_lb[it1])),  #'Tumor1'-Skin_2nd
                    ((U_ub[it1],Ci[it1]*Ud_ub[it1]),(U_lb[it2],Ci[it2]*Ud_lb[it2])),  #'Tumor2,'Tumor1
                    ((U_n[it1],Ci[it1]*Ud_n[it1]) ,(U_lb[ig],Ci[ig]*Ud_lb[ig])),  #'Tumor1'-Gold
                    ((U_n[i3],Ci[i3]*Ud_n[i3]) ,(U_ub[it2],Ci[it2]*Ud_ub[it2])),   #'Tumor2'-Skin_3rd
                    ((U_n[it2],Ci[it2]*Ud_n[it2]) ,(U_ub[ig],Ci[ig]*Ud_ub[ig])))  #'Tumor2'-Gold                                    
      #---------------------------Loss_estimations-----------------------------# 
      LossT_IFC, LossdT_IFC, Lossz =[], [], []   
                
      if self.Is_dTsquare:  #  Consider delay BC for air
          if plt_loss==0:
              Lossz0 = [40*tf.reduce_mean(tf.square(Ud_lb[0]-self.ND['Skin_1st'].Bi1*(
                         U_lb[0]- self.T_inf + self.ND['Skin_1st'].Rt*Ulbt[0])))]
          else: LossT_IFC.append(tf.square(Ud_lb[0]-self.ND['Skin_1st'].Bi1*(
                                U_lb[0]- self.T_inf + self.ND['Skin_1st'].Rt*Ulbt[0])))
      else:
         if plt_loss==0:Lossz0 = [60*tf.reduce_mean(tf.square(Ud_lb[0]-self.ND['Skin_1st'].Bi1*(U_lb[0]- self.T_inf)))]
         else: LossT_IFC.append(tf.square(Ud_lb[0]-self.ND['Skin_1st'].Bi1*(U_lb[0]- self.T_inf)))
      #-----------------------------------------------------------------------# 
      w = [2,2,1,6,1,2,2,2] if self.Is_dTsquare else [7,7,4,1]
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
    # Is_local_w : True | False :  If local weights applied
    Loss_f , Loss_f_data = [] ,[]
    T, Tt, Txx, Tyy, Tzz, Ttt = self.net_skinMerged(x, mixed = 1) 
    wf = [15, 30, 15, 25,10, 30] if self.Is_tumor else [20, 20, 10]#wf = [2,2,6,8]
    
    for j,key in enumerate(self.Tissues_name):
        ND = self.ND[key]
        Tavr = self.T_avr_output_of_Blood(x[j])
        Q = self.Q_estimation(x[j], ND, key)
        # physics informed
        if self.Is_dTsquare:
            f_u = (ND.A*ND.Rt+1)*Tt[j] + ND.Rt*Ttt[j] - ND.F0[0]*Txx[j] - ND.F0[1]*Tyy[j] -\
                              ND.F0[2]*Tzz[j] - ND.A*(Tavr-T[j]) - Q/ND.Q_Coeff()
        else:
            f_u = Tt[j] -ND.F0[0]*Txx[j]- ND.F0[1]*Tyy[j]- ND.F0[2]*Tzz[j]-\
                              ND.A*(Tavr-T[j])- Q/ND.Q_Coeff()                              
        # 1 - MSE PDEs for 3 skin_layers
        Loss_f_data.append(tf.square(f_u))
        if plt_loss==0:
            Loss_f.append(wf[j]*tf.reduce_mean(Loss_f_data[j]))
                
    if plt_loss: return Loss_f_data
    else: return Loss_f    # it is a list
  #---------------------------------------------------------------------------#
  def T_avr_output_of_Blood(self, x_f = []):
    # Estimating u_out at the exit of Arterial blood vessels 
    if self.Is_blood:
        # I need 6 inputs here, I just need the 3rd one -->[2]
        x_out = tf.concat([self.out_b[2] * tf.ones([x_f.shape[0],1],dtype=DTYPE),x_f[:,3:]],1)
        temp = tf.ones([4,self.d3-2], dtype=DTYPE)
        Xin = (temp, temp, x_out, temp, temp, temp)
        Tout,_ = self.net_BloodMerged_conceret(Xin)
        Tavr = Tout[2] * dTb_Tt   
    else:
        Tavr = self.T_in
    return Tavr
  # ------------------------------------------------------------------------#
  def Q_estimation(self, x, ND, Skin_layer =[]):
      alpha , Reff = 1.8, 0.1     
      Gauusian_xy = tf.exp((tf.square(ND.Inverse_Var(x[:,0:1],d='x')-0.5)+
                    tf.square(ND.Inverse_Var(x[:,1:2],d='y')-0.5))/\
                           (-2*self.sigma**2))/tf.sqrt(2*np.pi*self.sigma**2)

      if self.Is_Scattering: 
          Q = self.qs.Qz(x[:,2:3], Skin_layer)  
             
      else: Q = alpha *tf.exp(-alpha*ND.Inverse_Var(x[:,2:3],d='z'))
      
      if P_variable:return Q*(x[:,4:5]*dP + Pmin)*(1-Reff)*Gauusian_xy  #
      else:return Q*self.P0*(1-Reff)*Gauusian_xy  # Iin = P0 *(1 - Reff)     

 #%% *********************** Blood Vessels Losses ****************************#
  #***************************************************************************#
  #%%Initial Conditions and BC on Xout and Xin (inside the blood vessels)
  def Loss_Blood_vessels(self, X_dist_Blood=[]): # blood vessels are located in the "Skin-3rd"
    # xend: (Arterial-out-2(0), Venus-in-2(1))
    xend, xwall = X_dist_Blood
    #-------------------------------------------------------------------------#
    # Wall Data --> xbs, uwall at skin(xbf) and u wall at blood(xbf)
    loss_walls, xb_f, uwall_skin, uw_blood, duw_blood,_ = self.Loss_wall(xwall)
    
    # Estimating the loss values
    loss_f = self.blood_PDE(uwall_skin, uw_blood, duw_blood)    #losses [0-5]
    # Create other Datasets here    
    loss_in, LossTout, lossSym = self.Blood_Loss_IC_BCL(self.ft(xend,self.d3))     
    Losses = loss_f + loss_in + LossTout  #+ lossSym # + loss_endx

    return Losses, loss_walls
  #---------------------------------------------------------------------------#      
  def create_data_in_out(self,xend):
      #LB, UB = self.blood_band['lb'][0][0,1],self.blood_band['ub'][0][0,1]
      #time_rand = LB + (UB-LB)*tf.random.uniform([N0,1],dtype =DTYPE)
      N0 =  xend[0].shape[0]
      t_P_rand = xend[0][:,3:]
      
      Xend = tuple(tf.concat([xx[:,:3],t_P_rand], axis =1) for xx in xend)
      Xin = tuple(tf.concat([xin *tf.ones([N0,1],dtype =DTYPE),t_P_rand], axis =1) for xin in self.in_b) 
      Xout = tuple(tf.concat([xout *tf.ones([N0,1],dtype =DTYPE),t_P_rand], axis =1) for xout in self.out_b)
      return Xin,Xout,Xend
  #---------------------------------------------------------------------------#   
  def Blood_Loss_IC_BCL(self, Xend): 
      xb_in, xb_out , xend= self.create_data_in_out(Xend)

      ubin_pred, ubin_l_pred = self.net_BloodMerged_conceret(xb_in)
      ubout_pred, ubout_l_pred = self.net_BloodMerged_conceret(xb_out)
      # Loss 
      # 1 - symmetry loss : in blood levels 1 : input for Arterial and output for venus
      lossSym = tf.reduce_mean(tf.square(ubin_l_pred[1]))+\
                tf.reduce_mean(tf.square(ubout_l_pred[4]))          
      # output of Arterial an level 3 ==Temp output blood at level 3
      u3out_pred = self.net_skinMerged((xend[0],), mixed = 2, T_num = self.Tissues_name.index('Skin_3rd'))
      LossTout = tf.reduce_mean(tf.square(ubout_pred[2]*dTb_Tt - u3out_pred)) 
      #-----------------------------------------------------------------------# 
      # 2) loss_in: MSE BC in at l = 0 of Blood_vessels
      # Tin for blood vessels based on the Tout of previous vessels
      loss_in = []
      Tinput = tf.constant(self.T_in, dtype=DTYPE) 
      for ii in self.blood_name: 
          if ii== 0: 
              Tin = Tinput
          elif ii==5:# u5in_pred
              Tin = self.net_skinMerged((xend[1],), mixed = 2, T_num = self.Tissues_name.index('Skin_3rd'))
              Tin = Tin/dTb_Tt
          else:
              Tin = ubout_pred[ii-1] if ii<3 else ubout_pred[ii+1]
          loss_in.append(tf.reduce_mean(tf.square(ubin_pred[ii]-Tin)))

      return loss_in, [LossTout], [lossSym]
  #---------------------------------------------------------------------------#
  def blood_PDE(self, u_wall, ub, ub_dL,plt_loss=0):
    # u_Wall, ub, ub_dL : estimated temperatures in blood vessels and on their walls 
    Loss_f = []
    wb = [2,1,2,4,4,1]  # it was [2,1,2,6,3,1] before 15 seconds
    for j in self.blood_name:
        factorm = self.ND['Skin_3rd'].param.factor_blood(j)
        L0 = self.ND['Skin_3rd'].L0_blood_basedOn3rd(j)
        L = tf.constant(L0[1],dtype=DTYPE) 
        f_u_avr = ub_dL[j] - factorm[0] *L*(u_wall[j]/dTb_Tt - ub[j])

        if j ==2: #if (m =self.m_level-1 and key == 'Arterial')
            f_u_avr = f_u_avr -factorm[1]*L*(ub[j] - self.ND['Skin_3rd'].ND_T(0,blood=True))
        # 0 - Loss of PDE
        if plt_loss:
            Loss_f.append(tf.square(f_u_avr))
        else:
            Loss_f.append(wb[j]*tf.reduce_mean(tf.square(f_u_avr)))

    return Loss_f  # Loss_adpt_b[0-6]
  
  #%% *************** Blood Vessels | Tissues| wall Losses *******************#
  #***************************************************************************#
  def Loss_wall(self, xwall , plt_loss=0): #--> 26 losses for walls
      # blood vessels are located in the "Skin-3rd"
      uw_skin, duw_skin_Bi, xb_f, D = self.U_Wall_Skin(xwall)  # is a tuple
      uw_blood, duw_blood = self.net_BloodMerged_conceret(xb_f)
      
      # dT/dn *1/Bi + (Tw-Tb) = 0 --> Bi = Bi*L for lb and -Bi*L for ub
      if plt_loss:
          loss_f = self.blood_PDE(uw_skin, uw_blood, duw_blood, plt_loss=1)
          loss_walls = [loss_f[j] + tf.square(duw_skin_Bi[j]-
                        (uw_blood[j]*dTb_Tt - uw_skin[j])) for j in range(len(uw_blood))]
      #-----------------------------------------------------------------------#
      else:
        loss_walls_ = [tf.reduce_mean(tf.square(duw_skin_Bi[j][D[j][ii]:D[j][ii+1],:]-
                       (uw_blood[j][D[j][ii]:D[j][ii+1],:]*dTb_Tt - uw_skin[j][D[j][ii]:D[j][ii+1],:]))) 
                          for j in range(len(uw_blood)) for ii in range(len(D[j])-1)] 
        ww = [1,3,2,3,3,4,4,4,2,1] # new after 15 seconds (loss5 (1-->3)) (loss8 3-->5)
        loss_walls = [ww[i]*tf.reduce_sum([loss_walls_[j] for j in self.wall_group[i]])
                                          for i in range(len(self.wall_group))]
      return loss_walls, xb_f, uw_skin, uw_blood, duw_blood, D
  #---------------------------------------------------------------------------#
  def U_Wall_Skin(self, xwall):
    # BC on walls using the model of 'Skin_3rd' for six blood vessles 
    xwall = tuple(tuple(tf.reshape(xwall[j][i],(-1,self.d3)) for i in range(len(xwall[j]))) 
                                    for j in range(len(xwall)))  
    Dwall ,Bi, _ = self.ND['Skin_3rd'].Bi_blood_wall2(endx = True) 
    N_cum =[]
    for i in range(len(xwall)):
       wall_dim = tf.shape_n(xwall[i])
       dd = [int(jj[0]) for jj in wall_dim] 
       N_cum += [[sum(dd[:i]) for i in range(len(dd)+1)]]
    #-------------------------------------------------------------------------#
    xwall = tuple(tf.concat(xwi ,axis =0) for xwi in xwall)
        
    xb_f = tuple(tf.gather(params = xwall[i], indices = [m_dir[i],3,4]  if P_variable 
                else [m_dir[i],3], axis=1) for i in range(len(xwall)))    
     
    T_b, dT_b = (), ()
    i3 = self.Tissues_name.index('Skin_3rd')
    for wii,xw in enumerate(xwall):
        dT_all = ()
        T_all, *gard_T = self.net_skinMerged((xw,), mixed = 0, T_num = i3)
        for jj in range(len(Dwall[wii])):
            dT_all += (gard_T[Dwall[wii][jj]][i3][N_cum[wii][jj]:N_cum[wii][jj+1],:]/Bi[wii][jj],)
        T_b += (T_all[i3],)
        dT_b +=(tf.concat(dT_all, axis=0),)

    return T_b, dT_b, xb_f, N_cum
  #***************************************************************************#      
  def train_vars(self):
      if self.Is_blood:
         #return  {0: [self.merged_model.trainable_weights[i] for i in range(0,24)],
         #         1: [self.merged_model.trainable_weights[i] for i in range(24,36)] +\
         #                 self.merged_model_blood.trainable_weights,
         #         2: [self.merged_model.trainable_weights[i] for i in range(12,36)],
         #         3:  self.merged_model.trainable_weights + self.merged_model_blood.trainable_weights}
         
         return self.merged_model.trainable_weights + \
                 self.merged_model_blood.trainable_weights 
      else:
          return self.merged_model.trainable_weights 
  #%% ------------------------------------------------------------------------#
  ##**************************************************************************#
  @tf.function#(jit_compile = True)  
  def loss_fn_opt(self, X_dist_Tissue, X_blood = [], first_batch = False , opt ='sequential'):  
     with tf.GradientTape(persistent=True) as tape:
         Losses_T = self.Loss_Skin_tissues(X_dist_Tissue)
         W_Losses_T = [self.w_t[i]*Losses_T[i] for i in range(len(self.w_t))]
         #--------------------------------------------------------------------#
         if self.Is_blood:
            Losses_B, Losses_W = self.Loss_Blood_vessels(X_blood)
            W_Losses_B = [self.w_b[i]*Losses_B[i] for i in range(len(self.w_b))]
            W_Losses_W = [self.w_wall[i]*Losses_W[i] for i in range(len(self.w_wall))]
         else:
             W_Losses_B ,Losses_B, Losses_W, W_Losses_W = [], [], [], []

         W_Losses = W_Losses_T + W_Losses_B + W_Losses_W
         Losses = Losses_T + Losses_B + Losses_W
         wLoss = tf.reduce_sum(W_Losses)

     #-----------------------------------------------------------------------# 
     # Horovod: add Horovod Distributed GradientTape. --> Step 5
     tape = hvd.DistributedGradientTape(tape)
     grads = tape.gradient(wLoss , self.train_vars())
     #ind = {0:list(range(0,34)), 1: list(range(34,51)), 2:list(range(0,51))}
     #for ii in wLoss.keys():
     #   grads = tape.gradient(wLoss[ii] , self.train_vars()[ii])
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
  
  """
  @tf.function
  def dwPINN_update(self, Losses , weight_):
      Nw = len(weight_)
      #grads_wt = [-Losses[i] for i in range(Nw)]
      eta = 1500   # Learning rate
      for wi in range(Nw):
         weight_[wi].assign_add(eta*tf.reshape(Losses[wi]**2, [])) 
         if Losses[wi]<1e-6: weight_[wi].assign(1.0)
  """       
  #----------------------------------------------------------------------------#
  def main_Metropolis_Hasting_Alg(self, Xt ): 
      x0, x_f, x_lbx, x_ubx, x_lby , x_lbz, x_ubz, T0 = Xt
      
      x_f = self.Metropolis_Hasting_Alg_PDE(x_f)
      x0, T0 = self.Metropolis_Hasting_Alg_IC(x0,T0)
      x_ubx = self.Metropolis_Hasting_Alg_ubX(x_ubx)
      x_lbz, x_ubz = self.Metropolis_Hasting_Alg_z_IFC( x_lbz, x_ubz)      
      if self.Is_tumor:
          x_lbx, x_lby = self.Metropolis_Hasting_Alg_XY_IFC(x_lbx, x_lby)
      
      return (x0, x_f, x_lbx, x_ubx, x_lby , x_lbz, x_ubz, T0)

  #%%-------------------------------------------------------------------------#    
  # train loop    
  def train(self, save_path=[], max_Stop_err = 2.00e-5, 
            mcmh = False, data_path=[], LL_Loss =650 , opt = 'parallel',it_Stop_lbfgs=0,
            alpha_constant = 'False'): 
    #-------------------------------------------------------------------------#    
    # initializations
    if not save_path: save_path = self.path_
    if not data_path: data_path = self.path_T0
    it_Stop = self.it_Stop
    
    it , stopping_run   = 0, 0
    self.LL_Loss = LL_Loss
    self.losses_lbfgs = []
    
    region = 'Tumor_blood' if (self.Is_blood and self.Is_tumor) else 'Tumor' if self.Is_tumor else 'Skin-blood'
    #--------------------------- indices if Losses ---------------------------#
    Nt = len(self.Tissues_name)
    # f |0 | ubx | lbx |y
    Loss_ind = {'Skin_1st':[0, Nt, 2*Nt, 3*Nt, 3*Nt+3, 3*Nt+6]}
    Loss_ind['Skin_2nd'] = [i+1 for i in Loss_ind['Skin_1st']]
    Loss_ind['Skin_3rd'] = [i+1 for i in Loss_ind['Skin_2nd']]
 
    ind_n , Ifc_n = 3*Nt+8 ,6
 
    if self.Is_tumor: 
        Loss_ind['Tumor1'] = [3, Nt+3, 2*Nt+3, ind_n,ind_n+3,ind_n+6,ind_n+9]
        Loss_ind['Tumor2'] = [i+1 for i in Loss_ind['Tumor1']]
        Loss_ind['Gold_Shell'] = [i+1 for i in Loss_ind['Tumor2']]
        ind_n +=12
        Ifc_n +=10
        
    Loss_ind['IFC'] = list(range(ind_n, Ifc_n + ind_n))
     
    if self.Is_blood: 
        ind_b = Ifc_n+ind_n#+2
        b_in = 13
        Loss_ind['Blood_PDE'] = list(range(ind_b, 6 + ind_b))
        Loss_ind['Blood_in'] = list(range(6 + ind_b, 12 + ind_b))
        Loss_ind['Blood_out_sys_endx'] = list(range(12 + ind_b, b_in + ind_b))
        Loss_ind['Blood_wall'] = list(range(b_in + ind_b,b_in + ind_b + len(self.wall_group)))    
    #-------------------------------------------------------------------------#
    if hvd.rank()==0:
        self.writer_train = tf.summary.create_file_writer(save_path + 'train')
        os.makedirs(save_path,exist_ok=True)
        start_time = time.time()
    #-------------------------------------------------------------------------#
    # 4- Distribute Data and initialize NN
    for i in range(hvd.size()):
        if hvd.rank()==i:
            if self.load_w_t:#self.load_data: 
                with open(data_path+'/Power'+str(hvd.rank())+'_Rank.pkl','rb') as f:
                      P_W_Data = pkl.load(f)
                  
                Losses = P_W_Data['Loss_far']
                self.w_t= P_W_Data['wL']['wt']
                if self.Is_blood:
                    self.w_wall, self.w_b = P_W_Data['wL']['wwall'], P_W_Data['wL']['wb']
            #-----------------------------------------------------------------#
            if self.load_data:
                self.Tissues_DataSet(rank_ = i, data_path=data_path)
                XT_tf = next(iter(self.train_dataset)) 
            else:
                XT_tf = next(iter(self.train_dataset.skip(i).take(1)))
            #-----------------------------------------------------------------#    
            print(f'************I am in rank {i}********************')
            if self.Is_blood:
                if self.load_data:
                    self.Blood_dataset(rank_ = i,data_path=data_path)
                XB_tf = next(iter(self.train_dataset_blood))
            else: XB_tf = [] 
    #-----------------------------------------------------------------#
    self.step_LR = int(2*(self.lr_cycle_start+4))
    Loss_overal = 2 
    weight_ = self.w_t + self.w_b + self.w_wall  if self.Is_blood else self.w_t
    it_stop = min(60000,int(it_Stop*0.45))
    Loss_num = len(weight_) 
    #---------------------------------------------------------------------#
    while stopping_run<=10002 and it<it_Stop :
   
      if it==0: 
          Loss, W_Loss = self.loss_fn_opt(XT_tf, XB_tf, first_batch = True, opt = opt) 
      else: 
          Loss, W_Loss = self.loss_fn_opt(XT_tf, XB_tf, first_batch = False, opt = opt) 
      #----------------------------------------------------------------------#
      if it<it_stop and Loss_overal<0.2 and hvd.rank()==0:
         if mcmh and it%200==0: 
            XT_tf = self.main_Metropolis_Hasting_Alg(XT_tf)
            if self.Is_blood:
                XB_tf = self.Metropolis_Hasting_Alg_Wall(XB_tf)
      #---------------------------------------------------------------------#
      Losses = np.reshape(Loss,(Loss_num,1))
      W_Loss_t = np.reshape(W_Loss,(Loss_num,))
      Loss_overal = hvd.allreduce(np.sum(Losses), average=True)              
      #%%--------------------------------------------------------------------#
      # weight optimizations 
      if (it>self.it_Stop*0.1 and Loss_overal<0.1) or self.load_w_t: 
          if it%2==0:# and it<self.it_Stop*0.8: 
             self.dwPINN_update(tf.convert_to_tensor(Losses), weight_) 
             if it>self.it_Stop*0.25:
                for wi in range(Loss_num):
                   if weight_[wi]>5 and Losses[wi]<1e-5:weight_[wi].assign_sub(5e-7)
                   elif weight_[wi]>10 and Losses[wi]<5e-5:weight_[wi].assign_sub(2.5e-7)
                   elif Losses[wi]>1e-3:weight_[wi].assign_add(0.001)
      #%%--------------------------------------------------------------------#
      self.step0.assign_add(1)
      if it>self.it_Stop*0.94 or stopping_run>0: self.max_lr, self.base_lr = 1, 1
      else:
         if self.LR_method =='cycle':
            if it%10000==0 and (self.load_w_t or it>0):
                if it<=self.it_Stop*0.06:# or it>=self.it_Stop*0.72: 
                    self.alpha_max_min(W_Loss_t, it)
            if it%1000==0 and (self.load_w_t or it>0):
                if it>self.it_Stop*0.06: 
                    self.alpha_mine(W_Loss_t, it, stopping_run)
                    
            if it<self.it_Stop*0.06: self.max_lr = 0
            elif it>self.it_Stop*0.88: self.max_lr = 0.5  # 8e-6
            elif it>self.it_Stop*0.82: self.max_lr = 0.25 # 1e-5
         else:
            if it>=2200 and it%1000==0: self.alpha_mine(W_Loss_t, it, stopping_run)
  
      if alpha_constant:
          #self.max_lr, self.base_lr = 0.75,0.75
          if it>self.it_Stop*0.6: self.max_lr, self.base_lr = 1, 1
          else: self.max_lr = 0.5
      learning_rate = self.learning_rate_schedule(self.step0 , self.max_lr, self.base_lr)
      self.optimizer_method.learning_rate = learning_rate 
      if Loss_overal<max_Stop_err and it>self.it_Stop*0.25: stopping_run+=1
      #%%--------------------------------------------------------------------#
      if hvd.rank() == 0:
          if  it % 500 == 0:
             elapsed = time.time() - start_time
             tf.print('rank0: %s-Region,- %d s- ,It: %d, Loss: %.3e, Loss overal:%.3e, w-Loss: %.3e, Time: %.2f, lr: %.3e' 
                       %(region, self.Time[1], it, np.sum(Losses),Loss_overal,np.sum(W_Loss_t),
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
          #---------------------------------------------------------------#
          # Save Models
          if it %2500 == 0 or it==it_Stop-1 or stopping_run==10002:
             self.merged_model.save(save_path+'/tissue',save_traces=True)
                                    #save_format='keras_v3')#, 
             if self.Is_blood!=0:
                 self.merged_model_blood.save(save_path+'/blood', save_traces=True)#save_format='keras_v3'
      #------------------------ Save for  all ranks--------------------------
      if (it%2000==0 and it>3000) or  it==it_Stop-1 or stopping_run==10002:
          if it<it_stop:
              with open(save_path+'/newData_MCMH'+str(hvd.rank())+'_Rank.pkl','wb') as f:
                  pkl.dump({'X_tissue': XT_tf },f)
              if self.Is_blood: 
                  with open(save_path+'/newBloodData_MCMH'+str(hvd.rank())+'_Rank.pkl','wb') as f:
                     pkl.dump({'X_blood':XB_tf},f)
          #------------------------------------------------------------------#
          with open(save_path+'/Power'+str(hvd.rank())+'_Rank.pkl','wb') as f:
              if self.Is_blood:
                  Weight_save = {'wt':self.w_t , #self.weight_[:len(self.w_t)],
                                 'wwall':self.w_wall,# self.weight_[-len(self.w_wall):],
                                 'wb':self.w_b}#self.weight_[len(self.w_t):len(self.w_t)+len(self.w_b)]}
              else: Weight_save = {'wt': self.w_t}
              
              pkl.dump({'wL': Weight_save,'Loss_far':Losses,'LL_Loss':self.LL_Loss},f)
      #----------------------------------------------------------------------#        
      hvd.allreduce(tf.constant(0), name="Barrier")  #-------
      if it%2000==0: tf.print(f' hvd rank = { hvd.rank()} and iter is {it}')
      it+=1
    return Loss_overal   
   
  #***************************************************************************#
  #                       Metropolis-Hasting Algorithm                        #
  #***************************************************************************#
  def ft_inv(self, x, n = 0, D =0):
      D1 = D if D else self.D1_shape 
      n1 = n if n else self.d3-1 
      return tuple(tf.reshape(i,(D1,-1, n1)) for i in x)
  # Create new data for PDEs --> updating train dataset
  def concat_wall(self, xwall):
      xwall = tuple(tuple(tf.reshape(xwall[j][i],(-1,self.d3)) for i in range(len(xwall[j]))) 
                                    for j in range(len(xwall)))
      return tuple(tf.concat(xwi ,axis =0) for xwi in xwall)
  
  def Metropolis_Hasting_Alg_Wall(self, Xb): # MCMH for Xwall
    xend, xwall = Xb
    D_shape = xwall[0][0].shape[0]
    Loss_wall_data,*_, D = self.Loss_wall(xwall, plt_loss=1)
    #--------------Create proposed xwall and Estimate their loss--------------#
    if len(xwall[0][0].shape)==2:
        Nw = [[xwall[i][j].shape[0] for j in range(len(xwall[i]))] for i in range(len(xwall))]
    elif len(xwall[0][0].shape)==3:
        Nw = [[xwall[i][j].shape[0]*xwall[i][j].shape[1] for j in range(len(xwall[i]))] 
                                for i in range(len(xwall))]
   
    xwall_prop = mcmh_fun.new_prop_Wallpoints_tf(self.Wall_band['lb'], 
                                                 self.Wall_band['ub'],Nw, D_shape) 
    
    xwall_prop = tuple(self.fd(xwall_i) for xwall_i in xwall_prop)
    
    Loss_wall_prop,*_ = self.Loss_wall(xwall_prop, plt_loss=1)
    #------------------------find alpha and Xwall_new-------------------------#
    alphaw = [Loss_wall_prop[i]/Loss_wall_data[i] for i in range(len(Loss_wall_data))]
    del Loss_wall_data, Loss_wall_prop
    Xwallnew = self.Metropolis_Hasting_Alg_loss(alphaw ,self.concat_wall(xwall), self.concat_wall(xwall_prop))
    
    Xwall_new = tuple(tuple(tf.reshape(Xwallnew[j][D[j][ii]:D[j][ii+1],:], (D_shape,-1,self.d3)) 
                       for ii in range(len(D[j])-1)) for j in range(len(xwall)))
    return (xend, Xwall_new)  # update new dataset
  #---------------------------------------------------------------------------#
  def Metropolis_Hasting_Alg_PDE(self, Xf):
      Xf = self.ft(Xf,self.d3)
      Nf = [xii.shape[0] for xii in Xf]
      Xf_prop = mcmh_fun.new_prop_points_tf(self.LB_t, self.UB_t, Nf,
                                            self.C_band, Key_order = self.Tissues_name)
      Xf_prop = self.fd(Xf_prop)
      #-----------------------------------------------------------------------#
      #Xf_more = mcmh_fun.new_prop_points_tf(self.LB_t, self.UB_t, [max(int(xii.shape[0]*1e-3),2) for xii in Xf],
      #                                      self.C_band, Key_order = self.Tissues_name)
      #Xf_more = self.fd(Xf_more)
      #Xf_more = tuple(Xf_more[i] if More_[i] and Nf[i]<9e5 else  tf.constant([]) for i in range(len(Xf_more)))
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
      #X0_more=mcmh_fun.new_prop_points_tf(self.LB_t,self.UB_t,[max(int(xii.shape[0]*1e-3),2)for xii in X0],
      #                          self.C_band, Bound = 4,Key_order = self.Tissues_name)
      #X0_more = self.fd(X0_more)
      #X0_more = tuple(X0_more[i] if More_[i] and N0[i]<2e5 else  tf.constant([]) for i in range(len(X0_more)))
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
      #Xlbz_more=mcmh_fun.new_prop_points_tf(self.LB_t,self.UB_t,[max(int(xii.shape[0]*1e-3),2) for xii in Xlbz],
      #                          self.C_band, Bound = 3, Dir = 'lb',Key_order = self.Tissues_name)
      #Xlbz_more = self.fd(Xlbz_more)
      #Xlbz_more = tuple(Xlbz_more[i] if More_[i] and Nz[i]<2.2e5 else  tf.constant([]) for i in range(len(Xlbz_more)))
      #-----------------------------------------------------------------------#
      Xlbznew = self.Metropolis_Hasting_Alg_loss(alphaz[:len(Xlbz)], Xlbz, Xlbz_prop)#, Xlbz_more)
      
      if self.Is_tumor:
          alphaub = [0, 0, 0, 0, alphaz[-2], alphaz[-1]]
          Xubznew = self.Metropolis_Hasting_Alg_loss(alphaub, Xubz, Xubz_prop)
      else: Xubznew = Xubz
             
      return self.ft_inv(Xlbznew), self.ft_inv(Xubznew)
      #except: return Xlbznew, self.ft_inv(Xubznew)
  #---------------------------------------------------------------------------#
  def Metropolis_Hasting_Alg_XY_IFC(self, x, y): # in case we have either tumor or blood or both       
      Col =[0,1] if self.Is_tumor else [1] # if is_blood but not tumor
      X_new =() if self.Is_tumor else (x,) # if is_blood but not tumor
      for col in Col : # x and y      
         X = self.ft(x) if col==0 else self.ft(y)
         # for tumor1,tumor2, Gold
         d = [3,4,5] 
         Nt = [X[i].shape[0] if i in d else 0 for i in range(len(X))]
      
         X_prop = mcmh_fun.new_prop_points_tf(self.LB_t, self.UB_t, Nt, self.C_band,
                                 Bound = col+1, Dir = 'lb', Key_order = self.Tissues_name) 
         X_prop = self.fd(X_prop)
         # loss is defined for elements in d
         Loss_X_data = self.Skin_Loss_BCXY(X,      'lb', col = col, plt_loss=1)
         Loss_X_prop = self.Skin_Loss_BCXY(X_prop, 'lb', col = col, plt_loss=1)
         if col==1:
             Loss_y_data = self.Skin_Loss_BCXY(X,      'ub', col = col, plt_loss=1)
             Loss_y_prop = self.Skin_Loss_BCXY(X_prop, 'ub', col = col, plt_loss=1)
             Loss_X_data = [Loss_y_data[i] + Loss_X_data[i] for i in range(len(Loss_X_data))]
             Loss_X_prop = [Loss_y_prop[i] + Loss_X_prop[i] for i in range(len(Loss_X_prop))]
         #--------------------------------------------------------------------#
         alphax = [0 for i in range(len(X))]
         for i,di in enumerate(d):alphax[di] = Loss_X_prop[i]/Loss_X_data[i] 
         X_new += (self.ft_inv(self.Metropolis_Hasting_Alg_loss(alphax, X, X_prop)),)
         
      return X_new
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
               LR_prpos=np.abs(np.array(decay_rates)-2*R_L)
         #-----------------------------------------------------------#
         try: self.step_LR = np.where(LR_prpos==np.min(LR_prpos))[0][0] 
         except: self.step_LR = 20 # 8e-6
   
         if abs(self.step_LR_pre-self.step_LR)>2:#3100:#2100: # more than two step
             if self.step_LR_pre-self.step_LR<0: self.step_LR = self.step_LR_pre+2 #1500
             else: self.step_LR = self.step_LR_pre-2#1500                  
             if it>20000 and self.step_LR < 3:self.step_LR = 3
         self.max_lr, self.base_lr = decay_rates[self.step_LR],decay_rates[self.step_LR] 
         self.step0.assign(0)
         #--------------------------------------------------------------------#
    

