# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 02:21:39 2023
@author: fnazr
"""
#%%
import pickle as pkl
import time
import sys
from Fun2Call_newG_Loni import PINNs_NB_NT as pinn
DTYPE='float32' 

#----------------------------------------------------------
save_path0 = '/work/???'

Is_tumor, Is_Blood = False, False
Is_segments = [Is_tumor, Is_Blood]
new_case = False

load_w_t =  [False, True, True,  True,  True,    False]
load_data = [False , True, True,  True,  True,   True ]

place_ = [0, 4,4,4,4,4,4,4,4, 4, 4, 4, 4,4,4,4,3,3] 
cont_  = [0, 1,1,1,1,0,0,0,0, 1, 1, 1, 1,1,1,1,0, 0, 0]


pre_Time =  [[0,1], [290,300], [300,310], [310,320], [320,330], [330,340], [340,350],[350,360] ]
next_Time = [[0,1], [300,310], [310,320], [320,330], [330,340], [340,350], [350,360],[360,370]]#,  [125,130], [130,135], [135,140], [140,145], [145,150]]
iter_stop = [65000 for _  in range(len(next_Time))]


pre_Time =  [[0,1], [65,70], [330,340],[340,350],[350,360],[360,370],[370,380],[380,390]]
next_Time = [[0,1], [70,80] ,[340,350],[350,360],[360,370],[370,380],[380,390],[390,400]]
iter_stop[0] = 300000

#%%-----------------------------------------------------------------------------#
data_path1 = '/work/???'
data_path_err = '/work/???'
path_0_1s = 'NB_NT.pkl'

with open(data_path1 + path_0_1s, 'rb') as f:
   DataInfo = pkl.load(f)

T0_new = DataInfo['T0_new']
keys = DataInfo['Data_order']
Nf , N_points = DataInfo['Nf'] , DataInfo['All_points'] 
C_band3 = DataInfo['C_band3']
layers = DataInfo['layers']
minpoints = DataInfo['Min_num_ptns'][0]
LB, UB = DataInfo['LB'], DataInfo['UB']
Blood_Band = DataInfo['Blood_Band']
Blood_Band_ND = DataInfo['Blood_Band_ND']
Time = Blood_Band_ND['time'] 

blood_arg = (DataInfo['X_end'], DataInfo['X_wall'])
Tissue_arg = DataInfo['Tissue_data']
input_args =  {'args_tissue':Tissue_arg,'args_blood': blood_arg}


TotalPoints = 0
for ki in N_points: 
    TotalPoints+=sum(N_points[ki])
    print(f'Total number of points  in {ki} is {sum(N_points[ki])}')
print('Total number of points are: '  + str(TotalPoints))
print(f'UB is {UB}')

#%%-----------------------------------------------------------------------------#
def function_adam(ii):
    Blood_Band_ND['time'] = next_Time[ii]
    print(next_Time[ii])
    
    if ii==0: path_T0 = []
    elif ii==1: path_T0 = save_path0 + str(pre_Time[ii][0])+'s_'+str(pre_Time[ii][1]) + 's_NB_NT_tanh_softmax_mcmh_rand_small5s'
    else: path_T0 = save_path0 + str(pre_Time[ii][0])+'s_'+str(pre_Time[ii][1]) + 's_NB_NT_tanh_softmax_mcmh_rand_small5s_10s_2'
    
    LL_Loss = 120#2e2
    mcmh = True
    alpha_constant=False
    path_1 = path_T0
    
    
    path_ = path_T0
    print(f'path_T0 here is {path_T0}')

    if cont_[ii]:
       path_ = save_path0 + str(next_Time[ii][0])+'s_'+str(next_Time[ii][1]) + 's_NB_NT_tanh_softmax_mcmh_rand_small5s_10s_2'
       path_1 = path_
       mcmh = False
       alpha_constant = True

    
    data_path =path_
    save_path = save_path0 + str(next_Time[ii][0])+'s_'+str(next_Time[ii][1]) + 's_NB_NT_tanh_softmax_mcmh_rand_small5s_for_plot'#_adamw'
    print(f'path_T0 is {path_T0}')

    #%%-----------------------------------------------------------------------#
    start_time = time.time() 
    load_w_data =(load_w_t[ii],load_data[ii])
    Tn = 0#5# if ii==1 else 5
    ntime = 1 if ii==1 else 2

    ee= 2.5e-5#1.0e-5 if ii==0 else 3.85e-5
    model = pinn.PINNs_tf2_Power(input_args, keys, layers, LB, UB, Blood_Band_ND, Is_segments,
          new_case, load_w_data = load_w_data, path_ = path_, path_T0 = path_T0, ntime = ntime,T0 = T0_new ,
          minpoints = minpoints ,C_band = C_band3, it_Stop = iter_stop[ii], lr_cycle_start = place_[ii],
          P0 = 1.345, act_func =  'tanh',data_path_err = data_path_err, path_1 = path_1, tn =Tn )#,opt='adadelta')
          
    print(f' len of weights are {len(model.merged_model.trainable_weights)}*********')                  
           
           
    LossT = model.train(save_path = save_path, max_Stop_err = ee, mcmh = mcmh,
                        data_path = data_path,  LL_Loss = LL_Loss , alpha_constant=alpha_constant)
                             

    elapsed = time.time() - start_time
    print('Training time: %.4f hours' % (elapsed/3600))
#----------------------------------------------------------------------------------------------#
if __name__ == "__main__":
   print(f'*************{len(sys.argv)}*****')
   #print(f'*************{sys.argv[0]}*****')
   #print(f'*************{sys.argv[1]}*****')
   #iteration = int(sys.argv[1])
   for iteration in [0]:#3,4,5,6]:#,2,3,4]:#,6,7,8,9]:
       function_adam(iteration)
   
   
   
   
   
   
   
