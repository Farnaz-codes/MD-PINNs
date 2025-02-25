# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 02:21:39 2023
@author: fnazr
"""
#%%
import pickle as pkl
import time
import sys
DTYPE='float32' 
from Fun2Call_newG_Loni import PINNs_B_NT as pinn
#----------------------------------------------------------
save_path0 = '/home/....'
data_path1 = '/home/....'

Is_tumor, Is_Blood = False, True
Is_segments = [Is_tumor, Is_Blood]
new_case = False


load_w_t =  [True,  True,   True,   True,  True,  True,  False, False]
load_data = [True,  True,   True,   True,  True,  True,  False, True]

place_ = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4] 
cont_  = [0, 1, 1, 1, 1, 1,1,0,0,0,0,0,0,0,0,0,0,0]


pre_Time =  [[0,1], [10,15],[15,20], [60,65],  [50,55], [140,145]]
next_Time = [[0,1], [15,20],[20,25], [65,70],  [55,60], [145,150]]
iter_stop = [10000 for _  in range(len(next_Time))]

#%%-----------------------------------------------------------------------------#
with open(data_path1 + 'B_NT.pkl', 'rb') as f:
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

#%%-----------------------------------------------------------------------------#
def function_adam(ii):
    Blood_Band_ND['time'] = next_Time[ii]
    print(next_Time[ii])
    
    if ii==0: 
        path_T0 = save_path0 + str(pre_Time[ii][0])+'s_'+str(pre_Time[ii][1]) + 's_B_NT_tanh_softmax_mcmh_alphacycle_rand2'
    else : path_T0 = save_path0 + str(pre_Time[ii][0])+'s_'+str(pre_Time[ii][1]) + 's_B_NT_tanh_softmax_mcmh_alphacycle_rand_small5s_2'
    
    LL_Loss = 1.0e5 if ii==5 else 5.0e3
    mcmh = True
    alpha_constant=False
    
    
    path_ = path_T0 
    path_blood = path_
    if cont_[ii]:
       path_ = save_path0 + str(next_Time[ii][0])+'s_'+str(next_Time[ii][1]) + 's_B_NT_tanh_softmax_mcmh_alphacycle_rand_small5s_2'
       mcmh = False
       alpha_constant=True
       path_blood = path_
    
    data_path =path_
    save_path = save_path0 + str(next_Time[ii][0])+'s_'+str(next_Time[ii][1]) + 's_B_NT_tanh_softmax_mcmh_alphacycle_rand_small5s_2_cont_2'#_adamw'

    #%%-----------------------------------------------------------------------#
    start_time = time.time() 
    load_w_data =(load_w_t[ii],load_data[ii])

    ee= 9.0e-5
    ntime = 1# if ii==2 else 1
    model = pinn.PINNs_tf2_Power(input_args, keys, layers, LB, UB, Blood_Band_ND, Is_segments,
          new_case, load_w_data = load_w_data, path_ = path_, path_T0 = path_T0, ntime = ntime,
                      T0 = T0_new , minpoints = minpoints ,C_band = C_band3, it_Stop = iter_stop[ii],
                      lr_cycle_start = place_[ii], P0 = 1.345, LR_method ='cycle',act_func =  'tanh',path_blood=path_blood)#,opt='adadelta')
    print(f' len of weights are {len(model.merged_model.trainable_weights)}*********')                  
           
    LossT = model.train(save_path = save_path, max_Stop_err = ee, mcmh = mcmh,
                             data_path = data_path,  LL_Loss = LL_Loss, opt = 'parallel',it_Stop_lbfgs=0,
                             alpha_constant=alpha_constant)
                             

    elapsed = time.time() - start_time
    print('Training time: %.4f hours' % (elapsed/3600))
#----------------------------------------------------------------------------------------------#
if __name__ == "__main__":
   print(f'*************{len(sys.argv)}*****')
   #print(f'*************{sys.argv[0]}*****')
   #print(f'*************{sys.argv[1]}*****')
   #iteration = int(sys.argv[1])
   for iteration in [1]:#3,4,5,6]:#,2,3,4]:#,6,7,8,9]:
       function_adam(iteration)
   
   
   
   
   
   
   
