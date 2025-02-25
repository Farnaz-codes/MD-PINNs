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

from Fun2Call_newG_Loni import PINNs_B_T as pinn
#----------------------------------------------------------
save_path0 = '/work/???'
save_path0_new = '/work/???'

Is_tumor, Is_Blood = True, True
Is_segments = [Is_tumor, Is_Blood]
new_case = True

load_w_t =  [False,  True, False, False, False, False, False, False, False]
load_data = [False,  True,  True,  False,   True,  True,  True, True, True]

place_ = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3] 
cont_  = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0]

pre_Time =  [[0,1], [0,1], [490,500], [500,510], [510,525], [525,540], [540,555],[555,570],[570,585]]
next_Time = [[0,1], [1,5], [500,510], [510,525], [525,540], [540,555], [555,570],[570,585],[585,600]]
iter_stop = [70000 for _  in range(len(next_Time))]
iter_stop[3] = 85000
Tn = [1, 2, 2.5, 2.5,2.5,2.5,2.5,2.5,2.5,2.5,2.5,2.5]
#%%-----------------------------------------------------------------------------#
data_path1 = '/work/???'

with open(data_path1 + 'B_T_newG.pkl', 'rb') as f:
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
    
    if ii==1: 
        path_T0 = save_path0 + str(pre_Time[ii][0])+'s_'+str(pre_Time[ii][1]) + 's_B_T_newG2_tanh_softmax_mcmh_rand'
    else: path_T0 = save_path0_new + str(pre_Time[ii][0])+'s_'+str(pre_Time[ii][1]) + 's_B_T_newG2_b3'
    
    LL_Loss = 4.0e2
    mcmh = True
    alpha_constant=False
    
    path_ = path_T0
    path_blood = path_
    
    
    if cont_[ii]:
       path_ = save_path0 + str(next_Time[ii][0])+'s_'+str(next_Time[ii][1]) + 's_B_T_newG2_tanh_softmax_mcmh_rand_max2_2'
       LL_Loss = 5.0e2
       mcmh = False
       alpha_constant=True
       path_blood = path_
    
  
    data_path = save_path0 + str(pre_Time[ii][0])+'s_'+str(pre_Time[ii][1]) + 's_B_T_newG2_tanh_softmax_mcmh_rand_max2_2'
    save_path = save_path0 + str(next_Time[ii][0])+'s_'+str(next_Time[ii][1]) + 's_B_T_newG2_tanh_softmax_mcmh_rand_max2_2'#_adamw'

    #%%-----------------------------------------------------------------------#
    start_time = time.time() 
    load_w_data =(load_w_t[ii],load_data[ii])
    PDEw =  [50, 50, 20, 25,10, 35] # [1st, 2nd, 3rd, t1, t2, g]
    ntime = 4 if ii==3 else 6

    ee= 2.0e-4
    model = pinn.PINNs_tf2_Power(input_args, keys, layers, LB, UB, Blood_Band_ND, Is_segments,
          new_case, load_w_data = load_w_data, path_ = path_, path_T0 = path_T0, ntime = ntime,
                      T0 = T0_new , minpoints = minpoints ,C_band = C_band3, it_Stop = iter_stop[ii],
                      lr_cycle_start = place_[ii], P0 = 0.75, LR_method ='cycle',act_func =  'tanh',
                      path_blood = path_blood, PDEw = PDEw, tn = Tn[ii])#,opt='adadelta')
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
   for iteration in [8]:#,4,5,6]:#3,4,5,6]:#,2,3,4]:#,6,7,8,9]:
       function_adam(iteration)
   
   
   
   
   
   
   
