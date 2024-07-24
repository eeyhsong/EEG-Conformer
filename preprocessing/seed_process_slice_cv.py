# used for SEED dataset 5-fold cross validation

import numpy as np
import scipy.io

root_path = '/Data/SEED/seed_syh/data/'
save_path = '/Data/SEED/seed_syh/data_cv5fold/'
# i, j, k  subject, session, trial
for i in range(15):
    one_subject = []
    one_subject_label = []
    for j in range(3):

        one_session = []
        one_session_label = []
        # todo save training data 
        for k in range(15):
            one_trial = []
            trial_tmp = scipy.io.loadmat(root_path + 'S%d_%d_%d.mat' % (i+1, j+1, k+1))
            trial_data = trial_tmp['trial_data']
            trial_label = np.squeeze(trial_tmp['trial_label'])
            # change to using 2 seconds  200*2
            trial_number = np.int32(trial_data.shape[1]/200)
            for tmp_num in range(trial_number):
                one_trial.append(trial_data[:,tmp_num*200:(tmp_num+1)*200])
            one_trial_label = [trial_label]*trial_number
            # one_trial = np.transpose(one_trial, [2, 1, 0])
            one_session.append(one_trial)
            one_session_label.append(one_trial_label)

        data = np.array(one_session)
        label = np.array(one_session_label)
        np.save(save_path + 'S%d_session%d.npy'%(i+1, j+1), data)
        np.save(save_path + 'S%d_session%d_label'%(i+1, j+1), label)

        print('Finished Subject%d Session%d' % (i+1, j+1))


            
        
