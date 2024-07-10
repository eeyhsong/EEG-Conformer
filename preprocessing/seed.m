% An example to get the SEED dataset
% Just an example, you should change as you need.

% band pass to 4-47 Hz, standardization
% 15 subjects - 3 sessions - 15 trials
% sample rate 200 Hz
% label: -1 for negative, 0 for neutral and +1 for positive
% label changes to 0 1 2

% Save file S1_1_1  (channels, samples)

file_list = load('./seed_file/namefile_list.mat');
file_list = file_list.namefile_list;
short_name = load('./seed_file/short_name.mat');
short_name = short_name.short_name;
label = load('./seed_file/label.mat');
fc = 200;
Wl = 4; Wh = 47; 
Wn = [Wl*2 Wh*2]/fc;
[b,a]=cheby2(6,60,Wn);

for i = 1:15 % subject
    for j = 1:3 % session
        file_pre = file_list(i,j);
        file_path = strcat('/Datasets/SEED/Preprocessed_EEG/',file_pre,'.mat');
        session_data = load(file_path);
        for k = 1:15 % trial
            trial_data = eval(strcat('session_data.',short_name(i),'_eeg',num2str(k)));
            % trial_data = filtfilt(b,a,trial_data);
            trial_mean = mean(trial_data,2);
            trial_std = std(trial_data,1,2); 
            trial_data = (trial_data-trial_mean)./trial_std;
            trial_data = filtfilt(b,a,trial_data);
            trial_label = label.label(k);
            saveDir = strcat('/Datasets/SEED/seed_save/S',num2str(i),'_',num2str(j),'_',num2str(k),'.mat');
            save(saveDir,'trial_data','trial_label');
        end
    end
end

