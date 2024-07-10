% An example to get the BCI competition IV datasets 2a, 2b is the same
% Data from: http://www.bbci.de/competition/iv/
% using open-source toolbox Biosig on MATLAB: http://biosig.sourceforge.net/
% Just an example, you should change as you need.

function data = getData(subject_index)

subject_index = 6; % 1-9

%% T data
session_type = 'T'; % T and E
dir_1 = ['D:\MI\BCICIV_2a_gdf\A0',num2str(subject_index),session_type,'.gdf']; % set your path of the downloaded data
[s, HDR] = sload(dir_1);

% Label 
% label = HDR.Classlabel;
labeldir_1 = ['D:\MI\true_labels\A0',num2str(subject_index),session_type,'.mat'];
load(labeldir_1);
label_1 = classlabel;

% construct sample - data Section 1000*22*288
Pos = HDR.EVENT.POS; % use POS to get trials
% Dur = HDR.EVENT.DUR;
Typ = HDR.EVENT.TYP;

k = 0;
data_1 = zeros(1000,22,288);
for j = 1:length(Typ)
    if  Typ(j) == 768
        k = k+1;
        data_1(:,:,k) = s((Pos(j)+500):(Pos(j)+1499),1:22);
    end
end

% wipe off NaN
data_1(isnan(data_1)) = 0;


% E data
session_type = 'E';
dir_2 = ['D:\Lab\MI\BCICIV_2a_gdf\A0',num2str(subject_index),session_type,'.gdf'];
% dir = 'D:\Lab\MI\BCICIV_2a_gdf\A01E.gdf';
[s, HDR] = sload(dir_2);

% Label 
% label = HDR.Classlabel;
labeldir_2 = ['D:\Lab\MI\true_labels\A0',num2str(subject_index),session_type,'.mat'];
load(labeldir_2);
label_2 = classlabel;

% construct sample - data Section 1000*22*288
Pos = HDR.EVENT.POS;
% Dur = HDR.EVENT.DUR;
Typ = HDR.EVENT.TYP;

k = 0;
data_2 = zeros(1000,22,288);
for j = 1:length(Typ)
    if  Typ(j) == 768
        k = k+1;
        data_2(:,:,k) = s((Pos(j)+500):(Pos(j)+1499),1:22);
    end
end

% wipe off NaN
data_2(isnan(data_2)) = 0;

%% preprocessing
% option - band-pass filter
fc = 250; % sampling rate
Wl = 4; Wh = 40; % pass band
Wn = [Wl*2 Wh*2]/fc;
[b,a]=cheby2(6,60,Wn);
for j = 1:288
    data_1(:,:,j) = filtfilt(b,a,data_1(:,:,j));
    data_2(:,:,j) = filtfilt(b,a,data_2(:,:,j));
end

% option - a simple standardization
%{
eeg_mean = mean(data,3);
eeg_std = std(data,1,3); 
fb_data = (data-eeg_mean)./eeg_std;
%}

%% Save the data to a mat file 
data = data_1;
label = label_1;
% label = t_label + 1;
saveDir = ['D:\MI\standard_2a_data\A0',num2str(subject_index),'T.mat'];
save(saveDir,'data','label');

data = data_2;
label = label_2;
saveDir = ['D:\MI\standard_2a_data\A0',num2str(subject_index),'E.mat'];
save(saveDir,'data','label');

end


        
