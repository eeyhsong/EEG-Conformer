% An example to get the BCI competition IV datasets 2b, only for reference
% Data from: http://www.bbci.de/competition/iv/
% using open-source toolbox Biosig on MATLAB: http://biosig.sourceforge.net/
% Just an example, you should change as you need.

% get processed T data
function data = process(subject_index)

%% BioSig Get the data 
% T data
% subject_index = 1; %1-9
session_type = 'E';
dir = ['D:\MI\BCICIV_2b_gdf\B0',num2str(subject_index),'05',session_type,'.gdf'];
[s, HDR] = sload(dir);

% Label 
% label = HDR.Classlabel;
labeldir = ['D:\MI\2b_true_labels\B0',num2str(subject_index),'05',session_type,'.mat'];
load(labeldir);
label = classlabel;

% construct sample - data Section 1000*22*288
Pos = HDR.EVENT.POS;
% Dur = HDR.EVENT.DUR;
Typ = HDR.EVENT.TYP;

k = 0;
data_1 = zeros(1000,3,120);
for j = 1:length(Typ)
    if  Typ(j) == 768
        k = k+1;
        data_1(:,:,k) = s((Pos(j)+750):(Pos(j)+1749),1:3);
    end
end

% wipe off NaN
data_1(isnan(data_1)) = 0;

data = data_1;
pindex = randperm(120);
data = data(:, :, pindex);
label = label(pindex);

% 4-40 Hz
fc = 250;
fb_data = zeros(1000,3,120);

Wl = 4; Wh = 40; % ͨΧ
Wn = [Wl*2 Wh*2]/fc;
[b,a]=cheby2(6,60,Wn);
for j = 1:120
    fb_data(:,:,j) = filtfilt(b,a,data(:,:,j));
end


% eeg_mean = mean(fb_data,3);
% eeg_std = std(fb_data,1,3); 
% fb_data = (fb_data-eeg_mean)./eeg_std;

data = fb_data;

saveDir = ['D:\MI\standard_2b_data\strict_TE\B0',num2str(subject_index),'05E.mat'];
save(saveDir,'data','label');

end


