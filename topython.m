addpath('H:\data')
addpath('./func/')

path = 'H:\ephys_data\CW47\';
path = 'J:\ephys_data\CW48\';

mkdir([path 'python'])

lst = dir(path);
for j = 1:length(lst)
    if contains(lst(j).name, '_allData.mat')
        load([lst(j).folder '\' lst(j).name])
        namesplit = split(lst(j).name, '_');
        mkdir([path 'python\' strjoin(namesplit(2:4), '_')]) % Make the date
        

        % Get neural data by unit
        numunits = size(obj.units,2);
        units = cell(1, numunits);
        stabletrials = cell(1, numunits);
        imec = zeros(1, numunits);
        celltype = zeros(1, numunits);
        spikewidth = zeros(1, numunits);
        mean_waveform = cell(1, numunits);
        for i = 1:numunits
            trials = cell(1, size(obj.stimProb, 1)); % same number of trials for each unit

            for t = obj.units{1, i}.stable_trials   % go through all the stable trials
                indices = find(obj.units{1, i}.trials == t); % get all trial relevant spike positions
                spikes = obj.units{1, i}.spike_times(indices); % Get all corresponding spikes
                trials{t} = spikes; % Double array per trial
            end

            units{i} = trials; % Add all trial sorted spikes to unit
            imec(i) = str2double(obj.units{1, i}.imec); % include L/R ALM info
            [celltype(i), spikewidth(i)] = func_get_cell_type_kilosort(obj.units{1, i}.mean_waveform); % cell type info
            mean_waveform{i} = obj.units{1, i}.mean_waveform; % mean waveform
            stabletrials{i} = obj.units{1, i}.stable_trials; % add stable trials
        end 
        nametmp = namesplit(5);
        save([path, 'python\' strjoin(namesplit(2:4), '_') '\units.mat'], 'units', 'imec', 'stabletrials', 'mean_waveform', 'celltype')


        % Get behavioral data
        
        obj.Alarm_Nums();
        obj.Pole_Time();
        obj.Cue_Time();
        
        R_hit_tmp = ((char(obj.sides)=='r') & obj.trials.hitHistory);
        R_miss_tmp = ((char(obj.sides)=='r') & obj.trials.missHistory);
        R_ignore_tmp = ((char(obj.sides)=='r') & obj.trials.noResponseHistory);
        L_hit_tmp = ((char(obj.sides)=='l') & obj.trials.hitHistory);
        L_miss_tmp = ((char(obj.sides)=='l') & obj.trials.missHistory);
        L_ignore_tmp = ((char(obj.sides)=='l') & obj.trials.noResponseHistory);
        
        
        LickEarly_tmp = zeros(length(obj.eventsHistory),1);
        LickEarly_tmp(obj.trials.alarmNums,1) = 1;

        % Get i good trials
%         StimDur_tmp = 0;
%         StimLevel = 0;

        StimTrials_tmp = obj.stimProb;
%         i_performing = find(StimTrials_tmp>0);
%         if ~isempty(i_performing)
%             StimTrialsFlag_tmp = StimTrials_tmp;
%             seg_break_pt = i_performing(diff(i_performing)>1);
%             seg_break_pt = [seg_break_pt; i_performing(end)];
%         
%             for i_tmp = seg_break_pt'
%                 if i_tmp<6
%                     StimTrialsFlag_tmp(1:i_tmp) = 0;
%                 else
%                     StimTrialsFlag_tmp(i_tmp-5:i_tmp) = 0;
%                 end
%             end
%         
%             i_good_trials = find(StimTrialsFlag_tmp>0);
%         else
%             i_good_trials = [];
%         end
        i_good_trials = find(StimTrials_tmp>0);

        % Stim information

        total_trials = obj.trials.trialNums;
        for i_solo_trial = 1:total_trials
    
            % get AOM and Galvo info from wavesurfer
            if size(obj.wavesurfer.timestamp,1)>=i_solo_trial
                wave_time_tmp = obj.wavesurfer.timestamp(i_solo_trial,:);
                wave_aom_tmp = obj.wavesurfer.aom_input_trace(i_solo_trial,:);
                wave_xGalvo_tmp = obj.wavesurfer.xGalvo_trace(i_solo_trial,:);
                wave_yGalvo_tmp = obj.wavesurfer.yGalvo_trace(i_solo_trial,:);
            else
                wave_time_tmp = 0;
                wave_aom_tmp = 0;
                wave_xGalvo_tmp = 0;
                wave_yGalvo_tmp = 0;
            end
            
            
            % laser attributes
            AOM_input_tmp = round(max(wave_aom_tmp)*10)/10;
            AOM_data_tmp(i_solo_trial,:) = AOM_input_tmp;
            
            if AOM_input_tmp>0
                i_laser_on = find(wave_aom_tmp>.05);
                t_laser_on = wave_time_tmp(i_laser_on(1));
                t_laser_off = wave_time_tmp(i_laser_on(end));
                
                StimDur_tmp(i_solo_trial,:) = round((t_laser_off-t_laser_on)*10)/10;
                StimOnTime_tmp(i_solo_trial,:) = t_laser_on;
                StimLevel(i_solo_trial,:) = AOM_input_tmp;
            else
                StimDur_tmp(i_solo_trial,:) = 0;
                StimOnTime_tmp(i_solo_trial,:) = 0;
                StimLevel(i_solo_trial,:) = 0;

            end
            
            % Galvo x info

            if max(wave_xGalvo_tmp) < 0.1
                xGalvo(i_solo_trial,:) = min(wave_xGalvo_tmp);
            else
                xGalvo(i_solo_trial,:) = max(wave_xGalvo_tmp);
            end

            % Lick information
                        
        end




        
        save([path 'python\' strjoin(namesplit(2:4), '_') '\behavior.mat'], 'R_hit_tmp', 'R_miss_tmp', 'R_ignore_tmp', 'L_hit_tmp', 'L_miss_tmp', 'L_ignore_tmp', 'LickEarly_tmp', 'i_good_trials', 'StimDur_tmp', 'StimLevel', 'xGalvo')
        
        clearvars AOM_data_tmp StimDur_tmp StimOnTime_tmp StimLevel xGalvo

    end
end