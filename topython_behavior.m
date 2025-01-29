addpath('F:\data')
% path = 'F:\data\Behavior data\BAYLORCW037';
path = 'J:\ephys_data\Behavior data\CW61';

mkdir([path '\python_behavior'])

lst = dir(path);
for j = 1:length(lst)
    if contains(lst(j).name, 'behavior.mat')
        load([lst(j).folder '\' lst(j).name])
        namesplit = split(lst(j).name, '_');
        mkdir([path '\python_behavior\' namesplit{2}])
        
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

        % Get i good trials

        StimTrials_tmp = obj.stimProb;
        i_performing = find(StimTrials_tmp>0);
        if ~isempty(i_performing)
            StimTrialsFlag_tmp = StimTrials_tmp;
            seg_break_pt = i_performing(diff(i_performing)>1);
            seg_break_pt = [seg_break_pt; i_performing(end)];
        
            for i_tmp = seg_break_pt'
                if i_tmp<6
                    StimTrialsFlag_tmp(1:i_tmp) = 0;
                else
                    StimTrialsFlag_tmp(i_tmp-5:i_tmp) = 0;
                end
            end
        
            i_good_trials = find(StimTrialsFlag_tmp>0);
        else
            i_good_trials = [];
        end

        protocol = obj.sessionType; % ynmp or ynmp_sound

        % Delay length  
        total_trials = obj.trials.trialNums;
        for i_solo_trial = 1:total_trials
%             if ~contains(obj.sessionType{i_solo_trial}, 'sound')
            if isfield(obj.eventsHistory{i_solo_trial, 1}.States, 'EarlyLickSample')
%                 delay_duration{i_solo_trial} = 0;

                if ~isnan(obj.eventsHistory{i_solo_trial, 1}.States.EarlyLickSample)
                    LickEarly_tmp(i_solo_trial) = 1;
                end

            elseif isfield(obj.eventsHistory{i_solo_trial, 1}.States, 'DelayPeriod')

%                 delay_duration{i_solo_trial} = obj.eventsHistory{i_solo_trial, 1}.States.DelayPeriod(2) - obj.eventsHistory{i_solo_trial, 1}.States.DelayPeriod(1); 

                if  ~isnan(obj.eventsHistory{i_solo_trial, 1}.States.EarlyLickSample) | ~isnan(obj.eventsHistory{i_solo_trial, 1}.States.EarlyLickDelay)
                    LickEarly_tmp(i_solo_trial) = 1;
                    
                    if ~isnan(obj.eventsHistory{i_solo_trial, 1}.States.EarlyLickDelay)
                    
                        

                    end

                end

            else
%                 delay_duration{i_solo_trial} = 0;
            end

            if (char(obj.sides(i_solo_trial))=='r')
                
                if ~isnan(obj.eventsHistory{i_solo_trial, 1}.States.RewardConsumption)
                    R_hit_tmp(i_solo_trial) = 1;
                elseif ~isnan(obj.eventsHistory{i_solo_trial, 1}.States.TimeOut)
                    R_miss_tmp(i_solo_trial) = 1;
                else
                    R_ignore_tmp(i_solo_trial) = 1;
                end

            elseif (char(obj.sides(i_solo_trial))=='l')
                
                if ~isnan(obj.eventsHistory{i_solo_trial, 1}.States.RewardConsumption)
                    L_hit_tmp(i_solo_trial) = 1;
                elseif ~isnan(obj.eventsHistory{i_solo_trial, 1}.States.TimeOut)
                    L_miss_tmp(i_solo_trial) = 1;
                else
                    L_ignore_tmp(i_solo_trial) = 1;
                end


            end
        end

        delay_duration = obj.settings;
        
        save([path '\python_behavior\' namesplit{2} '\behavior.mat'], 'R_hit_tmp', 'R_miss_tmp', 'R_ignore_tmp', 'L_hit_tmp', 'L_miss_tmp', 'L_ignore_tmp', 'LickEarly_tmp', 'i_good_trials', 'protocol', 'delay_duration')
        
        clearvars delay_duration
    end
end