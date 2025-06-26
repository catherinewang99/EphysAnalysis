# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 16:16:14 2024

@author: catherinewang
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 15:21:58 2023

@author: Catherine Wang
"""

import sys
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from ephysSession import Session
import os
from numpy import concatenate as cat


class Behavior():
    def __init__(self, path, single=False, behavior_only=False, glmhmm=[]):
        
        # If not single: path is the folder "/.../python/" that contains all the sessions and python compatible mat data
        
        total_sessions = 0
        self.path = path
        self.sessions = []
        
        self.opto_trials = dict()
        
        self.i_good_trials = dict() # zero indexing in python

        self.L_correct = dict()
        self.R_correct = dict()
        
        self.early_lick = dict()
        
        self.L_wrong = dict()
        self.R_wrong = dict()
        
        self.L_ignore = dict()
        self.R_ignore = dict()
        
        self.stim_ON = dict()
        self.stim_level = dict()
        
        self.delay_duration = dict()
        self.protocol = dict()
        
        self.early_lick_time = dict()
        self.early_lick_side = dict()
        
        if not single:
        
            for i in os.listdir(path):
                
                if len(glmhmm) != 0:
                    if i not in glmhmm:
                        continue
                
                if os.path.isdir(os.path.join(path, i)):
                    for j in os.listdir(os.path.join(path, i)):
                        if 'behavior' in j:
                                                        
                            behavior_old = scio.loadmat(os.path.join(path, i, j))
                            behavior = behavior_old.copy()
                            self.sessions += [i]
    
                            self.L_correct[total_sessions] = cat(behavior['L_hit_tmp'])
                            self.R_correct[total_sessions] = cat(behavior['R_hit_tmp'])
                            
                            self.early_lick[total_sessions] = cat(behavior['LickEarly_tmp'])
                            
                            self.L_wrong[total_sessions] = cat(behavior['L_miss_tmp'])
                            self.R_wrong[total_sessions] = cat(behavior['R_miss_tmp'])
                            
                            self.L_ignore[total_sessions] = cat(behavior['L_ignore_tmp'])
                            self.R_ignore[total_sessions] = cat(behavior['R_ignore_tmp'])
                            if 'earlyLick_time' in behavior.keys():
                                self.early_lick_time[total_sessions] = cat(behavior['earlyLick_time'])
                                self.early_lick_side[total_sessions] = cat(behavior['earlyLick_side'])
                                
                            if behavior_only:
                                
                                self.delay_duration[total_sessions] = cat(behavior['delay_duration'])
                                self.protocol[total_sessions] = cat(cat(behavior['protocol']))
                                
                            elif not behavior_only:
                            
                                self.stim_ON[total_sessions] = np.where(cat(behavior['StimDur_tmp']) > 0)
                                # self.stim_level[total_sessions] = cat(behavior['StimLevel'])
                                self.i_good_trials[total_sessions] = cat(behavior['i_good_trials']) - 1 # zero indexing in python


                            total_sessions += 1
    
                            
            self.total_sessions = total_sessions
        
        elif single:
            behavior_old = scio.loadmat(os.path.join(path, 'behavior.mat'))
            behavior = behavior_old.copy()

            self.i_good_trials[total_sessions] = cat(behavior['i_good_trials']) - 1 # zero indexing in python

            self.L_correct[total_sessions] = cat(behavior['L_hit_tmp'])
            self.R_correct[total_sessions] = cat(behavior['R_hit_tmp'])
            
            self.early_lick[total_sessions] = cat(behavior['LickEarly_tmp'])
            
            self.L_wrong[total_sessions] = cat(behavior['L_miss_tmp'])
            self.R_wrong[total_sessions] = cat(behavior['R_miss_tmp'])
            
            self.L_ignore[total_sessions] = cat(behavior['L_ignore_tmp'])
            self.R_ignore[total_sessions] = cat(behavior['R_ignore_tmp'])
            
            self.stim_ON[total_sessions] = np.where(cat(behavior['StimDur_tmp']) > 0)
            if 'StimLevel' in behavior.keys():
                self.stim_level[total_sessions] = cat(behavior['StimLevel'])
            self.total_sessions = 1
            

    def plot_performance_over_sessions(self, all=False, exclude_EL = False, color_background = [], return_vals=False):
        
        reg = []
        opto_p = []

        for i in range(self.total_sessions):
            if all:
                correct = self.L_correct[i] + self.R_correct[i]
                
                if exclude_EL: # Exclude early lick
                    correct = np.delete(correct, np.where(self.early_lick[i])[0])
                    reg += [np.sum(correct) / len(correct)] 
                else: # Not excluding early lick
                    reg += [np.sum(correct) / len(self.L_correct[i])] 

            else:

                igood = self.i_good_trials[i]
                opto = self.stim_ON[i][0]
                igood_opto = np.setdiff1d(self.i_good_trials[i], self.stim_ON[i])
                # igood_opto = np.setdiff1d(range(len(self.L_correct[i])), self.stim_ON[i])
    
                # Filter out early lick
                opto = [o for o in opto if not self.early_lick[i][o]]
                igood_opto = [j for j in igood_opto if not self.early_lick[i][j]]
    
                reg += [np.sum([(self.L_correct[i][t] + self.R_correct[i][t]) for t in igood_opto]) / len(igood_opto)]
                    
                opto_p += [np.sum([(self.L_correct[i][t] + self.R_correct[i][t]) for t in opto]) / len(opto)]
                
        if return_vals:
            return reg
        
        if all:
            plt.plot(reg, 'g--')
            plt.scatter(range(len(reg)), reg, c='g', marker = 'o')
            plt.scatter(np.arange(len(reg))[color_background], np.array(reg)[color_background], c='r', marker = 'o')
            plt.title('Performance over time')
            # plt.xticks(range(self.total_sessions), self.sessions, rotation = 45)
            # plt.ylim(0.45,0.95)
            plt.xlabel('Session #')
            plt.ylabel('% correct')
            plt.axhline(0.5)
            plt.legend()
            plt.show()
            

        else:
            plt.plot(reg, 'g-', label='control')
            plt.plot(opto_p, 'r-', label = 'opto')
            plt.title('Performance over time')
            plt.xticks(range(self.total_sessions), self.sessions, rotation = 45)
            plt.axhline(0.5)
            plt.legend()
            plt.show()
            
            
            return reg, opto_p
            
        
    def plot_LR_performance_over_sessions(self):
        
        Lreg = []
        Rreg = []
        
        Lopto = []
        Ropto = []
         
        for i in range(self.total_sessions):
            
            igood = self.i_good_trials[i]
            opto = self.stim_ON[i][0]
            igood_opto = np.setdiff1d(igood, opto)
            
            # Filter out early lick
            opto = [o for o in opto if not self.early_lick[i][o]]
            igood_opto = [j for j in igood_opto if not self.early_lick[i][j]]

            # if not only_opto:
            Lreg += [np.sum([self.L_correct[i][t] for t in igood_opto]) / 
                     np.sum([(self.L_correct[i][t] + self.L_wrong[i][t] + self.L_ignore[i][t]) for t in igood_opto])]

            Rreg += [np.sum([self.R_correct[i][t] for t in igood_opto]) / 
                     np.sum([(self.R_correct[i][t] + self.R_wrong[i][t] + self.R_ignore[i][t]) for t in igood_opto])]
                
            # if only_opto:
            # opto_p += [np.sum([(self.L_correct[i][t] + self.R_correct[i][t]) for t in opto]) / len(opto)]
            
            Lopto += [np.sum([self.L_correct[i][t] for t in opto]) / 
                     np.sum([(self.L_correct[i][t] + self.L_wrong[i][t] + self.L_ignore[i][t]) for t in opto])]
            
            Ropto += [np.sum([self.R_correct[i][t] for t in opto]) / 
                     np.sum([(self.R_correct[i][t] + self.R_wrong[i][t] + self.R_ignore[i][t]) for t in opto])]
            
        plt.plot(Lreg, 'r-')
        plt.plot(Lopto, 'r--')
        
        plt.plot(Rreg, 'b-')
        plt.plot(Ropto, 'b--')
        
        plt.title('Performance over time')
        plt.xticks(range(self.total_sessions), self.sessions, rotation = 45)
        plt.axhline(0.5)
        plt.show()
        
    def plot_early_lick(self):
        
        EL = list()
        
        for i in range(self.total_sessions):
            
            rate = sum(self.early_lick[i]) / len(self.early_lick[i])
            EL.append(rate)
            
        plt.plot(EL, 'b-')
        plt.title('Early lick rate over time')
        plt.xticks(range(self.total_sessions), self.sessions, rotation = 45)
        plt.show()
    
    def plot_single_session(self, save=False):
         
        Lreg = []
        Rreg = []
        
        Lopto = []
        Ropto = []
         
        i=0            
        
        igood = self.i_good_trials[i]
        opto = self.stim_ON[i][0]
        opto = [o for o in opto if o in self.i_good_trials[i]]
        igood_opto = np.setdiff1d(igood, opto)
        
        # Filter out early lick
        opto = [o for o in opto if not self.early_lick[i][o]]
        igood_opto = [j for j in igood_opto if not self.early_lick[i][j]]

        # if not only_opto:
        Lreg += [np.sum([self.L_correct[i][t] for t in igood_opto]) / 
                 np.sum([(self.L_correct[i][t] + self.L_wrong[i][t] + self.L_ignore[i][t]) for t in igood_opto])]

        Rreg += [np.sum([self.R_correct[i][t] for t in igood_opto]) / 
                 np.sum([(self.R_correct[i][t] + self.R_wrong[i][t] + self.R_ignore[i][t]) for t in igood_opto])]
            
        # if only_opto:
        # opto_p += [np.sum([(self.L_correct[i][t] + self.R_correct[i][t]) for t in opto]) / len(opto)]
        
        Lopto += [np.sum([self.L_correct[i][t] for t in opto]) / 
                 np.sum([(self.L_correct[i][t] + self.L_wrong[i][t] + self.L_ignore[i][t]) for t in opto])]
        
        L_opto_num = len([(self.L_correct[i][t] + self.L_wrong[i][t] + self.L_ignore[i][t]) for t in opto])
        
        Ropto += [np.sum([self.R_correct[i][t] for t in opto]) / 
                 np.sum([(self.R_correct[i][t] + self.R_wrong[i][t] + self.R_ignore[i][t]) for t in opto])]
        
        R_opto_num =len([(self.R_correct[i][t] + self.R_wrong[i][t] + self.R_ignore[i][t]) for t in opto])
        
        plt.plot(cat((Lreg, Lopto)), 'r-', marker='o', label='Left')
        # plt.plot(Lopto, 'r--')
        
        plt.plot(cat((Rreg, Ropto)), 'b-', marker='o', label='Right')
        # plt.plot(Ropto, 'b--')
        
        plt.title('Unilateral ALM optogenetic effect')
        plt.xticks([0, 1], ['Control', 'Opto'])
        plt.ylim(0, 1)
        plt.xlabel('Proportion correct')
        plt.legend()
        
        if save:
            plt.savefig(self.path + 'stim_behavioral_effect.jpg')
        
        plt.show()
        
        return L_opto_num, R_opto_num
    
    
    
    def plot_single_session_multidose(self, save=False):
         
        Lreg = []
        Rreg = []
        
        Lopto = []
        Ropto = []
        
        L_opto_num, R_opto_num = 0, 0
         
        i=0            
        
        igood = self.i_good_trials[i]
        opto = self.stim_ON[i][0]
        igood_opto = np.setdiff1d(igood, opto)
        
        opto_levels = np.array(list(set(self.stim_level[0]))) 
        
        # Filter out early lick
        opto = [o for o in opto if not self.early_lick[i][o]]
        igood_opto = [j for j in igood_opto if not self.early_lick[i][j]]

        Lreg += [np.sum([self.L_correct[i][t] for t in igood_opto]) / 
                 np.sum([(self.L_correct[i][t] + self.L_wrong[i][t] + self.L_ignore[i][t]) for t in igood_opto])]

        Rreg += [np.sum([self.R_correct[i][t] for t in igood_opto]) / 
                 np.sum([(self.R_correct[i][t] + self.R_wrong[i][t] + self.R_ignore[i][t]) for t in igood_opto])]
            
        for level in opto_levels:
            if level == 0:
                continue
            
            opto = np.where(self.stim_level[0] == level)
            
            Lopto += [np.sum([self.L_correct[i][t] for t in opto]) / 
                     np.sum([(self.L_correct[i][t] + self.L_wrong[i][t] + self.L_ignore[i][t]) for t in opto])]
            
            L_opto_num += len([(self.L_correct[i][t] + self.L_wrong[i][t] + self.L_ignore[i][t]) for t in opto])
            
            Ropto += [np.sum([self.R_correct[i][t] for t in opto]) / 
                     np.sum([(self.R_correct[i][t] + self.R_wrong[i][t] + self.R_ignore[i][t]) for t in opto])]
            
            R_opto_num += len([(self.R_correct[i][t] + self.R_wrong[i][t] + self.R_ignore[i][t]) for t in opto])
        
        plt.plot(cat((Lreg, Lopto)), 'r-', marker='o', label='Left')
        # plt.plot(Lopto, 'r--')
        
        plt.plot(cat((Rreg, Ropto)), 'b-', marker='o', label='Right')
        # plt.plot(Ropto, 'b--')
        
        plt.title('Late delay optogenetic effect on unilateral ALM')
        ticks = ['{} AOM'.format(x) for x in opto_levels[1:]]
        plt.xticks(range(len(opto_levels)), ['Control'] + ticks)
        plt.ylim(0, 1)
        plt.xlabel('Proportion correct')
        plt.ylabel('Perturbation condition')
        plt.legend()
        
        if save:
            plt.savefig(self.path + 'stimDOSE_behavioral_effect.jpg')
        
        plt.show()       

        return L_opto_num, R_opto_num
    
    
    def plot_licks_single_sess(self):
        # JH Plot
        return None

    def learning_progression(self, window = 50, save=False, imaging=False,
                             return_results=False, include_delay = True, color_background = [],
                             early_lick_ylim=True):
        """
        Plot the learning progression with three panels indicating delay duration, performance,
        and early lick rate over sessions

        Parameters
        ----------
        window : int, optional
            Window to average over. The default is 50.
        save : bool, optional
            Whether to save fig somewhere. The default is False.
        imaging : bool, optional
            Only show imaging days. The default is False.
        return_results : bool, optional
            Return values. The default is False.
        color_background : list, optional
            Which sessions to provide a colored background for, zero-indexed.

        Returns
        -------
        Three lists
            Returns each of the panels as lists.

        """
        # Figures showing learning over protocol
        if include_delay:
            f, axarr = plt.subplots(3, 1, sharex='col', figsize=(16,10))
        else:
            f, axarr = plt.subplots(2, 1, sharex='col', figsize=(16,10))

        # Concatenate all sessions
        delay_duration = np.array([])
        correctarr = np.array([])
        earlylicksarr = np.array([])
        num_trials = [0]
        window = int(window/2)
        background_trials = []
        
        for sess in range(self.total_sessions):
            
            
            # delay = np.convolve(self.delay_duration[sess], np.ones(window)/window, mode = 'same')
            delay = self.delay_duration[sess]
            
            if imaging:
                if 3 not in delay or len(set(delay)) > 1:
                    continue
                # if 'CW028' in self.path and sess ==1:
                #     continue
            
            delay_duration = np.append(delay_duration, delay[window:-window])

            # delay_duration = np.append(delay_duration, self.delay_duration[sess][window:-window])
            
            correct = self.L_correct[sess] + self.R_correct[sess]
            correct = np.convolve(correct, np.ones(window*2)/(window*2), mode = 'same')
            correctarr = np.append(correctarr, correct[window:-window])
            
            earlylicks = np.convolve(self.early_lick[sess], np.ones(window*2)/(window*2), mode = 'same')
            earlylicksarr = np.append(earlylicksarr, earlylicks[window:-window])
            
            if sess in color_background:
                background_trials += [sum(num_trials)]
                
            num_trials += [len(self.L_correct[sess])-(window*2)]
            
            if sess in color_background:
                background_trials += [sum(num_trials)]
                
        num_trials = np.cumsum(num_trials)
        
        if include_delay:
            # Protocol
            
            axarr[0].plot(delay_duration, 'r')
            axarr[0].set_ylabel('Delay duration (s)')
            axarr[0].set_ylim(-0.1, 4)
    
            
            # Performance
            
            axarr[1].plot(correctarr, 'g')        
            axarr[1].set_ylabel('% correct')
            axarr[1].axhline(y=0.7, alpha = 0.5, color='orange')
            axarr[1].axhline(y=0.5, alpha = 0.5, color='red', ls = '--')
            # axarr[1].set_ylim(0.4, 1)
            
            # Early licking
            
            axarr[2].plot(earlylicksarr, 'b')        
            axarr[2].set_ylabel('% Early licks')
            axarr[2].set_xlabel('Trials')
            if early_lick_ylim:
                axarr[2].set_ylim(0, 0.4)
        
        else:
            # Performance
            
            axarr[0].plot(correctarr, 'g')        
            axarr[0].set_ylabel('% correct')
            axarr[0].axhline(y=0.7, alpha = 0.5, color='orange')
            axarr[0].axhline(y=0.5, alpha = 0.5, color='red', ls = '--')
            # axarr[0].set_ylim(0.4, 1)
            
            # Early licking
            
            axarr[1].plot(earlylicksarr, 'b')        
            axarr[1].set_ylabel('% Early licks')
            axarr[1].set_xlabel('Trials')
            if early_lick_ylim:
                axarr[1].set_ylim(0, 0.4)
        
        # Color background (optional)
        
        if len(color_background) != 0:
            for i in range(len(color_background)):
                if include_delay:

                    axarr[0].axvspan(background_trials[2*i], background_trials[(2*i)+1], ymin = -0.1, ymax = 4, color = 'red', alpha=0.3)
                    axarr[1].axvspan(background_trials[2*i], background_trials[(2*i)+1], ymin = 0, ymax = 1, color = 'red', alpha=0.3)
                    axarr[2].axvspan(background_trials[2*i], background_trials[(2*i)+1], ymin = 0, ymax = 1, color = 'red', alpha=0.3)
                else:
                    axarr[0].axvspan(background_trials[2*i], background_trials[(2*i)+1], ymin = 0, ymax = 1, color = 'red', alpha=0.3)
                    axarr[1].axvspan(background_trials[2*i], background_trials[(2*i)+1], ymin = 0, ymax = 1, color = 'red', alpha=0.3)
                
        # Denote separate sessions
        
        for num in num_trials:
            axarr[0].axvline(num, color = 'grey', alpha=0.5, ls = '--')
            axarr[1].axvline(num, color = 'grey', alpha=0.5, ls = '--')
            if include_delay:
                axarr[2].axvline(num, color = 'grey', alpha=0.5, ls = '--')
            
        
        if save:
            # plt.savefig(self.path + r'\learningcurve.png')
            print("saving")
            plt.savefig(save)
        plt.show()
        
        
        if return_results:
            
            return delay_duration, correctarr, num_trials
        
    def learning_progression_no_EL(self, window = 50, save=False, imaging=False, return_results=False):
        
        # Figures showing learning over protocol
        
        f, axarr = plt.subplots(2, 1, sharex='col', figsize=(16,12))
        
        # Concatenate all sessions
        delay_duration = np.array([])
        correctarr = np.array([])
        earlylicksarr = np.array([])
        num_trials = []
        window = int(window/2)
        
        for sess in range(self.total_sessions):
            

            # delay = np.convolve(self.delay_duration[sess], np.ones(window)/window, mode = 'same')
            delay = self.delay_duration[sess]
            
            if imaging:
                if 3 not in delay or len(set(delay)) > 1:
                    continue
                # if 'CW028' in self.path and sess ==1:
                #     continue
            
            delay_duration = np.append(delay_duration, delay[window:-window])

            # delay_duration = np.append(delay_duration, self.delay_duration[sess][window:-window])
            
            correct = self.L_correct[sess] + self.R_correct[sess]
            correct = np.convolve(correct, np.ones(window*2)/(window*2), mode = 'same')
            correctarr = np.append(correctarr, correct[window:-window])
            
            earlylicks = np.convolve(self.early_lick[sess], np.ones(window*2)/(window*2), mode = 'same')
            earlylicksarr = np.append(earlylicksarr, earlylicks[window:-window])
            
            num_trials += [len(self.L_correct[sess])-(window*2)]
        num_trials = np.cumsum(num_trials)
        
        # Protocol
        
        axarr[0].plot(delay_duration, 'r')
        axarr[0].set_ylabel('Delay duration (s)')

        
        # Performance
        
        axarr[1].plot(correctarr, 'g')        
        axarr[1].set_ylabel('% correct')
        axarr[1].axhline(y=0.7, alpha = 0.5, color='orange')
        axarr[1].axhline(y=0.5, alpha = 0.5, color='red', ls = '--')
        axarr[1].set_ylim(0, 1)
        

        axarr[1].set_xlabel('Trials')
        
        
        # Denote separate sessions
        
        for num in num_trials:
            axarr[0].axvline(num, color = 'grey', alpha=0.5, ls = '--')
            axarr[1].axvline(num, color = 'grey', alpha=0.5, ls = '--')
        
        if save:
            plt.savefig(self.path + r'\learningcurve.pdf')
        plt.show()
        
        
        if return_results:
            
            return delay_duration, correctarr, cat(([0], num_trials))
            
    def get_acc_EL(self, window = 50, imaging=False, sessions=None):
        """Returns agg accuracy over all sessions
        
        Parameters
        -------
        window : int
        imaging : bool
        sessions : tuple, optional
            if provided, use these session for the return array
    
        Returns
        -------
        int
            int corresponding to the length of shortest trial in whole session
        """
        # Concatenate all sessions
        delay_duration = np.array([])
        correctarr = np.array([])
        earlylicksarr = np.array([])
        num_trials = []
        window = int(window/2)
        
        if sessions != None:
            start = sessions[0]
            end = sessions[1]
        else:
            start = 0
            end = self.total_sessions
        
        for sess in range(start, end):
            

            # delay = np.convolve(self.delay_duration[sess], np.ones(window)/window, mode = 'same')
            delay = self.delay_duration[sess]
            
            if imaging:
                if 3 not in delay or len(set(delay)) > 1:
                    continue
                # if 'CW028' in self.path and sess ==1:
                #     continue
            
            delay_duration = np.append(delay_duration, delay[window:-window])

            # delay_duration = np.append(delay_duration, self.delay_duration[sess][window:-window])
            
            correct = self.L_correct[sess] + self.R_correct[sess]
            correct = np.convolve(correct, np.ones(window*2)/(window*2), mode = 'same')
            correctarr = np.append(correctarr, correct[window:-window])
            
            earlylicks = np.convolve(self.early_lick[sess], np.ones(window*2)/(window*2), mode = 'same')
            earlylicksarr = np.append(earlylicksarr, earlylicks[window:-window])
            
            num_trials += [len(self.L_correct[sess])-(window*2)]
        num_trials = np.cumsum(num_trials)
          
        return earlylicksarr, correctarr, cat(([0], num_trials))
        
    def correct_error(self, i_good=False):
        """
        Return array of correct and error behavior (0: error 1: correct) for 
        given single session
        
        Inputs
        -------
        i_good : bool
            If use i_good_trials ONLY

        Returns
        -------
        array
            array of 0 and 1s corresponding to error and correct decisions
            \

        """
        if i_good:
            
            return (self.L_correct[0][self.i_good_trials[0]] + self.R_correct[0][self.i_good_trials[0]]).astype(int)
        
        else:
            
            return (self.L_correct[0] + self.R_correct[0]).astype(int) 


    def time_to_reach_perf(self, performance, delay_threshold, window=20):
        """
        
        The number of trials (or other metric) needed to reach a performance threshold
        for a given delay length upper limit 
        i.e. number of trials needed to reach 70% at less than 1s delay

    
        Parameters
        ----------
        performance : TYPE
            DESCRIPTION.
        delay_threshold : TYPE
            DESCRIPTION.
        window : int
            the number of trials to calculate the performance, default 20.

        Returns
        -------
        None.

        """
        
        trial_count = 0
        
        for sess in range(self.total_sessions):
            
            delay = self.delay_duration[sess]
            
            if len(np.where(delay > delay_threshold)[0]) == 0: # If no delay lengths exceed the threshold
                                
                correct = self.L_correct[sess] + self.R_correct[sess]
                correct = np.convolve(correct, np.ones(window*2)/(window*2), mode = 'same')
                correct = correct[window:-window]
                
                if len(np.where(correct > performance)[0]) > 0: # if it exceeds performance % at some point
                
                    trial_count += np.where(correct>performance)[0][0] + window
                    return trial_count
                
                else:
                    
                    trial_count += len(delay)
        

        