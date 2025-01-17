# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 13:19:10 2024

@author: catherinewang

"""
import sys
sys.path.append("C:\scripts\Ephys analysis\ephys_pipeline")
import numpy as np
from numpy import concatenate as cat
import matplotlib.pyplot as plt
from scipy import stats
import copy
import scipy.io as scio
from sklearn.preprocessing import normalize
from ephysSession import Session
import sympy
from random import shuffle
import time
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
plt.rcParams['pdf.fonttype'] = 42 

class Mode(Session):
    def __init__(self, path, side = 'all', passive=False, laser='blue', only_ppyr=True,
                 proportion_train = 0.5, lickdir=False,
                 responsive_neurons = [], train_test_trials = [],
                 binsize=400, timestep=10):
    # def __init__(self, path, lickdir=True, use_reg=False, triple=False, filter_reg= True, 
    #              layer_num='all', responsive_neurons = [], use_selective= False, use_background_sub=False,
    #              baseline_normalization = "dff_avg", proportion_train = 0.5,
    #              train_test_trials = []):
        """
        Child object of Session that allows for activity mode calculations 
        Mostly adds new functions and also train test split of trials 

        Parameters
        ----------
        path : TYPE
            DESCRIPTION.
        lickdir : TYPE, optional
            DESCRIPTION. The default is True.
        use_reg : TYPE, optional
            DESCRIPTION. The default is False.
        triple : TYPE, optional
            DESCRIPTION. The default is False.
        filter_reg : TYPE, optional
            DESCRIPTION. The default is True.
        layer_num : TYPE, optional
            DESCRIPTION. The default is 'all'.
        responsive_neurons : TYPE, optional
            DESCRIPTION. The default is [].
        use_selective : TYPE, optional
            DESCRIPTION. The default is False.
        use_background_sub : TYPE, optional
            DESCRIPTION. The default is False.
        baseline_normalization : TYPE, optional
            DESCRIPTION. The default is "dff_avg".
        proportion_train : TYPE, optional
            DESCRIPTION. The default is 0.5.
        train_test_trials : list, optional
            Contains R train L train R test L test. The default is [], meaning
            sort in house. Indices are only for control, i good non stim, and 
            no early lick trials. Give an index in range(len(R_control_trials)),
            or example. two lists of four lists (correct error)

        Returns
        -------
        None.

        """
        # Inherit all parameters and functions of session.py
        super().__init__(path=path, side=side, passive=passive, laser=laser,
                         only_ppyr=only_ppyr)
        
        self.lickdir = lickdir
        self.binsize = binsize
        self.timestep = timestep
        # if len(responsive_neurons) == 0:
        #     _ = self.get_stim_responsive_neurons()
        # else:
        #     self.responsive_neurons = self.good_neurons[responsive_neurons]
        #     self.good_neurons = self.good_neurons[responsive_neurons]
            

        # Construct train and test sets for control and opto trials
        # built this section so we can split trials into train/test and track at the same time 
        # for error bar creation in some subsequent graphs

        self.proportion_train = proportion_train
        self.proportion_test = 1 - proportion_train
        self.i_good_non_stim_trials = [t for t in self.i_good_trials if not self.stim_ON[t]]

        ### SORT CONTROL TRIALS:
        # First method: in house
        if len(train_test_trials) == 0:
                
            print("in house train test sorting")
            
            r_trials = [i for i in self.i_good_non_stim_trials if not self.early_lick[i] and self.R_correct[i]]
            l_trials = [i for i in self.i_good_non_stim_trials if not self.early_lick[i] and self.L_correct[i]]
            np.random.shuffle(r_trials) # shuffle the items in 
            np.random.shuffle(l_trials) # shuffle the items in 
            numr, numl = len(r_trials), len(l_trials)
            
            self.r_train_idx, self.l_train_idx = r_trials[:int(numr*self.proportion_train)], l_trials[:int(numl*self.proportion_train)]
            self.r_test_idx, self.l_test_idx = r_trials[int(numr*self.proportion_train):], l_trials[int(numl*self.proportion_train):]
                        
            r_trials = [i for i in self.i_good_non_stim_trials if not self.early_lick[i] and self.R_wrong[i]]
            l_trials = [i for i in self.i_good_non_stim_trials if not self.early_lick[i] and self.L_wrong[i]]
            np.random.shuffle(r_trials) # shuffle the items in 
            np.random.shuffle(l_trials) # shuffle the items in 
            numr, numl = len(r_trials), len(l_trials)

            self.r_train_err_idx, self.l_train_err_idx = r_trials[:int(numr*self.proportion_train)], l_trials[:int(numl*self.proportion_train)]
            self.r_test_err_idx, self.l_test_err_idx = r_trials[int(numr*self.proportion_train):], l_trials[int(numl*self.proportion_train):]
          
            if proportion_train == 1:
                self.r_test_idx, self.l_test_idx = self.r_train_idx, self.l_train_idx
                self.r_test_err_idx, self.l_test_err_idx = self.r_train_err_idx, self.l_train_err_idx
                
        # Second method: read in from outside
        else:
            self.r_train_idx, self.l_train_idx, self.r_test_idx, self.l_test_idx = train_test_trials[0]
            self.r_train_err_idx, self.l_train_err_idx, self.r_test_err_idx, self.l_test_err_idx = train_test_trials[1]
            

        ### SORT OPTO TRIALS:
        # Sort by trial type
        if not lickdir:
            
            r_trials = [i for i in self.i_good_trials if not self.early_lick[i] and self.stim_ON[i] and (self.R_correct[i] or self.R_wrong[i])]
            l_trials = [i for i in self.i_good_trials if not self.early_lick[i] and self.stim_ON[i] and (self.L_correct[i] or self.L_wrong[i])]
            
            r_trials_stimright = [i for i in self.i_good_trials if not self.early_lick[i] and self.stim_ON[i] and self.stim_side[i] == 'R' and (self.R_correct[i] or self.R_wrong[i])]
            l_trials_stimright = [i for i in self.i_good_trials if not self.early_lick[i] and self.stim_ON[i] and self.stim_side[i] == 'R' and (self.L_correct[i] or self.L_wrong[i])]
            r_trials_stimleft = [i for i in self.i_good_trials if not self.early_lick[i] and self.stim_ON[i] and self.stim_side[i] == 'L' and (self.R_correct[i] or self.R_wrong[i])]
            l_trials_stimleft = [i for i in self.i_good_trials if not self.early_lick[i] and self.stim_ON[i] and self.stim_side[i] == 'L' and (self.L_correct[i] or self.L_wrong[i])]            
        # Sort by lick dir
        else:
            print('Sort by lick dir')
                        
            r_trials = [i for i in self.i_good_trials if not self.early_lick[i] and self.stim_ON[i] and (self.R_correct[i] or self.L_wrong[i])]
            l_trials = [i for i in self.i_good_trials if not self.early_lick[i] and self.stim_ON[i] and (self.L_correct[i] or self.R_wrong[i])]
            
            r_trials_stimright = [i for i in self.i_good_trials if not self.early_lick[i] and self.stim_ON[i] and self.stim_side[i] == 'R' and (self.R_correct[i] or self.L_wrong[i])]
            l_trials_stimright = [i for i in self.i_good_trials if not self.early_lick[i] and self.stim_ON[i] and self.stim_side[i] == 'R' and (self.L_correct[i] or self.R_wrong[i])]
            r_trials_stimleft = [i for i in self.i_good_trials if not self.early_lick[i] and self.stim_ON[i] and self.stim_side[i] == 'L' and (self.R_correct[i] or self.L_wrong[i])]
            l_trials_stimleft = [i for i in self.i_good_trials if not self.early_lick[i] and self.stim_ON[i] and self.stim_side[i] == 'L' and (self.L_correct[i] or self.R_wrong[i])]            
        
        np.random.shuffle(r_trials) # shuffle the items in 
        np.random.shuffle(l_trials) # shuffle the items in 
        numr, numl = len(r_trials), len(l_trials)
            
        self.r_train_opto_idx, self.l_train_opto_idx = r_trials[:int(numr*self.proportion_train)], l_trials[:int(numl*self.proportion_train)]
        self.r_test_opto_idx, self.l_test_opto_idx = r_trials[int(numr*self.proportion_train):], l_trials[int(numl*self.proportion_train):]
        
        self.r_opto_idx = r_trials
        self.l_opto_idx = l_trials

        self.r_opto_stim_right_idx = r_trials_stimright
        self.l_opto_stim_right_idx = l_trials_stimright   
        
        self.r_opto_stim_left_idx = r_trials_stimleft
        self.l_opto_stim_left_idx = l_trials_stimleft   
        
        ## ASSIGN NEURAL ACTIVITY PER train_idx / test_idx
        
        counter = 0
        for n in self.good_neurons:
                        
            # Splits neural data according to existing indices (sorted above)
            r_train, l_train, r_test, l_test = self.train_test_split_data_ctl(n, binsize=binsize, timestep=timestep) #, r, l) # discards unstable trials
            r_err_train, l_err_train, r_err_test, l_err_test = self.train_test_split_data_err(n, binsize=binsize, timestep=timestep)# , r_err, l_err) # Sorts similarly but indices not saved
            r_opto_train, l_opto_train, r_opto_test, l_opto_test = self.train_test_split_data_opto(n, binsize=binsize, timestep=timestep)#, r_opto, l_opto) # Always includes error trials
            
            if counter == 0:
                
                self.PSTH_r_train_correct = np.reshape(r_train, (1,-1))
                self.PSTH_l_train_correct = np.reshape(l_train, (1,-1))
                self.PSTH_r_train_error = np.reshape(r_err_train, (1,-1))
                self.PSTH_l_train_error = np.reshape(l_err_train, (1,-1))
                # self.PSTH_r_train_opto = np.reshape(r_opto_train, (1,-1))
                # self.PSTH_l_train_opto = np.reshape(l_opto_train, (1,-1))

                self.PSTH_r_test_correct = np.reshape(r_test, (1,-1))
                self.PSTH_l_test_correct = np.reshape(l_test, (1,-1))
                self.PSTH_r_test_error = np.reshape(r_err_test, (1,-1))
                self.PSTH_l_test_error = np.reshape(l_err_test, (1,-1))
                # self.PSTH_r_test_opto = np.reshape(r_opto_test, (1,-1))
                # self.PSTH_l_test_opto = np.reshape(l_opto_test, (1,-1))
                
            else:
                self.PSTH_r_train_correct = np.concatenate((self.PSTH_r_train_correct, np.reshape(r_train, (1,-1))), axis = 0)
                self.PSTH_l_train_correct = np.concatenate((self.PSTH_l_train_correct, np.reshape(l_train, (1,-1))), axis = 0)
                self.PSTH_r_train_error = np.concatenate((self.PSTH_r_train_error, np.reshape(r_err_train, (1,-1))), axis = 0)
                self.PSTH_l_train_error = np.concatenate((self.PSTH_l_train_error, np.reshape(l_err_train, (1,-1))), axis = 0)
                # self.PSTH_r_train_opto = np.concatenate((self.PSTH_r_train_opto, np.reshape(r_opto_train, (1,-1))), axis = 0)
                # self.PSTH_l_train_opto = np.concatenate((self.PSTH_l_train_opto, np.reshape(l_opto_train, (1,-1))), axis = 0)
                
                self.PSTH_r_test_correct = np.concatenate((self.PSTH_r_test_correct, np.reshape(r_test, (1,-1))), axis = 0)
                self.PSTH_l_test_correct = np.concatenate((self.PSTH_l_test_correct, np.reshape(l_test, (1,-1))), axis = 0)
                self.PSTH_r_test_error = np.concatenate((self.PSTH_r_test_error, np.reshape(r_err_test, (1,-1))), axis = 0)
                self.PSTH_l_test_error = np.concatenate((self.PSTH_l_test_error, np.reshape(l_err_test, (1,-1))), axis = 0)
                # self.PSTH_r_test_opto = np.concatenate((self.PSTH_r_test_opto, np.reshape(r_opto_test, (1,-1))), axis = 0)
                # self.PSTH_l_test_opto = np.concatenate((self.PSTH_l_test_opto, np.reshape(l_opto_test, (1,-1))), axis = 0)
                
            counter += 1
            
        self.T_cue_aligned_sel = self.t
            
        time_epochs = [self.sample, self.delay, self.response]
        self.time_epochs = time_epochs

    
    def basis_col(self, A):
        # Bases
    
        # basis_col(A) produces a basis for the subspace of Eucldiean n-space 
        # spanned by the vectors {u1,u2,...}, where the matrix A is formed from 
        # these vectors as its columns. That is, the subspace is the column space 
        # of A. The columns of the matrix that is returned are the basis vectors 
        # for the subspace. These basis vectors will be a subset of the original 
        # vectors. An error is returned if a basis for the zero vector space is 
        # attempted to be produced.
    
        # For example, if the vector space V = span{u1,u2,...}, where u1,u2,... are
        # row vectors, then set A to be [u1' u2' ...].
    
        # For example, if the vector space V = Col(B), where B is an m x n matrix,
        # then set A to be equal to B.
    
        matrix_size = np.shape(A)
    
        m = matrix_size[0]
        n = matrix_size[1]
    
        if np.array_equal(A, np.zeros((m,n))):
            raise ValueError('There does not exist a basis for the zero vector space.')
        elif n == 1:
            basis = A
        else:
            flag = 0
    
            if n == 2:
                multiple = A[0,1]/A[0,0]
                count = 0
    
                for i in range(m):
                    if A[i,1]/A[i,0] == multiple:
                        count = count + 1
    
                if count == m:
                    basis = A[:,0].reshape(-1, 1)
                    flag = 1
    
            if flag == 0:
                ref_A, pivot_columns = sympy.Matrix(A).rref() # double check if works
    
                B = np.zeros((m, len(pivot_columns)))
    
                for i in range(len(pivot_columns)):
                    B[:,i] = A[:,pivot_columns[i]]
    
                basis = B
    
        return basis
    
    def is_orthogonal_set(self, A):
        """
        Orthogonal Sets
    
        Determines if a set of vectors in Euclidean n-space is orthogonal. The matrix A
        is formed from these vectors as its columns. That is, the subspace spanned by the
        set of vectors is the column space of A. The value 1 is returned if the set is
        orthogonal. The value 0 is returned if the set is not orthogonal.
    
        For example, if the set of row vectors (u1, u2, ...) is to be determined for
        orthogonality, set A to be equal to np.array([u1, u2, ...]).T.
        """
        matrix_size = A.shape
        n = matrix_size[1]
        tolerance = 1e-10
    
        if n == 1:
            return 1
        else:
            G = A.T @ A - np.eye(n)
            if np.max(np.abs(G)) <= tolerance:
                return 1
            else:
                return 0
    
    def is_orthonormal_set(self, A):
        """
        Orthonormal Sets
    
        Determines if a set of vectors in Euclidean n-space is orthonormal. The matrix A
        is formed from these vectors as its columns. That is, the subspace spanned by the
        set of vectors is the column space of A. The value 1 is returned if the set is
        orthonormal. The value 0 is returned if the set is not orthonormal. An error is
        returned if a set containing only zero vectors is attempted to be determined for
        orthonormality.
    
        For example, if the set of row vectors (u1, u2, ...) is to be determined for
        orthonormality, set A to be equal to np.array([u1, u2, ...]).T.
        """
        matrix_size = A.shape
        m = matrix_size[0]
        n = matrix_size[1]
        tolerance = 1e-10
    
        if np.allclose(A, np.zeros((m, n))):
            raise ValueError('The set that contains just zero vectors cannot be orthonormal.')
        elif n == 1:
            if np.abs(np.linalg.norm(A[:, 0]) - 1) <= tolerance:
                return 1
            else:
                return 0
        else:
            if self.is_orthogonal_set(A) == 1:
                length_counter = 0
                for i in range(n):
                    if np.abs(np.linalg.norm(A[:, i]) - 1) <= tolerance:
                        length_counter += 1
    
                if length_counter == n:
                    return 1
                else:
                    return 0
            else:
                return 0



        
    def Gram_Schmidt_process(self, A):
        """
        Gram-Schmidt Process
        
        Gram_Schmidt_process(A) produces an orthonormal basis for the subspace of
        Eucldiean n-space spanned by the vectors {u1,u2,...}, where the matrix A 
        is formed from these vectors as its columns. That is, the subspace is the
        column space of A. The columns of the matrix that is returned are the 
        orthonormal basis vectors for the subspace. An error is returned if an
        orthonormal basis for the zero vector space is attempted to be produced.
    
        For example, if the vector space V = span{u1,u2,...}, where u1,u2,... are
        row vectors, then set A to be [u1' u2' ...].
    
        For example, if the vector space V = Col(B), where B is an m x n matrix,
        then set A to be equal to B.
        """
        matrix_size = np.shape(A)
    
        m = matrix_size[0]
        n = matrix_size[1]
    
        if np.array_equal(A, np.zeros((m,n))):
            raise ValueError('There does not exist any type of basis for the zero vector space.')
        elif n == 1:
            orthonormal_basis = A[:, 0]/np.linalg.norm(A[:, 0])
        else:
            flag = 0
    
            if self.is_orthonormal_set(A) == 1:
                self.orthonormal_basis = A
                flag = 1
    
            if flag == 0:
                if np.linalg.matrix_rank(A) != n:
                    A = self.basis_col(A)
                
                matrix_size = np.shape(A)
                m = matrix_size[0]
                n = matrix_size[1]
    
                orthonormal_basis = A[:, 0]/np.linalg.norm(A[:, 0])
    
                for i in range(1, n):
                    u = A[:, i]
                    v = np.zeros((m, 1))
    
                    for j in range(i):
                        v -= np.dot(u, orthonormal_basis[:, j]) * orthonormal_basis[:, j]
    
                    v_ = u + v
                    orthonormal_basis[:, i] = v_/np.linalg.norm(v_)
    
        return orthonormal_basis
    
    def train_test_split_data_ctl(self, n, binsize, timestep):
        
        # Splits data into train and test sets according to exisiting indices
        
        r_train_idx, l_train_idx, r_test_idx, l_test_idx = self.r_train_idx, self.l_train_idx, self.r_test_idx, self.l_test_idx

        
        r_train, time, _ = self.get_PSTH(n, self.r_train_idx, binsize=binsize, timestep=timestep)
        l_train, _, _ = self.get_PSTH(n, self.l_train_idx, binsize=binsize, timestep=timestep)
        r_test, _, _ = self.get_PSTH(n, self.r_test_idx, binsize=binsize, timestep=timestep)
        l_test, _, _ = self.get_PSTH(n, self.l_test_idx, binsize=binsize, timestep=timestep)
        
        self.t = time
        
        return r_train, l_train, r_test, l_test    
    
    def train_test_split_data_opto(self, n, binsize, timestep):
        
        # Splits data into train and test sets according to exisiting indices
        
        r_train_idx, l_train_idx, r_test_idx, l_test_idx = self.r_train_opto_idx, self.l_train_opto_idx, self.r_test_opto_idx, self.l_test_opto_idx
    
        r_train, _, _ = self.get_PSTH(n, self.r_train_opto_idx, binsize=binsize, timestep=timestep)
        l_train, _, _ = self.get_PSTH(n, self.l_train_opto_idx, binsize=binsize, timestep=timestep)
        r_test, _, _ = self.get_PSTH(n, self.r_test_opto_idx, binsize=binsize, timestep=timestep)
        l_test, _, _ = self.get_PSTH(n, self.l_test_opto_idx, binsize=binsize, timestep=timestep)
        
        return r_train, l_train, r_test, l_test    
    
    def train_test_split_data_err(self, n, binsize, timestep):
        
        # Splits data into train and test sets (50/50 split)
        
        r_train_idx, l_train_idx, r_test_idx, l_test_idx = self.r_train_err_idx, self.l_train_err_idx, self.r_test_err_idx, self.l_test_err_idx
        
        r_train, _, _ = self.get_PSTH(n, self.r_train_err_idx, binsize=binsize, timestep=timestep)
        l_train, _, _ = self.get_PSTH(n, self.l_train_err_idx, binsize=binsize, timestep=timestep)
        r_test, _, _ = self.get_PSTH(n, self.r_test_err_idx, binsize=binsize, timestep=timestep)
        l_test, _, _ = self.get_PSTH(n, self.l_test_err_idx, binsize=binsize, timestep=timestep)
        
        return r_train, l_train, r_test, l_test
    
    def func_compute_activity_modes_DRT(self, input_, ctl=True, lickdir=False, use_LDA=False):
    
        # Inputs: Left Right Correct Error traces of ALL neurons that are selective
        #           time stamps for analysis?
        #           time epochs
        # Outputs: Orthonormal basis (nxn) where n = # of neurons
        #           activity variance of each dimension (nx1)
        
        # Actual method uses SVD decomposition
        
        T_cue_aligned_sel = self.T_cue_aligned_sel 
        time_epochs = self.time_epochs
    
        t_sample = time_epochs[0]
        t_delay = time_epochs[1]
        t_response = time_epochs[2]
        
        if ctl:
            PSTH_yes_correct, PSTH_no_correct = input_
            activityRL = np.concatenate((PSTH_yes_correct, PSTH_no_correct), axis=1)

        else:
            PSTH_yes_correct, PSTH_no_correct, PSTH_yes_error, PSTH_no_error = input_
            activityRL = np.concatenate((PSTH_yes_correct, PSTH_no_correct, PSTH_yes_error, PSTH_no_error), axis=1)

    
        activityRL = activityRL - np.mean(activityRL, axis=1, keepdims=True) # remove?
        u, s, v = np.linalg.svd(activityRL.T)
        proj_allDim = activityRL.T @ v
    
        # Variance of each dimension normalized
        var_s = np.square(np.diag(s[0:proj_allDim.shape[1]]))
        var_allDim = var_s / np.sum(var_s)
    
        # Relevant choice dims
        CD_stim_mode = [] # Sample period
        CD_choice_mode = [] # Late delay period
        CD_outcome_mode = [] # Response period
        CD_sample_mode = [] # wt during the first 400 ms of the sample epoch
        CD_delay_mode = []
        CD_go_mode = []
        Ramping_mode = []
        GoDirection_mode = [] # To calculate the go direction (GD), we subtracted (rlick-right, t + rlick-left, t)/2 after the Go cue (Tgo < t < Tgo + 0.1 s) from that before the Go cue (Tgo - 0.1 s < t < Tgo), followed by normalization by its own norm. 
        
        if ctl:
            
            wt = (PSTH_yes_correct - PSTH_no_correct)/2
            i_t = np.where((T_cue_aligned_sel > t_sample) & (T_cue_aligned_sel < t_delay))[0]
            CD_stim_mode = np.mean(wt[:, i_t], axis=1)
        
            wt = (PSTH_yes_correct - PSTH_no_correct)/2
            i_t = np.where((T_cue_aligned_sel > t_delay) & (T_cue_aligned_sel < t_response))[0]
            CD_choice_mode = np.mean(wt[:, i_t], axis=1)
            
            wt = (PSTH_yes_correct + PSTH_no_correct)/2
            i_t = np.where((T_cue_aligned_sel > t_response) & (T_cue_aligned_sel < (t_response + 1.3)))[0]
            CD_outcome_mode = np.mean(wt[:, i_t], axis=1)
            
           
            wt = (PSTH_yes_correct - PSTH_no_correct)/2
            i_t = np.where((T_cue_aligned_sel > (t_sample+0.2)) & (T_cue_aligned_sel < (t_sample+0.4)))[0]
            CD_sample_mode = np.mean(wt[:, i_t], axis=1)
            
            i_t = np.where((T_cue_aligned_sel > (t_response-0.3)) & (T_cue_aligned_sel < (t_response-0.1)))[0]
            CD_delay_mode = np.mean(wt[:, i_t], axis=1)
            
            i_t = np.where((T_cue_aligned_sel > (t_response+0.1)) & (T_cue_aligned_sel < (t_response+0.3)))[0]
            CD_go_mode = np.mean(wt[:, i_t], axis=1)
            
            wt = (PSTH_yes_correct + PSTH_no_correct)/2
            i_t1 = np.where((T_cue_aligned_sel > (t_sample-0.3)) & (T_cue_aligned_sel < (t_sample-0.1)))[0]
            i_t2 = np.where((T_cue_aligned_sel > (t_response-0.3)) & (T_cue_aligned_sel < (t_response-0.1)))[0]
            Ramping_mode = np.mean(wt[:, i_t2], axis=1) - np.mean(wt[:, i_t1], axis=1)
            
            i_t1 = np.where((T_cue_aligned_sel > (t_response-0.1)) & (T_cue_aligned_sel < t_response))[0]
            i_t2 = np.where((T_cue_aligned_sel > t_response) & (T_cue_aligned_sel < (t_response+0.1)))[0]
            GoDirection_mode = np.mean(wt[:, i_t2], axis=1) - np.mean(wt[:, i_t1], axis=1)
            
        elif lickdir:
            
            wt = (PSTH_yes_correct + PSTH_no_error) / 2 - (PSTH_no_correct + PSTH_yes_error) / 2
            i_t = np.where((T_cue_aligned_sel > t_sample) & (T_cue_aligned_sel < t_delay))[0]
            CD_stim_mode = np.mean(wt[:, i_t], axis=1)
            
            wt = (PSTH_yes_correct + PSTH_no_error) / 2 - (PSTH_no_correct + PSTH_yes_error) / 2
            i_t = np.where((T_cue_aligned_sel > t_delay) & (T_cue_aligned_sel < t_response))[0]
            CD_choice_mode = np.mean(wt[:, i_t], axis=1)
            
            wt = (PSTH_yes_correct + PSTH_no_correct) / 2 - (PSTH_yes_error + PSTH_no_error) / 2
            i_t = np.where((T_cue_aligned_sel > t_response) & (T_cue_aligned_sel < (t_response)))[0]
            CD_outcome_mode = np.mean(wt[:, i_t], axis=1)
            
            
            wt = PSTH_yes_correct - PSTH_no_correct
            i_t = np.where((T_cue_aligned_sel > (t_sample+0.2)) & (T_cue_aligned_sel < (t_sample+0.4)))[0]
            CD_sample_mode = np.mean(wt[:, i_t], axis=1)
            
            i_t = np.where((T_cue_aligned_sel > (t_response-0.3)) & (T_cue_aligned_sel < (t_response-0.1)))[0]
            CD_delay_mode = np.mean(wt[:, i_t], axis=1)
            
            i_t = np.where((T_cue_aligned_sel > (t_response+0.1)) & (T_cue_aligned_sel < (t_response+0.3)))[0]
            CD_go_mode = np.mean(wt[:, i_t], axis=1)
            
            wt = (PSTH_yes_correct + PSTH_no_correct)/2
            i_t1 = np.where((T_cue_aligned_sel > (t_sample-0.3)) & (T_cue_aligned_sel < (t_sample-0.1)))[0]
            i_t2 = np.where((T_cue_aligned_sel > (t_response-0.3)) & (T_cue_aligned_sel < (t_response-0.1)))[0]
            Ramping_mode = np.mean(wt[:, i_t2], axis=1) - np.mean(wt[:, i_t1], axis=1)
            
            i_t1 = np.where((T_cue_aligned_sel > (t_response-0.1)) & (T_cue_aligned_sel < t_response))[0]
            i_t2 = np.where((T_cue_aligned_sel > t_response) & (T_cue_aligned_sel < (t_response+0.1)))[0]
            GoDirection_mode = np.mean(wt[:, i_t2], axis=1) - np.mean(wt[:, i_t1], axis=1)
        
        # elif use_LDA: # not implemented
            
            # # wt = (PSTH_yes_correct + PSTH_yes_error) / 2 - (PSTH_no_correct + PSTH_no_error) / 2
            # i_t = np.where((T_cue_aligned_sel > t_sample) & (T_cue_aligned_sel < t_delay))[0]
            # x = np.vstack((((PSTH_yes_correct + PSTH_yes_error) / 2)[:, i_t], ((PSTH_no_correct + PSTH_no_error) / 2)[:, i_t]))
            # y = cat((np.zeros((PSTH_yes_correct + PSTH_yes_error).shape[0]), np.ones((PSTH_no_correct + PSTH_no_error).shape[0])))
            # clf=LDA()
            
            
            # CD_stim_mode = np.mean(wt[:, i_t], axis=1)
        
            # wt = (PSTH_yes_correct + PSTH_no_error) / 2 - (PSTH_no_correct + PSTH_yes_error) / 2
            # i_t = np.where((T_cue_aligned_sel > t_delay) & (T_cue_aligned_sel < t_response))[0]
            # CD_choice_mode = np.mean(wt[:, i_t], axis=1)
            
            # wt = (PSTH_yes_correct + PSTH_no_correct) / 2 - (PSTH_yes_error + PSTH_no_error) / 2
            # i_t = np.where((T_cue_aligned_sel > t_response) & (T_cue_aligned_sel < (t_response)))[0]
            # CD_outcome_mode = np.mean(wt[:, i_t], axis=1)
            
           
            # wt = PSTH_yes_correct - PSTH_no_correct
            # i_t = np.where((T_cue_aligned_sel > (t_sample)) & (T_cue_aligned_sel < (t_sample)))[0]
            # CD_sample_mode = np.mean(wt[:, i_t], axis=1)
            
            # i_t = np.where((T_cue_aligned_sel > (t_response)) & (T_cue_aligned_sel < (t_response)))[0]
            # CD_delay_mode = np.mean(wt[:, i_t], axis=1)
            
            # i_t = np.where((T_cue_aligned_sel > (t_response)) & (T_cue_aligned_sel < (t_response)))[0]
            # CD_go_mode = np.mean(wt[:, i_t], axis=1)
            
            # wt = (PSTH_yes_correct + PSTH_no_correct)/2
            # i_t1 = np.where((T_cue_aligned_sel > (t_sample-3)) & (T_cue_aligned_sel < (t_sample-1)))[0]
            # i_t2 = np.where((T_cue_aligned_sel > (t_response-3)) & (T_cue_aligned_sel < (t_response-1)))[0]
            # Ramping_mode = np.mean(wt[:, i_t2], axis=1) - np.mean(wt[:, i_t1], axis=1)
            
            # i_t1 = np.where((T_cue_aligned_sel > (t_response-2)) & (T_cue_aligned_sel < t_response))[0]
            # i_t2 = np.where((T_cue_aligned_sel > t_response) & (T_cue_aligned_sel < (t_response+2)))[0]
            # GoDirection_mode = np.mean(wt[:, i_t2], axis=1) - np.mean(wt[:, i_t1], axis=1)
            
        else: # Lickdir except for stim mode, STANDARD
            
            
            wt = (PSTH_yes_correct + PSTH_yes_error) / 2 - (PSTH_no_correct + PSTH_no_error) / 2
            i_t = np.where((T_cue_aligned_sel > t_sample) & (T_cue_aligned_sel < t_delay))[0]
            CD_stim_mode = np.mean(wt[:, i_t], axis=1)
        
            wt = (PSTH_yes_correct + PSTH_no_error) / 2 - (PSTH_no_correct + PSTH_yes_error) / 2
            i_t = np.where((T_cue_aligned_sel > t_delay) & (T_cue_aligned_sel < t_response))[0]
            CD_choice_mode = np.mean(wt[:, i_t], axis=1)
            
            wt = (PSTH_yes_correct + PSTH_no_correct) / 2 - (PSTH_yes_error + PSTH_no_error) / 2
            i_t = np.where((T_cue_aligned_sel > t_response) & (T_cue_aligned_sel < (t_response)))[0]
            CD_outcome_mode = np.mean(wt[:, i_t], axis=1)
            
           
            wt = PSTH_yes_correct - PSTH_no_correct
            i_t = np.where((T_cue_aligned_sel > (t_sample+0.2)) & (T_cue_aligned_sel < (t_sample+0.4)))[0]
            CD_sample_mode = np.mean(wt[:, i_t], axis=1)
            
            i_t = np.where((T_cue_aligned_sel > (t_response-0.3)) & (T_cue_aligned_sel < (t_response-0.1)))[0]
            CD_delay_mode = np.mean(wt[:, i_t], axis=1)
            
            i_t = np.where((T_cue_aligned_sel > (t_response+0.1)) & (T_cue_aligned_sel < (t_response+0.3)))[0]
            CD_go_mode = np.mean(wt[:, i_t], axis=1)
            
            wt = (PSTH_yes_correct + PSTH_no_correct)/2
            i_t1 = np.where((T_cue_aligned_sel > (t_sample-0.3)) & (T_cue_aligned_sel < (t_sample-0.1)))[0]
            i_t2 = np.where((T_cue_aligned_sel > (t_response-0.3)) & (T_cue_aligned_sel < (t_response-0.1)))[0]
            Ramping_mode = np.mean(wt[:, i_t2], axis=1) - np.mean(wt[:, i_t1], axis=1)
            
            i_t1 = np.where((T_cue_aligned_sel > (t_response-0.1)) & (T_cue_aligned_sel < t_response))[0]
            i_t2 = np.where((T_cue_aligned_sel > t_response) & (T_cue_aligned_sel < (t_response+0.1)))[0]
            GoDirection_mode = np.mean(wt[:, i_t2], axis=1) - np.mean(wt[:, i_t1], axis=1)

        
        CD_stim_mode = CD_stim_mode / np.linalg.norm(CD_stim_mode)
        CD_choice_mode = CD_choice_mode / np.linalg.norm(CD_choice_mode)
        CD_outcome_mode = CD_outcome_mode / np.linalg.norm(CD_outcome_mode)
        CD_sample_mode = CD_sample_mode / np.linalg.norm(CD_sample_mode)
        CD_delay_mode = CD_delay_mode / np.linalg.norm(CD_delay_mode)
        CD_go_mode = CD_go_mode / np.linalg.norm(CD_go_mode)
        Ramping_mode = Ramping_mode / np.linalg.norm(Ramping_mode)
        GoDirection_mode = GoDirection_mode / np.linalg.norm(GoDirection_mode)
        
        # Reshape all activity modes
        
        CD_stim_mode = np.reshape(CD_stim_mode, (-1, 1)) 
        CD_choice_mode = np.reshape(CD_choice_mode, (-1, 1)) 
        CD_outcome_mode = np.reshape(CD_outcome_mode, (-1, 1))
        CD_sample_mode = np.reshape(CD_sample_mode, (-1, 1)) 
        CD_delay_mode = np.reshape(CD_delay_mode, (-1, 1)) 
        CD_go_mode = np.reshape(CD_go_mode, (-1, 1)) 
        Ramping_mode = np.reshape(Ramping_mode, (-1, 1)) 
        GoDirection_mode = np.reshape(GoDirection_mode, (-1, 1)) 
        
        start_time = time.time()
        input_ = np.concatenate((CD_stim_mode, CD_choice_mode, CD_outcome_mode, CD_sample_mode, CD_delay_mode, CD_go_mode, Ramping_mode, GoDirection_mode, v), axis=1)
        # orthonormal_basis = self.Gram_Schmidt_process(input_)
        orthonormal_basis, _ = np.linalg.qr(input_, mode='complete')  # lmao
        
        proj_allDim = np.dot(activityRL.T, orthonormal_basis)
        var_allDim = np.sum(proj_allDim**2, axis=0)
        var_allDim = var_allDim[~np.isnan(var_allDim)]
        
        var_allDim = var_allDim / np.sum(var_allDim)
        
        print("Runtime: {} secs".format(time.time() - start_time))
        return orthonormal_basis, var_allDim
    
    def KD_LDA2(self, ll, rr, rs=None):

        if rs is not None:
            x = np.vstack((ll, rr, rs))
            groupVar = np.vstack((np.ones((ll.shape[0], 1)),2 * np.ones((rr.shape[0], 1)), 3 * np.ones((rs.shape[0], 1))))
        else:
            x = np.vstack((ll, rr))
            groupVar = np.vstack((np.ones((ll.shape[0], 1)), 2 * np.ones((rr.shape[0], 1))))
    
        xm = np.mean(x, axis=0)
        n = x.shape[0]
        x = x - np.tile(xm, (n, 1))
        T = x.T @ x
    
        # Now compute the Within sum of squares matrix
        W = np.zeros_like(T)
        for j in range(1, 3):
            r = np.where(groupVar == j)[0]
            nr = len(r)
            if nr > 1:
                z = x[r, :]
                xm = np.mean(z, axis=0)
                z = z - np.tile(xm, (nr, 1))
                W += z.T @ z
    
        B = T - W
    
        U, S, Vt = np.linalg.svd(np.linalg.pinv(W) @ B)
        v = Vt.T
    
        cr = rr @ v
        cl = ll @ v
        sgn = np.sign(np.mean(cr[:, 0]) - np.mean(cl[:, 0]))
        v[:, 0] *= sgn
    
        return v
    
    def func_compute_epoch_decoder(self, input_, epoch, ctl=True):
    
        # Inputs: Left Right Correct Error traces of ALL neurons that are selective
        #           time stamps for analysis?
        #           time epochs
        # Outputs: Orthonormal basis (nxn) where n = # of neurons
        #           activity variance of each dimension (nx1)
        
        # Actual method uses SVD decomposition
        
        if ctl:
            PSTH_yes_correct, PSTH_no_correct = input_
        else:
            PSTH_yes_correct, PSTH_no_correct, PSTH_yes_error, PSTH_no_error = input_
    
        # CD_all = self.KD_LDA2(self.PSTH_r_train_correct, self.PSTH_l_train_correct, rs=None)
        CD_all = []
        for t in epoch:
            # CD_all = LDA().fit(np.vstack((self.PSTH_r_train_correct[:,t], self.PSTH_l_train_correct[:,t])).T, [0,1])
                         # cat((np.ones(self.PSTH_r_train_correct.shape[0]), np.zeros(self.PSTH_l_train_correct.shape[0]))))

            CD_all += [(PSTH_yes_correct[:,t] - PSTH_no_correct[:,t]) / 2]
        CD_choice_mode = np.mean(CD_all, axis=0)          
        # return CD_choice_mode, 0
        
        activityRL = np.concatenate((PSTH_yes_correct, PSTH_no_correct), axis=1)
        activityRL = activityRL - np.mean(activityRL, axis=1, keepdims=True) # remove?
        u, s, v = np.linalg.svd(activityRL.T)
        proj_allDim = activityRL.T @ v
    
        # Variance of each dimension normalized
        var_s = np.square(np.diag(s[0:proj_allDim.shape[1]]))
        var_allDim = var_s / np.sum(var_s)
    
        # Relevant choice dims
        # CD_choice_mode = [] # Late delay period
        
        CD_choice_mode = CD_choice_mode / np.linalg.norm(CD_choice_mode)
        # return CD_choice_mode, 0
        # Reshape 
        
        CD_choice_mode = np.reshape(CD_choice_mode, (-1, 1)) 

        start_time = time.time()
        input_ = np.concatenate((CD_choice_mode, v), axis=1)
        # orthonormal_basis = self.Gram_Schmidt_process(input_)
        orthonormal_basis, _ = np.linalg.qr(input_, mode='complete')  # lmao
        
        proj_allDim = np.dot(activityRL.T, orthonormal_basis)
        var_allDim = np.sum(proj_allDim**2, axis=0)
        var_allDim = var_allDim[~np.isnan(var_allDim)]
        
        var_allDim = var_allDim / np.sum(var_allDim)
        
        print("Runtime: {} secs".format(time.time() - start_time))
        return orthonormal_basis, var_allDim
    
    def func_compute_persistent_decoder(self, input_, epoch):
    
        # Inputs: Left Right Correct Error traces of ALL neurons that are selective
        #           time stamps for analysis?
        #           time epochs
        # Outputs: Orthonormal basis (nxn) where n = # of neurons
        #           activity variance of each dimension (nx1)
        
        # Actual method uses SVD decomposition
        
        PSTH_yes_correct, PSTH_no_correct, PSTH_yes_opto, PSTH_no_opto = input_
    
        # CD_all = self.KD_LDA2(self.PSTH_r_train_correct, self.PSTH_l_train_correct, rs=None)
        CD_all = []
        for t in epoch:
            # CD_all = LDA().fit(np.vstack((self.PSTH_r_train_correct[:,t], self.PSTH_l_train_correct[:,t])).T, [0,1])
                         # cat((np.ones(self.PSTH_r_train_correct.shape[0]), np.zeros(self.PSTH_l_train_correct.shape[0]))))

            CD_all += [((PSTH_yes_correct[:,t] + PSTH_no_correct[:,t]) / 2) - 
                       ((PSTH_yes_opto[:,t] + PSTH_no_opto[:,t]) / 2)]
        CD_choice_mode = np.mean(CD_all, axis=0)          
        # return CD_choice_mode, 0
        
        activityRL = np.concatenate([PSTH_yes_correct, PSTH_no_correct, PSTH_yes_opto, PSTH_no_opto], axis=1)
        activityRL = activityRL - np.mean(activityRL, axis=1, keepdims=True) # remove?
        u, s, v = np.linalg.svd(activityRL.T)
        proj_allDim = activityRL.T @ v
    
        # Variance of each dimension normalized
        var_s = np.square(np.diag(s[0:proj_allDim.shape[1]]))
        var_allDim = var_s / np.sum(var_s)
    
        # Relevant choice dims
        # CD_choice_mode = [] # Late delay period
        
        CD_choice_mode = CD_choice_mode / np.linalg.norm(CD_choice_mode)
        # return CD_choice_mode, 0
        # Reshape 
        
        CD_choice_mode = np.reshape(CD_choice_mode, (-1, 1)) 

        start_time = time.time()
        input_ = np.concatenate((CD_choice_mode, v), axis=1)
        # orthonormal_basis = self.Gram_Schmidt_process(input_)
        orthonormal_basis, _ = np.linalg.qr(input_, mode='complete')  # lmao
        
        proj_allDim = np.dot(activityRL.T, orthonormal_basis)
        var_allDim = np.sum(proj_allDim**2, axis=0)
        var_allDim = var_allDim[~np.isnan(var_allDim)]
        
        var_allDim = var_allDim / np.sum(var_allDim)
        
        print("Runtime: {} secs".format(time.time() - start_time))
        return orthonormal_basis, var_allDim
    
    
    def plot_CDalt(self, epoch=None, lickdir=False, save=None, plot=True):
        """
        This method doesn't orthogonalize against the other modes

        Parameters
        ----------
        epoch : TYPE, optional
            DESCRIPTION. The default is None.
        save : TYPE, optional
            DESCRIPTION. The default is None.
        plot : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        orthonormal_basis : TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.

        """
        if epoch is not None:
            orthonormal_basis, var_allDim = self.func_compute_epoch_decoder([self.PSTH_r_train_correct, 
                                                                            self.PSTH_l_train_correct], epoch)
        else:
            
            orthonormal_basis, var_allDim = self.func_compute_epoch_decoder([self.PSTH_r_train_correct, 
                                                                            self.PSTH_l_train_correct], range(self.delay+int(1.5*1/self.fs), self.response))
        


        activityRL_train= np.concatenate((self.PSTH_r_train_correct, 
                                        self.PSTH_l_train_correct), axis=1)

        activityRL_test= np.concatenate((self.PSTH_r_test_correct, 
                                        self.PSTH_l_test_correct), axis=1)
        
        r_corr = np.where(self.R_correct)[0]
        l_corr = np.where(self.L_correct)[0]
        
        r_trials = [i for i in r_corr if i in self.i_good_non_stim_trials and not self.early_lick[i]]
        l_trials = [i for i in l_corr if i in self.i_good_non_stim_trials and not self.early_lick[i]]

        
        x = np.arange(-6.97,4,self.fs)[:self.time_cutoff]

        # orthonormal_basis = orthonormal_basis.reshape(-1,1)
        i_pc = 0

        # Project for every trial
        for t in self.r_test_idx:
            activity = self.dff[0, r_trials[t]][self.good_neurons] 
            activity = activity - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activity.shape[1]))
            proj_allDim = np.dot(activity.T, orthonormal_basis)
            plt.plot(x, proj_allDim[:len(self.T_cue_aligned_sel), i_pc], 'b', alpha = 0.5,  linewidth = 0.5)
            
        for t in self.l_test_idx:
            activity = self.dff[0, l_trials[t]][self.good_neurons]
            activity = activity - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activity.shape[1]))
            proj_allDim = np.dot(activity.T, orthonormal_basis)
            plt.plot(x, proj_allDim[:len(self.T_cue_aligned_sel), i_pc], 'r', alpha = 0.5, linewidth = 0.5)
            
        # Correct trials
        activityRL_test = activityRL_test - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activityRL_test.shape[1]))  # remove mean
        proj_allDim = np.dot(activityRL_test.T, orthonormal_basis)

        
        # ax = axs.flatten()[0]
        plt.plot(x, proj_allDim[:len(self.T_cue_aligned_sel), i_pc], 'b', linewidth = 2)
        plt.plot(x, proj_allDim[len(self.T_cue_aligned_sel):, i_pc], 'r', linewidth = 2)
        plt.title("Choice decoder projections")
        plt.axvline(self.sample, color = 'grey', alpha=0.5, ls = '--')
        plt.axvline(self.delay, color = 'grey', alpha=0.5, ls = '--')
        plt.axvline(0, color = 'grey', alpha=0.5, ls = '--')
        plt.ylabel('CD_delay projection (a.u.)')
        
        if save is not None:
            plt.savefig(save)
            
        plt.show()
        # axs[0, 0].set_ylabel('Activity proj.')
        # axs[3, 0].set_xlabel('Time')
        
        return orthonormal_basis, np.mean(activityRL_train, axis=1)[:, None]
    
    def plot_CD(self, mode_input='choice', epoch=None, ctl=False, lickdir=False, 
                save=None, plot=True, remove_top = False, auto_corr_return=False,
                fix_axis=None, remove_n = [], single_trial = False,
                return_traces = False):
        "This method orthogonalizes the various modes"

        
        
        if ctl:
            orthonormal_basis, var_allDim = self.func_compute_activity_modes_DRT([self.PSTH_r_train_correct, 
                                                                                self.PSTH_l_train_correct], ctl=ctl, 
                                                                                lickdir=lickdir)
        else:
            orthonormal_basis, var_allDim = self.func_compute_activity_modes_DRT([self.PSTH_r_train_correct, 
                                                                                self.PSTH_l_train_correct, 
                                                                                self.PSTH_r_train_error, 
                                                                                self.PSTH_l_train_error], ctl=ctl, 
                                                                                lickdir=lickdir)           
            
        activityRL_train= np.concatenate((self.PSTH_r_train_correct, 
                                        self.PSTH_l_train_correct, 
                                        self.PSTH_r_train_error, 
                                        self.PSTH_l_train_error), axis=1)
        good_neurons = self.good_neurons
        
        activityRL_test= np.concatenate((self.PSTH_r_test_correct, 
                                        self.PSTH_l_test_correct), axis=1)
        
        idx_map = {'choice': 1, 'action':5, 'stimulus':0, 'ramping':7}
        idx = idx_map[mode_input]

        orthonormal_basis = orthonormal_basis[:, idx]
        
        # if remove_top:
        #     bottom_idx = np.argsort(orthonormal_basis)[:-10] # Remove top 10 contributors
            
        #     good_neurons = self.good_neurons[bottom_idx]
        #     orthonormal_basis = orthonormal_basis[bottom_idx]
            
        #     activityRL_train= np.concatenate((self.PSTH_r_train_correct[bottom_idx], 
        #                                     self.PSTH_l_train_correct[bottom_idx]), axis=1)
    
        #     activityRL_test= np.concatenate((self.PSTH_r_test_correct[bottom_idx], 
        #                                     self.PSTH_l_test_correct[bottom_idx]), axis=1)
        # elif len(remove_n) != 0:
            
        #     # keep all those not in remove_n
        #     keep_n = [i for i in np.arange(len(self.good_neurons)) if i not in remove_n]
            
        #     good_neurons = self.good_neurons[keep_n]
        #     orthonormal_basis = orthonormal_basis[keep_n]
            
        #     activityRL_train= np.concatenate((self.PSTH_r_train_correct[keep_n], 
        #                                     self.PSTH_l_train_correct[keep_n]), axis=1)
    
        #     activityRL_test= np.concatenate((self.PSTH_r_test_correct[keep_n], 
        #                                     self.PSTH_l_test_correct[keep_n]), axis=1)
            
        
        # else:
        #     good_neurons = self.good_neurons
            

        #     activityRL_train= np.concatenate((self.PSTH_r_train_correct, 
        #                                     self.PSTH_l_train_correct), axis=1)
    
        #     activityRL_test= np.concatenate((self.PSTH_r_test_correct, 
        #                                     self.PSTH_l_test_correct), axis=1)
        


        
        x = self.t
        if single_trial:
            
            #FIXME: needs to be fixed from imaging to ephys
            
            r_corr = np.where(self.R_correct)[0]
            l_corr = np.where(self.L_correct)[0]
            
            r_trials = [i for i in r_corr if i in self.i_good_non_stim_trials and not self.early_lick[i]]
            l_trials = [i for i in l_corr if i in self.i_good_non_stim_trials and not self.early_lick[i]]
            
            proj_allDimR = []
            # Project for every trial
            for t in self.r_test_idx:
                activity = self.dff[0, r_trials[t]][good_neurons] 
                activity = activity - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activity.shape[1]))
                proj_allDim = np.dot(activity.T, orthonormal_basis)
                proj_allDimR += [proj_allDim[:len(self.T_cue_aligned_sel)]]
    
                if plot:
                    plt.plot(x, proj_allDim[:len(self.T_cue_aligned_sel)], 'b', alpha = 0.5,  linewidth = 0.5)
            
            proj_allDimL = []
            for t in self.l_test_idx:
                activity = self.dff[0, l_trials[t]][good_neurons]
                activity = activity - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activity.shape[1]))
                proj_allDim = np.dot(activity.T, orthonormal_basis)
                proj_allDimL += [proj_allDim[:len(self.T_cue_aligned_sel)]]
    
                if plot:
                    plt.plot(x, proj_allDim[:len(self.T_cue_aligned_sel)], 'r', alpha = 0.5, linewidth = 0.5)
                    
            if auto_corr_return:
                
                return proj_allDimR, proj_allDimL
                
            # Correct trials
            activityRL_test = activityRL_test - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activityRL_test.shape[1]))  # remove mean
            proj_allDim = np.dot(activityRL_test.T, orthonormal_basis)
    
            
            if plot:
                plt.plot(x, proj_allDim[:len(self.T_cue_aligned_sel)], 'b', linewidth = 2)
                plt.plot(x, proj_allDim[len(self.T_cue_aligned_sel):], 'r', linewidth = 2)
                plt.title('{} decoder projections'.format(mode_input))
                plt.axvline(self.sample, color = 'grey', alpha=0.5, ls = '--')
                plt.axvline(self.delay, color = 'grey', alpha=0.5, ls = '--')
                plt.axvline(0, color = 'grey', alpha=0.5, ls = '--')
                plt.ylabel('CD_{} projection (a.u.)'.format(mode_input))
                if fix_axis is not None:
                    plt.ylim(fix_axis)
            
            if save is not None:
                plt.savefig(save)
            if plot:
                plt.show()

        else:
            
            # Correct trials
            activityRL_test = activityRL_test - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activityRL_test.shape[1]))  # remove mean
            proj_allDim = np.dot(activityRL_test.T, orthonormal_basis)
            
            
            if plot:
                
                plt.plot(x, proj_allDim[:len(self.T_cue_aligned_sel)], 'b', linewidth = 2)
                plt.plot(x, proj_allDim[len(self.T_cue_aligned_sel):], 'r', linewidth = 2)
                plt.title('{} decoder projections'.format(mode_input))
                plt.axvline(self.sample, color = 'grey', alpha=0.5, ls = '--')
                plt.axvline(self.delay, color = 'grey', alpha=0.5, ls = '--')
                plt.axvline(self.response, color = 'grey', alpha=0.5, ls = '--')
                plt.ylabel('CD_{} projection (a.u.)'.format(mode_input))
                if fix_axis is not None:
                    plt.ylim(fix_axis)
            
            if save is not None:
                plt.savefig(save)
            if plot:
                plt.show()
                
            if return_traces:
                return proj_allDim[:len(self.T_cue_aligned_sel)], proj_allDim[len(self.T_cue_aligned_sel):]
                
        return orthonormal_basis, np.mean(activityRL_train, axis=1)[:, None]
                                                                            
    def plot_activity_modes_err(self):
        # plot activity modes
        # all trials
        

        orthonormal_basis, var_allDim = self.func_compute_activity_modes_DRT([self.PSTH_r_train_correct, 
                                                                            self.PSTH_l_train_correct, 
                                                                            self.PSTH_r_train_error, 
                                                                            self.PSTH_l_train_error], ctl=False)
        
        activityRL_train= np.concatenate((self.PSTH_r_train_correct, 
                                        self.PSTH_l_train_correct, 
                                        self.PSTH_r_train_error, 
                                        self.PSTH_l_train_error), axis=1)
    
        activityRL_test= np.concatenate((self.PSTH_r_test_correct, 
                                        self.PSTH_l_test_correct), axis=1)
        
        activityRLerr_test = np.concatenate((self.PSTH_r_test_error, 
                                             self.PSTH_l_test_error), axis = 1)
        
        
        T_cue_aligned_sel = self.T_cue_aligned_sel
        

        
        
        # Correct trials
        activityRL_test = activityRL_test - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activityRL_test.shape[1]))  # remove mean
        proj_allDim = np.dot(activityRL_test.T, orthonormal_basis)
        
        # Error trials
        activityRLerr_test = activityRLerr_test - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activityRLerr_test.shape[1]))  # remove mean
        proj_allDim_err = np.dot(activityRLerr_test.T, orthonormal_basis)
        
        fig, axs = plt.subplots(4, 4, figsize=(12, 16))
        for i_pc in range(16):
            ax = axs.flatten()[i_pc]
            ax.plot(T_cue_aligned_sel, proj_allDim[:len(T_cue_aligned_sel), i_pc], 'b')
            ax.plot(T_cue_aligned_sel, proj_allDim[len(T_cue_aligned_sel):, i_pc], 'r')
            ax.plot(T_cue_aligned_sel, proj_allDim_err[:len(T_cue_aligned_sel), i_pc], color=[.7, .7, 1])
            ax.plot(T_cue_aligned_sel, proj_allDim_err[len(T_cue_aligned_sel):, i_pc], color=[1, .7, .7])
            ax.set_title("Mode {}".format(i_pc + 1))
            
        axs[0, 0].set_ylabel('Activity proj.')
        axs[3, 0].set_xlabel('Time')
        return proj_allDim[:len(T_cue_aligned_sel), i_pc], proj_allDim[len(T_cue_aligned_sel):, i_pc]
        # plt.show()

    def plot_activity_modes_ctl(self):
        
        T_cue_aligned_sel = self.T_cue_aligned_sel


            
        orthonormal_basis, var_allDim = self.func_compute_activity_modes_DRT([self.PSTH_r_train_correct, 
                                                                            self.PSTH_l_train_correct], ctl = True)
        
        activityRL_train= np.concatenate((self.PSTH_r_train_correct, 
                                        self.PSTH_l_train_correct), axis=1)

        activityRL_test= np.concatenate((self.PSTH_r_test_correct, 
                                        self.PSTH_l_test_correct), axis=1)
        
    
    
        # Correct trials
        activityRL_test = activityRL_test - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activityRL_test.shape[1]))  # remove mean
        proj_allDim = np.dot(activityRL_test.T, orthonormal_basis)
        

        fig, axs = plt.subplots(4, 4, figsize=(12, 16))
        for i_pc in range(16):
            ax = axs.flatten()[i_pc]
            ax.plot(T_cue_aligned_sel, proj_allDim[:len(T_cue_aligned_sel), i_pc], 'b')
            ax.plot(T_cue_aligned_sel, proj_allDim[len(T_cue_aligned_sel):, i_pc], 'r')
            ax.set_title("Mode {}".format(i_pc + 1))
            
        axs[0, 0].set_ylabel('Activity proj.')
        axs[3, 0].set_xlabel('Time')
        
        return proj_allDim[:len(T_cue_aligned_sel), i_pc], proj_allDim[len(T_cue_aligned_sel):, i_pc]


    def plot_activity_modes_opto(self, error = False):
        
        T_cue_aligned_sel = self.T_cue_aligned_sel

            
        # TODO: should I only train on correct trials?
        orthonormal_basis, var_allDim = self.func_compute_activity_modes_DRT([self.PSTH_r_train_correct, 
                                                                            self.PSTH_l_train_correct], ctl = True)
        
        activityRL_train= np.concatenate((self.PSTH_r_train_correct, 
                                        self.PSTH_l_train_correct), axis=1)

        activityRL_test= np.concatenate((self.PSTH_r_test_correct, 
                                        self.PSTH_l_test_correct), axis=1)
        
        activityRLerr_test = np.concatenate((self.PSTH_r_test_opto, 
                                             self.PSTH_l_test_opto), axis = 1)
        
        if error:
            
            activityRLerr_test = np.concatenate((self.PSTH_r_test_opto_err, 
                                                 self.PSTH_l_test_opto_err), axis = 1)
    
    
        # Correct trials
        activityRL_test = activityRL_test - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activityRL_test.shape[1]))  # remove mean
        proj_allDim = np.dot(activityRL_test.T, orthonormal_basis)
        
        # Opto trials
        activityRLerr_test = activityRLerr_test - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activityRLerr_test.shape[1]))  # remove mean
        proj_allDim_err = np.dot(activityRLerr_test.T, orthonormal_basis)
        
        fig, axs = plt.subplots(4, 4, figsize=(12, 16))
        for i_pc in range(16):
            ax = axs.flatten()[i_pc]
            ax.plot(T_cue_aligned_sel, proj_allDim[:len(T_cue_aligned_sel), i_pc], 'b')
            ax.plot(T_cue_aligned_sel, proj_allDim[len(T_cue_aligned_sel):, i_pc], 'r')
            ax.plot(T_cue_aligned_sel, proj_allDim_err[:len(T_cue_aligned_sel), i_pc], color=[.7, .7, 1])
            ax.plot(T_cue_aligned_sel, proj_allDim_err[len(T_cue_aligned_sel):, i_pc], color=[1, .7, .7])
            ax.set_title("Mode {}".format(i_pc + 1))
            
        axs[0, 0].set_ylabel('Activity proj. with opto trials')
        axs[3, 0].set_xlabel('Time')
        
        return proj_allDim[:len(T_cue_aligned_sel), i_pc], proj_allDim[len(T_cue_aligned_sel):, i_pc]

        

    
        
    def plot_behaviorally_relevant_modes(self, plot=True, ctl=False, lickdir=False):
        # plot behaviorally relevant activity modes only
        # separates trials into train vs test sets
        mode_ID = np.array([1, 2, 6, 3, 7, 8, 9])
        mode_name = ['stimulus', 'choice', 'action', 'outcome', 'ramping', 'go', 'response']
        
        if ctl:
            orthonormal_basis, var_allDim = self.func_compute_activity_modes_DRT([self.PSTH_r_train_correct, 
                                                                                self.PSTH_l_train_correct], ctl=ctl, 
                                                                                lickdir=lickdir)
        else:
            orthonormal_basis, var_allDim = self.func_compute_activity_modes_DRT([self.PSTH_r_train_correct, 
                                                                                self.PSTH_l_train_correct, 
                                                                                self.PSTH_r_train_error, 
                                                                                self.PSTH_l_train_error], ctl=ctl, 
                                                                                lickdir=lickdir)           
            
        activityRL_train= np.concatenate((self.PSTH_r_train_correct, 
                                        self.PSTH_l_train_correct, 
                                        self.PSTH_r_train_error, 
                                        self.PSTH_l_train_error), axis=1)

        activityRL_test= np.concatenate((self.PSTH_r_test_correct, 
                                        self.PSTH_l_test_correct), axis=1)
        
        activityRLerr_test = np.concatenate((self.PSTH_r_test_error, 
                                             self.PSTH_l_test_error), axis = 1)
        
        
        T_cue_aligned_sel = self.T_cue_aligned_sel
        
        # Correct trials
        activityRL_test = activityRL_test - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activityRL_test.shape[1]))  # remove mean
        proj_allDim = np.dot(activityRL_test.T, orthonormal_basis)
        var_allDim = np.sum(proj_allDim ** 2, axis=0)
        var_allDim = var_allDim[~np.isnan(var_allDim)]

        var_allDim /= np.sum(var_allDim)
        
        
        # Error trials
        activityRLerr_test = activityRLerr_test - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activityRLerr_test.shape[1]))  # remove mean
        proj_allDim_err = np.dot(activityRLerr_test.T, orthonormal_basis)
        
        if plot:
            plt.figure()
            plt.bar(range(len(mode_ID)), var_allDim[mode_ID-1])
            plt.xticks(range(7), mode_ID)
            plt.xlabel('Activity modes')
            plt.ylabel('Frac var.')
            plt.title(f'Total Cross Validated Var Explained: {np.sum(var_allDim[mode_ID]):.4f}')
            
            n_plot = 0
            plt.figure()
            for i_mode in mode_ID-1:
                n_plot += 1
                print(f'plotting mode {n_plot}')
                proj_iPC_allBtstrp = np.zeros((20, activityRL_test.shape[1]))
                projErr_iPC_allBtstrp = np.zeros((20, activityRLerr_test.shape[1]))
                for i_btstrp in range(20):
                    i_sample = np.random.choice(range(activityRL_test.shape[0]), activityRL_test.shape[0], replace=True)
                    proj_iPC_allBtstrp[i_btstrp,:] = np.dot(activityRL_test[i_sample,:].T, orthonormal_basis[i_sample, i_mode])
                    projErr_iPC_allBtstrp[i_btstrp,:] = np.dot(activityRLerr_test[i_sample,:].T, orthonormal_basis[i_sample, i_mode])
                
                plt.subplot(2, 4, n_plot)
                self.func_plot_mean_and_sem(T_cue_aligned_sel, projErr_iPC_allBtstrp[:,:len(T_cue_aligned_sel)], '#6666ff', '#ccccff', 2)
                self.func_plot_mean_and_sem(T_cue_aligned_sel, projErr_iPC_allBtstrp[:,len(T_cue_aligned_sel):], '#ff6666', '#ffcccc', 2)
                self.func_plot_mean_and_sem(T_cue_aligned_sel, proj_iPC_allBtstrp[:,:len(T_cue_aligned_sel)], 'b', '#9999ff', 2)
                self.func_plot_mean_and_sem(T_cue_aligned_sel, proj_iPC_allBtstrp[:,len(T_cue_aligned_sel):], 'r', '#ff9999', 2)
                
                # y_scale = np.mean(np.concatenate((proj_iPC_allBtstrp, projErr_iPC_allBtstrp)))
                # plt.plot([-2.6,-2.6],[min(y_scale), max(y_scale)]*1.2,'k:') 
                # plt.plot([-1.3,-1.3],[min(y_scale), max(y_scale)]*1.2,'k:')
                # plt.plot([0,0],[min(y_scale), max(y_scale)]*1.2,'k:')
                
                # plt.xlim([-3.2, 2.2])
                plt.title(f'mode {mode_name[n_plot-1]}')
    
            plt.subplot(2, 4, 1)
            plt.ylabel('Activity proj.')
            plt.xlabel('Time')
            plt.show()
        return orthonormal_basis, np.mean(activityRL_train, axis=1)[:, None]
    
    def plot_behaviorally_relevant_modes_opto(self, error=False):
        # plot behaviorally relevant activity modes only
        # separates trials into train vs test sets
        mode_ID = np.array([1, 2, 6, 3, 7, 8, 9])
        mode_name = ['stimulus', 'choice', 'action', 'outcome', 'ramping', 'go', 'response']
        
        orthonormal_basis, var_allDim = self.func_compute_activity_modes_DRT([self.PSTH_r_train_correct, 
                                                                            self.PSTH_l_train_correct], ctl=True)
        
        activityRL_train= np.concatenate((self.PSTH_r_train_correct, 
                                        self.PSTH_l_train_correct), axis=1)

        activityRL_test= np.concatenate((self.PSTH_r_test_correct, 
                                        self.PSTH_l_test_correct), axis=1)
        
        activityRLerr_test = np.concatenate((self.PSTH_r_test_opto, 
                                             self.PSTH_l_test_opto), axis = 1)
        if error:
            
            activityRLerr_test = np.concatenate((self.PSTH_r_test_opto_err, 
                                                 self.PSTH_l_test_opto_err), axis = 1)
        
        T_cue_aligned_sel = self.T_cue_aligned_sel
        
        # Correct trials
        activityRL_test = activityRL_test - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activityRL_test.shape[1]))  # remove mean
        proj_allDim = np.dot(activityRL_test.T, orthonormal_basis)
        var_allDim = np.sum(proj_allDim ** 2, axis=0)
        var_allDim = var_allDim[~np.isnan(var_allDim)]

        var_allDim /= np.sum(var_allDim)
        
        
        # Error trials
        activityRLerr_test = activityRLerr_test - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activityRLerr_test.shape[1]))  # remove mean
        proj_allDim_err = np.dot(activityRLerr_test.T, orthonormal_basis)
        
        plt.figure()
        plt.bar(range(len(mode_ID)), var_allDim[mode_ID-1])
        plt.xticks(range(7), mode_ID)
        plt.xlabel('Activity modes')
        plt.ylabel('Frac var.')
        plt.title(f'Total Cross Validated Var Explained: {np.sum(var_allDim[mode_ID]):.4f}')
        
        n_plot = 0
        plt.figure()
        for i_mode in mode_ID-1:
            n_plot += 1

            print(f'plotting mode {n_plot}')
            
            proj_iPC_allBtstrp = np.zeros((20, activityRL_test.shape[1]))
            projErr_iPC_allBtstrp = np.zeros((20, activityRLerr_test.shape[1]))
            for i_btstrp in range(20):
                i_sample = np.random.choice(range(activityRL_test.shape[0]), activityRL_test.shape[0], replace=True)
                proj_iPC_allBtstrp[i_btstrp,:] = np.dot(activityRL_test[i_sample,:].T, orthonormal_basis[i_sample, i_mode])
                projErr_iPC_allBtstrp[i_btstrp,:] = np.dot(activityRLerr_test[i_sample,:].T, orthonormal_basis[i_sample, i_mode])
            
            plt.subplot(2, 4, n_plot)
            self.func_plot_mean_and_sem(T_cue_aligned_sel, projErr_iPC_allBtstrp[:,:len(T_cue_aligned_sel)], '#6666ff', '#ccccff', 2)
            self.func_plot_mean_and_sem(T_cue_aligned_sel, projErr_iPC_allBtstrp[:,len(T_cue_aligned_sel):], '#ff6666', '#ffcccc', 2)
            self.func_plot_mean_and_sem(T_cue_aligned_sel, proj_iPC_allBtstrp[:,:len(T_cue_aligned_sel)], 'b', '#9999ff', 2)
            self.func_plot_mean_and_sem(T_cue_aligned_sel, proj_iPC_allBtstrp[:,len(T_cue_aligned_sel):], 'r', '#ff9999', 2)
            
            # y_scale = np.mean(np.concatenate((proj_iPC_allBtstrp, projErr_iPC_allBtstrp)))
            # plt.plot([-2.6,-2.6],[min(y_scale), max(y_scale)]*1.2,'k:') 
            # plt.plot([-1.3,-1.3],[min(y_scale), max(y_scale)]*1.2,'k:')
            # plt.plot([0,0],[min(y_scale), max(y_scale)]*1.2,'k:')
            
            # plt.xlim([-3.2, 2.2])
            plt.title(f'mode {mode_name[n_plot-1]}')

        plt.subplot(2, 4, 1)
        plt.ylabel('Activity proj. with opto trials')
        plt.xlabel('Time')
        
        return None

    def func_plot_mean_and_sem(self, x, y, line_color='b', fill_color='b', sem_option=1, n_std=1):
        
        """
        :param x: 1D numpy array with length m (m features)
        :param y: 2D numpy array with shape (n, m) (n observations, m features)
        :param line_color: line color (default 'b')
        :param fill_color: fill color (default 'b')
        :param sem_option: standard error option (1: sem, 2: std, 3: bootstrapping) (default 1)
        :param n_std: standard deviation multiplier (default 1)
        """
    
        x_line = x
        y_line = np.mean(y, axis=0)
    
        if sem_option == 1:
            y_sem = np.std(y, axis=0) / np.sqrt(y.shape[0])
        elif sem_option == 2:
            y_sem = np.std(y, axis=0)
        elif sem_option == 3:
            y_tmp = np.zeros((1000, y.shape[1]))
            for i in range(1000):
                y_isample = np.random.choice(y.shape[0], size=y.shape[0], replace=True)
                y_tmp[i, :] = np.mean(y[y_isample, :], axis=0)
            y_sem = np.std(y_tmp, axis=0)
        else:
            y_sem = np.std(y, axis=0) / np.sqrt(y.shape[0])

        
        plt.plot(y_line, line_color)

        plt.fill_between(x, y_line - y_sem, 
                            y_line + y_sem,
                            color=[fill_color])


    def get_single_trial_recovery_vector(self, trial):
        
        """
        Find the recovery vector for a single stim trial over all neurons
        
        for each neuron:
            get average activity during control trials delay period
            build population level activity vector for L vs R control trials 
            
        for every stim trial:
            get average recovery trace by subtracing late delay - stim period (early delay)
            
            
        """
        stim_period = [] #Shape will be (nx6)
        poststim_period = []
        for n in self.good_neurons:
            
            # stim_period += [self.dff[0, trial][n, self.delay: self.delay + 6]]
            poststim_period += [self.dff[0, trial][n, self.response-int(1/6*1/self.fs)]]
            
        # Take diff between post stim - stim
        # StimRecovery_mode = np.mean(np.array(poststim_period), axis=1) - np.mean(np.array(stim_period), axis=1)
        # Recover vector should be (nx1)

        # Normalize
        # StimRecovery_mode = StimRecovery_mode / np.linalg.norm(StimRecovery_mode)
        
        return np.array(poststim_period)
    
    
    def get_all_recovery_vectors(self):
        
        """
        Get all the normalized recovery vectors over stim trials
        
        Returns
        ------
            list : shape is trials x neurons
        """
        
        vectors = []
        
        for trial in np.where(self.stim_ON)[0]:
            
            vectors += [self.get_single_trial_recovery_vector(trial)]    
        
        return np.array(vectors)

        
## DECODING ANALYSIS ##
        
    def decision_boundary(self, mode_input='choice', opto=False, error=False, 
                          persistence=False, ctl=False, remove_n = []):
        """
        Calculate decision boundary across trials of CD
        
        
        Use method from Guang's paper
        
        Use orthgonalized CD's and specify which mode to use
        
        persistence : bool, optional
            If True, then the calculation for sample mode deocding is based on 
            the end of the delay instead of end of sample period
            
        remove_n : list, optional
            If not empty, remove these neurons from the calculation of decoding acc
        """
        
        idx_map = {'choice': 1, 'action':5, 'stimulus':0}
        idx = idx_map[mode_input]

        orthonormal_basis, mean = self.plot_behaviorally_relevant_modes(plot=False, ctl=ctl) # one method
        orthonormal_basis = orthonormal_basis[:, idx]
        
        if len(remove_n) != 0:
            keep_n = [i for i in np.arange(len(self.good_neurons)) if i not in remove_n]

            good_neurons = self.good_neurons[keep_n]
            orthonormal_basis = orthonormal_basis[keep_n]
            
            activityRL_train= np.concatenate((self.PSTH_r_train_correct[keep_n], 
                                            self.PSTH_l_train_correct[keep_n]), axis=1)
    
            activityRL_test= np.concatenate((self.PSTH_r_test_correct[keep_n], 
                                            self.PSTH_l_test_correct[keep_n]), axis=1)
        
        
        else:
            good_neurons = self.good_neurons
            

            activityRL_train= np.concatenate((self.PSTH_r_train_correct, 
                                            self.PSTH_l_train_correct), axis=1)
    
            activityRL_test= np.concatenate((self.PSTH_r_test_correct, 
                                            self.PSTH_l_test_correct), axis=1)
            
            
        
        
        r_corr = np.where(self.R_correct)[0]
        l_corr = np.where(self.L_correct)[0]
        
        r_trials = [i for i in r_corr if i in self.i_good_non_stim_trials and not self.early_lick[i]]
        l_trials = [i for i in l_corr if i in self.i_good_non_stim_trials and not self.early_lick[i]]

        
        x = np.arange(-6.97,4,self.fs)[:self.time_cutoff]

        # orthonormal_basis = orthonormal_basis.reshape(-1,1)
        i_pc = 0
        projright, projleft = [], []
        
        time_point_map = {'choice': self.response-int(1/6*1/self.fs), 'action':self.response+int(7/6*1/self.fs), 'stimulus':self.delay-int(1/6*1/self.fs)}
        # DEcode stim using end of delay
        if persistence:
            time_point_map = {'choice': self.response-int(1/6*1/self.fs), 'action':self.response+int(7/6*1/self.fs), 'stimulus':self.response-int(1/6*1/self.fs)}
        time_point = time_point_map[mode_input]
        
        # Project for every trial in train set for DB
        for t in self.r_train_idx:
            activity = self.dff[0, r_trials[t]][good_neurons] 
            activity = activity 
            proj_allDim = np.dot(activity.T, orthonormal_basis)
            projright += [proj_allDim[time_point]]
            
        for t in self.l_train_idx:
            activity = self.dff[0, l_trials[t]][good_neurons]
            activity = activity 
            proj_allDim = np.dot(activity.T, orthonormal_basis)
            projleft += [proj_allDim[time_point]]

        


        db = ((np.mean(projright) / np.var(projright)) + (np.mean(projleft) / np.var(projleft))) / (((1/ np.var(projright))) + (1/ np.var(projleft)))
        sign = np.mean(projright) > db

        decoderchoice = []
        if opto:
            # Project for every trial
        
            r_opto, l_opto = self.get_trace_matrix_multiple(good_neurons, opto=True)
    
            activityRL_opto= np.concatenate((r_opto, l_opto), axis=1)
            
            r_corr = np.where(self.R_correct + self.L_wrong)[0]
            l_corr = np.where(self.L_correct + self.R_wrong)[0]
            # Project for every opto trial
            r_trials = [i for i in r_corr if self.stim_ON[i] and not self.early_lick[i]]
            l_trials = [i for i in l_corr if self.stim_ON[i] and not self.early_lick[i]]
            

            for r in r_trials:
                activity = self.dff[0, r][good_neurons] 
                activity = activity - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activity.shape[1]))
                proj_allDim = np.dot(activity.T, orthonormal_basis)
                if sign:
                    decoderchoice += [proj_allDim[time_point]>db]
                else:
                    decoderchoice += [proj_allDim[time_point]<db]

                # plt.plot(x, proj_allDim[:len(self.T_cue_aligned_sel)], 'b', alpha = 0.5,  linewidth = 0.5)
                
            for l in l_trials:
                activity = self.dff[0, l][good_neurons]
                activity = activity - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activity.shape[1]))
                proj_allDim = np.dot(activity.T, orthonormal_basis)
                if sign:
                    decoderchoice += [proj_allDim[time_point]<db]
                else:
                    decoderchoice += [proj_allDim[time_point]>db]
                # plt.plot(x, proj_allDim[:len(self.T_cue_aligned_sel)], 'r', alpha = 0.5, linewidth = 0.5)
                
                
        else:
            # Project for every trial
            for t in self.r_test_idx:
                activity = self.dff[0, r_trials[t]][good_neurons] 
                activity = activity - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activity.shape[1]))
                proj_allDim = np.dot(activity.T, orthonormal_basis)
    
                if sign:
                    decoderchoice += [proj_allDim[time_point]>db]
                else:
                    decoderchoice += [proj_allDim[time_point]<db]                # plt.plot(x, proj_allDim[:len(self.T_cue_aligned_sel)], 'b', alpha = 0.5,  linewidth = 0.5)
                # plt.scatter(x[time_point],[proj_allDim[time_point]], color='b')
                
            for t in self.l_test_idx:
                activity = self.dff[0, l_trials[t]][good_neurons]
                activity = activity -np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activity.shape[1]))
                proj_allDim = np.dot(activity.T, orthonormal_basis)
    
                if sign:
                    decoderchoice += [proj_allDim[time_point]<db]
                else:
                    decoderchoice += [proj_allDim[time_point]>db]                # plt.plot(x, proj_allDim[:len(self.T_cue_aligned_sel)], 'r', alpha = 0.5, linewidth = 0.5)
                # plt.scatter(x[time_point],[proj_allDim[time_point]], color='r')
            
        
            # Exclude:
            if error:
                # include error trials in the test results as well
                r_test_err = [i for i in self.i_good_non_stim_trials if not self.early_lick[i] and self.L_wrong[i]]
                l_test_err = [i for i in self.i_good_non_stim_trials if not self.early_lick[i] and self.R_wrong[i]]
                
        
                for t in r_test_err:
                    activity = self.dff[0, t][good_neurons] 
                    activity = activity - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activity.shape[1]))
                    proj_allDim = np.dot(activity.T, orthonormal_basis)
                    if sign:
                        decoderchoice += [proj_allDim[time_point]>db]
                    else:
                        decoderchoice += [proj_allDim[time_point]<db]      
                        
                for t in l_test_err:
                    activity = self.dff[0, t][good_neurons] 
                    activity = activity - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activity.shape[1]))
                    proj_allDim = np.dot(activity.T, orthonormal_basis)
                    if sign:
                        decoderchoice += [proj_allDim[time_point]<db]
                    else:
                        decoderchoice += [proj_allDim[time_point]>db]        
        
        return orthonormal_basis, np.mean(activityRL_train, axis=1)[:, None], db, decoderchoice
    
    def plot_performance_distfromCD(self, opto=False):
        """
        Replicates Li et al., 2016 Fig 4b, where performance is calculated across
        trial types as a function of distance from CD trajectores in the last 
        timestep before Go cue (last 400ms)
        
        Parameters
        ----------
        opto : bool, optional
            Whether to plot opto trials
            
        Returns
        -------
        None.

        """
        save=None
        orthonormal_basis, var_allDim = self.func_compute_epoch_decoder([self.PSTH_r_train_correct, 
                                                                        self.PSTH_l_train_correct], range(self.delay+9, self.response))
        activityRL_train= np.concatenate((self.PSTH_r_train_correct, 
                                        self.PSTH_l_train_correct), axis=1)

        activityRL_test= np.concatenate((self.PSTH_r_test_correct, 
                                        self.PSTH_l_test_correct), axis=1)
        
        r_corr = np.where(self.R_correct)[0]
        l_corr = np.where(self.L_correct)[0]
        
        r_trials = [i for i in r_corr if i in self.i_good_non_stim_trials and not self.early_lick[i]]
        l_trials = [i for i in l_corr if i in self.i_good_non_stim_trials and not self.early_lick[i]]

        
        x = np.arange(-6.97,4,self.fs)[:self.time_cutoff]

        # orthonormal_basis = orthonormal_basis.reshape(-1,1)
        i_pc = 0
        projright, projleft = [], []
        
        projright, projleft = [], []
        # Project for every trial in train set for DB
        for t in self.r_train_idx:
            activity = self.dff[0, r_trials[t]][self.good_neurons] 
            activity = activity 
            proj_allDim = np.dot(activity.T, orthonormal_basis)
            projright += [proj_allDim[self.response-int(1/6*1/self.fs), i_pc]]
            
        for t in self.l_train_idx:
            activity = self.dff[0, l_trials[t]][self.good_neurons]
            activity = activity 
            proj_allDim = np.dot(activity.T, orthonormal_basis)
            projleft += [proj_allDim[self.response-int(1/6*1/self.fs), i_pc]]



        db = ((np.mean(projright) / np.var(projright)) + (np.mean(projleft) / np.var(projleft))) / (((1/ np.var(projright))) + (1/ np.var(projleft)))

        decoderchoice = []

        # Project for every trial
        for t in self.r_test_idx:
            activity = self.dff[0, r_trials[t]][self.good_neurons] 
            activity = activity - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activity.shape[1]))
            proj_allDim = np.dot(activity.T, orthonormal_basis)

            decoderchoice += [proj_allDim[self.response-int(1/6*1/self.fs), i_pc]<db]
            plt.plot(x, proj_allDim[:len(self.T_cue_aligned_sel), i_pc], 'b', alpha = 0.5,  linewidth = 0.5)
            # plt.scatter(x[self.response-1],[proj_allDim[self.response-1, i_pc]], color='b')
            
        for t in self.l_test_idx:
            activity = self.dff[0, l_trials[t]][self.good_neurons]
            activity = activity - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activity.shape[1]))
            proj_allDim = np.dot(activity.T, orthonormal_basis)

            decoderchoice += [proj_allDim[self.response-int(1/6*1/self.fs), i_pc]>db]
            plt.plot(x, proj_allDim[:len(self.T_cue_aligned_sel), i_pc], 'r', alpha = 0.5, linewidth = 0.5)
            # plt.scatter(x[self.response-1],[proj_allDim[self.response-1, i_pc]], color='r')
            
        # Correct trials
        activityRL_test = activityRL_test - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activityRL_test.shape[1]))  # remove mean
        proj_allDim = np.dot(activityRL_test.T, orthonormal_basis)

        
        # ax = axs.flatten()[0]
        plt.plot(x, proj_allDim[:len(self.T_cue_aligned_sel), i_pc], 'b', linewidth = 2)
        plt.plot(x, proj_allDim[len(self.T_cue_aligned_sel):, i_pc], 'r', linewidth = 2)
        plt.title("Choice decoder projections")
        plt.axvline(self.sample, color = 'grey', alpha=0.5, ls = '--')
        plt.axvline(self.delay, color = 'grey', alpha=0.5, ls = '--')
        plt.axvline(0, color = 'grey', alpha=0.5, ls = '--')
        plt.ylabel('CD_delay projection (a.u.)')
        plt.scatter(x[self.response-1], [db])

        if save is not None:
            plt.savefig(save)
            

        plt.show()
        
        return decoderchoice
        
      
## Modes with optogenetic inhibition
    
    def plot_CD_opto(self, stim_side, mode_input = 'choice', save=None, return_traces = False,
                     return_applied = False, normalize=False, ctl=False, lickdir=False):
        '''
        Plots similar figure as Li et al 2016 Fig 3c to view the effect of
        photoinhibition on L/R CD traces
        
        Returns
        -------
        R then L for control, opto traces as well as error bars
        
        '''
                
        if ctl:
            orthonormal_basis, var_allDim = self.func_compute_activity_modes_DRT([self.PSTH_r_train_correct, 
                                                                                self.PSTH_l_train_correct], ctl=ctl, 
                                                                                lickdir=lickdir)
            activityRL_train= np.concatenate((self.PSTH_r_train_correct, 
                                            self.PSTH_l_train_correct), axis=1)
            
        else:
            orthonormal_basis, var_allDim = self.func_compute_activity_modes_DRT([self.PSTH_r_train_correct, 
                                                                                self.PSTH_l_train_correct, 
                                                                                self.PSTH_r_train_error, 
                                                                                self.PSTH_l_train_error], ctl=ctl, 
                                                                                lickdir=lickdir)           
            
            activityRL_train= np.concatenate((self.PSTH_r_train_correct, 
                                            self.PSTH_l_train_correct, 
                                            self.PSTH_r_train_error, 
                                            self.PSTH_l_train_error), axis=1)
        
        activityRL_test= np.concatenate((self.PSTH_r_test_correct, 
                                        self.PSTH_l_test_correct), axis=1)
        
        idx_map = {'choice': 1, 'action':5, 'stimulus':0, 'ramping':7}
        idx = idx_map[mode_input]

        orthonormal_basis = orthonormal_basis[:, idx]
        
    

        activityRL_test= np.concatenate((self.PSTH_r_test_correct, 
                                        self.PSTH_l_test_correct), axis=1)
        
       
        x = self.t
      
        if normalize:
            # Get mean and STD
            
            proj_allDim = np.dot(activityRL_train.T, orthonormal_basis)
            meantrain, meanstd = np.mean(proj_allDim), np.std(proj_allDim)

        # Correct trials
        activityRL_test = activityRL_test - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activityRL_test.shape[1]))  # remove mean
        proj_allDim = np.dot(activityRL_test.T, orthonormal_basis)
        if normalize:

            proj_allDim = (proj_allDim - meantrain) / meanstd
        
        control_traces = proj_allDim[:len(self.T_cue_aligned_sel)], proj_allDim[len(self.T_cue_aligned_sel):]
        if not return_traces:
            # Plot average control traces as dotted lines
            plt.plot(x, proj_allDim[:len(self.T_cue_aligned_sel)], 'b', ls = '--', linewidth = 0.5)
            plt.plot(x, proj_allDim[len(self.T_cue_aligned_sel):], 'r', ls = '--', linewidth = 0.5)
            plt.title("Choice decoder projections with opto")
            plt.axvline(self.sample, color = 'grey', alpha=0.5, ls = '--')
            plt.axvline(self.delay, color = 'grey', alpha=0.5, ls = '--')
            plt.axvline(self.response, color = 'grey', alpha=0.5, ls = '--')
            plt.ylabel('CD_delay projection (a.u.)')
            
        
        # PLOT OPTO # 
        
        # r_opto, l_opto = self.get_trace_matrix_multiple(self.good_neurons, opto=True)

        # activityRL_opto= np.concatenate((r_opto, l_opto), axis=1)
  
        if stim_side == 'R':
            r_trials, l_trials = self.r_opto_stim_right_idx, self.l_opto_stim_right_idx
        elif stim_side == 'L':
            r_trials, l_trials = self.r_opto_stim_left_idx, self.l_opto_stim_left_idx

        r_proj = []
        l_proj = []
        
        for r in r_trials:

            activity, _, _ = self.get_PSTH_multiple(self.good_neurons, [r], 
                                                      binsize=self.binsize, 
                                                      timestep=self.timestep)

            activity = activity - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activity.shape[1]))
            proj_allDim = np.dot(activity.T, orthonormal_basis)
            r_proj += [proj_allDim[:len(self.T_cue_aligned_sel)]]

            
        for l in l_trials:
            activity, _, _ = self.get_PSTH_multiple(self.good_neurons, [l], 
                                                      binsize=self.binsize, 
                                                      timestep=self.timestep)
            activity = activity - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activity.shape[1]))
            proj_allDim = np.dot(activity.T, orthonormal_basis)
            l_proj += [proj_allDim[:len(self.T_cue_aligned_sel)]]

            
            
        # Opto trials
        # activityRL_opto = activityRL_opto - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activityRL_opto.shape[1]))  # remove mean
        # proj_allDim = np.dot(activityRL_opto.T, orthonormal_basis)
        
        # Instead, just average left and right traces
        l_proj_mean = np.mean(l_proj, axis=0)
        r_proj_mean = np.mean(r_proj, axis=0)
        
        if normalize:

            proj_allDim = (proj_allDim - meantrain) / meanstd
            l_proj = (l_proj - meantrain) / meanstd
            r_proj = (r_proj - meantrain) / meanstd
            
        if return_traces:
            opto_traces = proj_allDim[:len(self.T_cue_aligned_sel)], proj_allDim[len(self.T_cue_aligned_sel):]
            error_bars = stats.sem(r_proj, axis=0), stats.sem(l_proj, axis=0)
            if return_applied:
                return control_traces, opto_traces, error_bars, orthonormal_basis, np.mean(activityRL_train, axis=1)[:, None], meantrain, meanstd
            else:

                return control_traces, opto_traces, error_bars
        
        plt.plot(x, r_proj_mean, 'b', linewidth = 2)
        plt.plot(x, l_proj_mean, 'r', linewidth = 2)
        
        plt.fill_between(x, l_proj_mean - stats.sem(l_proj, axis=0), 
                 l_proj_mean +  stats.sem(l_proj, axis=0),
                 color=['#ffaeb1'])
        plt.fill_between(x, r_proj_mean - stats.sem(r_proj, axis=0), 
                 r_proj_mean + stats.sem(r_proj, axis=0),
                 color=['#b4b2dc'])
        
        plt.hlines(y=max(cat((l_proj_mean, r_proj_mean))) + 0.5, xmin=self.delay, xmax=self.delay+1, linewidth=10, color='lightblue')

        if save is not None:
            plt.savefig(save)
        plt.show()

        if return_applied:
            return orthonormal_basis, np.mean(activityRL_train, axis=1)[:, None], meantrain, meanstd

        # axs[0, 0].set_ylabel('Activity proj.')
        # axs[3, 0].set_xlabel('Time')

        
    def plot_persistent_mode_opto(self, epoch=None, save=None):
        '''
        Plots similar figure as Li et al 2016 Fig 3c to view the effect of
        photoinhibition on L/R CD traces
        '''
        if epoch is not None:
            orthonormal_basis, var_allDim = self.func_compute_persistent_decoder([self.PSTH_r_train_correct, 
                                                                            self.PSTH_l_train_correct,
                                                                            self.PSTH_r_train_opto,
                                                                            self.PSTH_l_train_opto], epoch)
        else:
            
            orthonormal_basis, var_allDim = self.func_compute_persistent_decoder([self.PSTH_r_train_correct, 
                                                                            self.PSTH_l_train_correct,
                                                                            self.PSTH_r_train_opto,
                                                                            self.PSTH_l_train_opto], range(self.response, self.response+3))
            
        activityRL_train= np.concatenate([self.PSTH_r_train_correct, 
                                        self.PSTH_l_train_correct,
                                        self.PSTH_r_train_opto,
                                        self.PSTH_l_train_opto], axis=1)

        activityRL_test= np.concatenate((self.PSTH_r_test_correct, 
                                        self.PSTH_l_test_correct), axis=1)
        

        x = np.arange(-6.97,4,self.fs)[:self.time_cutoff]

        # orthonormal_basis = orthonormal_basis.reshape(-1,1)
        i_pc = 0

        # Project for every control trial
        # for t in self.r_test_idx:
        #     activity = self.dff[0, r_trials[t]][self.good_neurons] 
        #     activity = activity - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activity.shape[1]))
        #     proj_allDim = np.dot(activity.T, orthonormal_basis)
        #     # plt.plot(x, proj_allDim[:len(self.T_cue_aligned_sel), i_pc], 'b', alpha = 0.5,  linewidth = 0.5)
            
        # for t in self.l_test_idx:
        #     activity = self.dff[0, l_trials[t]][self.good_neurons]
        #     activity = activity - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activity.shape[1]))
        #     proj_allDim = np.dot(activity.T, orthonormal_basis)
        #     # plt.plot(x, proj_allDim[:len(self.T_cue_aligned_sel), i_pc], 'r', alpha = 0.5, linewidth = 0.5)
            
            
        # Correct trials
        activityRL_test = activityRL_test - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activityRL_test.shape[1]))  # remove mean
        proj_allDim = np.dot(activityRL_test.T, orthonormal_basis)

        
        # Plot average control traces as dotted lines
        plt.plot(x, proj_allDim[:len(self.T_cue_aligned_sel), i_pc], 'b', ls = '--', linewidth = 0.5)
        plt.plot(x, proj_allDim[len(self.T_cue_aligned_sel):, i_pc], 'r', ls = '--', linewidth = 0.5)
        plt.title("Choice decoder projections with opto")
        plt.axvline(self.sample, color = 'grey', alpha=0.5, ls = '--')
        plt.axvline(self.delay, color = 'grey', alpha=0.5, ls = '--')
        plt.axvline(0, color = 'grey', alpha=0.5, ls = '--')
        plt.ylabel('CD_delay projection (a.u.)')
        
        
        
        
        r_corr = np.where(self.R_correct + self.R_wrong)[0]
        l_corr = np.where(self.L_correct + self.L_wrong)[0]
        
        r_trials = [i for i in r_corr if self.stim_ON[i] and not self.early_lick[i]]
        l_trials = [i for i in l_corr if self.stim_ON[i] and not self.early_lick[i]]
        
        # r_opto, l_opto = self.get_trace_matrix_multiple(self.good_neurons, opto=True)

        activityRL_opto= np.concatenate((self.PSTH_r_test_opto, self.PSTH_l_test_opto), axis=1)
        
        # Project for every opto trial

        
        r_proj = []
        l_proj = []
        for t in self.r_test_opto_idx:
            activity = self.dff[0, r_trials[t]][self.good_neurons] 
            activity = activity - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activity.shape[1]))
            proj_allDim = np.dot(activity.T, orthonormal_basis)
            r_proj += [proj_allDim[:len(self.T_cue_aligned_sel), i_pc]]
            # plt.plot(x, proj_allDim[:len(self.T_cue_aligned_sel), i_pc], 'b', alpha = 0.5,  linewidth = 0.5)
            
        for t in self.l_test_opto_idx:
            activity = self.dff[0, l_trials[t]][self.good_neurons] 
            activity = activity - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activity.shape[1]))
            proj_allDim = np.dot(activity.T, orthonormal_basis)
            l_proj += [proj_allDim[:len(self.T_cue_aligned_sel), i_pc]]
            # plt.plot(x, proj_allDim[:len(self.T_cue_aligned_sel), i_pc], 'r', alpha = 0.5, linewidth = 0.5)
            
            
        # Opto trials
        activityRL_opto = activityRL_opto - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activityRL_opto.shape[1]))  # remove mean
        proj_allDim = np.dot(activityRL_opto.T, orthonormal_basis)
        
        plt.plot(x, proj_allDim[:len(self.T_cue_aligned_sel), i_pc], 'b', linewidth = 2)
        plt.plot(x, proj_allDim[len(self.T_cue_aligned_sel):, i_pc], 'r', linewidth = 2)
        
        plt.fill_between(x, proj_allDim[len(self.T_cue_aligned_sel):, i_pc] - stats.sem(l_proj, axis=0), 
                 proj_allDim[len(self.T_cue_aligned_sel):, i_pc] +  stats.sem(l_proj, axis=0),
                 color=['#ffaeb1'])
        plt.fill_between(x, proj_allDim[:len(self.T_cue_aligned_sel), i_pc] - stats.sem(r_proj, axis=0), 
                 proj_allDim[:len(self.T_cue_aligned_sel), i_pc] + stats.sem(r_proj, axis=0),
                 color=['#b4b2dc'])
        
        plt.hlines(y=max(proj_allDim[:, i_pc]) + 0.5, xmin=self.delay, xmax=-2, linewidth=10, color='red')

        if save is not None:
            plt.savefig(save)
            
        plt.show()
        


    def plot_sorted_CD_opto(self, epoch=None, save=None, return_traces = False, normalize=True):
        '''
        Plots similar figure as Chen et al Fig 5FG to view the effect of selectivity on
        photoinhibition on L/R CD traces
        
        Returns
        -------
        R then L for control, opto traces as well as error bars
        
        '''
        if epoch is not None:
            orthonormal_basis, var_allDim = self.func_compute_epoch_decoder([self.PSTH_r_train_correct, 
                                                                            self.PSTH_l_train_correct], epoch)
        else:
            
            orthonormal_basis, var_allDim = self.func_compute_epoch_decoder([self.PSTH_r_train_correct, 
                                                                            self.PSTH_l_train_correct], range(self.delay+12, self.response))
        activityRL_train= np.concatenate((self.PSTH_r_train_correct, 
                                        self.PSTH_l_train_correct), axis=1)

        activityRL_test= np.concatenate((self.PSTH_r_test_correct, 
                                        self.PSTH_l_test_correct), axis=1)
        

       
        x = np.arange(-6.97,4,self.fs)[:self.time_cutoff]

        # orthonormal_basis = orthonormal_basis.reshape(-1,1)
        i_pc = 0
        
        r_corr = np.where(self.R_correct)[0]
        l_corr = np.where(self.L_correct)[0]
        
        r_trials = [i for i in r_corr if i in self.i_good_non_stim_trials and not self.early_lick[i]]
        l_trials = [i for i in l_corr if i in self.i_good_non_stim_trials and not self.early_lick[i]]
        
        projright, projleft = [],[]

        # Project for every control trial, before stim
        for t in self.r_test_idx:
            activity = self.dff[0, r_trials[t]][self.good_neurons] 
            activity = activity - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activity.shape[1]))
            proj_allDim = np.dot(activity.T, orthonormal_basis)
            projright += [proj_allDim[self.delay-int(1/2*1/self.fs):self.delay, i_pc]]

            # plt.plot(x, proj_allDim[:len(self.T_cue_aligned_sel), i_pc], 'b', alpha = 0.5,  linewidth = 0.5)
            
        for t in self.l_test_idx:
            activity = self.dff[0, l_trials[t]][self.good_neurons]
            activity = activity - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activity.shape[1]))
            proj_allDim = np.dot(activity.T, orthonormal_basis)
            projleft += [proj_allDim[self.delay-int(1/2*1/self.fs):self.delay, i_pc]]

            # plt.plot(x, proj_allDim[:len(self.T_cue_aligned_sel), i_pc], 'r', alpha = 0.5, linewidth = 0.5)
            
        r_median = np.median(projright)
        l_median = np.median(projleft)
        print(r_median, l_median)
        db = ((np.mean(projright) / np.var(projright)) + (np.mean(projleft) / np.var(projleft))) / (((1/ np.var(projright))) + (1/ np.var(projleft)))

        # if normalize:
        #     # Get mean and STD
            
        #     proj_allDim = np.dot(activityRL_train.T, orthonormal_basis)
        #     meantrain, meanstd = np.mean(proj_allDim), np.std(proj_allDim)

        # Correct trials: re-project based on low or high selectivity
        
        # Project for every control trial, before stim
        r_proj_allDim_strong = []
        r_proj_allDim_weak = []
        for t in self.r_test_idx:
            activity = self.dff[0, r_trials[t]][self.good_neurons] 
            activity = activity - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activity.shape[1]))
            proj_allDim = np.dot(activity.T, orthonormal_basis)
            
            proj_point = np.mean(proj_allDim[self.delay-int(1/2*1/self.fs):self.delay, i_pc])
            if proj_point < r_median and proj_point < db:
                r_proj_allDim_weak += [[proj_allDim[:self.time_cutoff, i_pc]]]
            elif proj_point > r_median and proj_point < db:
                r_proj_allDim_strong += [[proj_allDim[:self.time_cutoff, i_pc]]]


            # plt.plot(x, proj_allDim[:len(self.T_cue_aligned_sel), i_pc], 'b', alpha = 0.5,  linewidth = 0.5)
        
        l_proj_allDim_strong = []
        l_proj_allDim_weak = []
        for t in self.l_test_idx:
            activity = self.dff[0, l_trials[t]][self.good_neurons]
            activity = activity - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activity.shape[1]))
            proj_allDim = np.dot(activity.T, orthonormal_basis)
            
            proj_point = np.mean(proj_allDim[self.delay-int(1/2*1/self.fs):self.delay, i_pc])
            # print(proj_allDim.shape)
            if proj_point < l_median and proj_point > db:
                l_proj_allDim_weak += [proj_allDim[:self.time_cutoff, i_pc]]
            elif proj_point > l_median and proj_point > db:
                l_proj_allDim_strong += [proj_allDim[:self.time_cutoff, i_pc]]
            else:
                print('Error')
                print(proj_point, l_median, db)
        # if normalize:

        #     proj_allDim = (proj_allDim - meantrain) / meanstd
        
        # control_traces = proj_allDim[:len(self.T_cue_aligned_sel), i_pc], proj_allDim[len(self.T_cue_aligned_sel):, i_pc]
        
        if not return_traces:
            f, ax = plt.subplots(1,2, sharey='row')
            # Plot average control traces as dotted lines
            ax[0].plot(x, np.mean(r_proj_allDim_weak, axis=0)[0], 'b', ls = '--', linewidth = 1)
            ax[0].plot(x, np.mean(l_proj_allDim_weak, axis=0), 'r', ls = '--', linewidth = 1)
            ax[0].set_title("Weak selectivity")
            ax[0].axvline(self.sample, color = 'grey', alpha=0.5, ls = '--')
            ax[0].axvline(self.delay, color = 'grey', alpha=0.5, ls = '--')
            ax[0].axvline(0, color = 'grey', alpha=0.5, ls = '--')
            ax[0].set_ylabel('CD_delay projection (a.u.)')

            ax[1].plot(x, np.mean(r_proj_allDim_strong, axis=0)[0], 'b', ls = '--', linewidth = 1)
            ax[1].plot(x, np.mean(l_proj_allDim_strong, axis=0), 'r', ls = '--', linewidth = 1)
            ax[1].set_title("Strong selectivity")
            ax[1].axvline(self.sample, color = 'grey', alpha=0.5, ls = '--')
            ax[1].axvline(self.delay, color = 'grey', alpha=0.5, ls = '--')
            ax[1].axvline(0, color = 'grey', alpha=0.5, ls = '--')
            ax[1].set_ylabel('CD_delay projection (a.u.)')     
            
            
            
            
            # ax[0].fill_between(x, np.mean(l_proj_allDim_weak, axis=0)[0] - stats.sem(l_proj_allDim_weak, axis=0), 
            #          np.mean(l_proj_allDim_weak, axis=0)[0] +  stats.sem(l_proj_allDim_weak, axis=0),
            #          color=['#ffaeb1'])
            # ax[0].fill_between(x, np.mean(r_proj_allDim_weak, axis=0)[0] - stats.sem(r_proj_allDim_weak, axis=0), 
            #          np.mean(r_proj_allDim_weak, axis=0)[0] + stats.sem(r_proj_allDim_weak, axis=0),
            #          color=['#b4b2dc'])
            
            
            # ax[1].fill_between(x, np.mean(r_proj_allDim_strong, axis=0)[0] - stats.sem(r_proj_allDim_strong, axis=0), 
            #          np.mean(r_proj_allDim_strong, axis=0)[0] +  stats.sem(r_proj_allDim_strong, axis=0),
            #          color=['#b4b2dc'])
            # ax[1].fill_between(x,  np.mean(l_proj_allDim_strong, axis=0) - stats.sem(l_proj_allDim_strong, axis=0), 
            #         np.mean(l_proj_allDim_strong, axis=0) + stats.sem(l_proj_allDim_strong, axis=0),
            #          color=['#ffaeb1'])
            # plt.show()
            # return l_proj_allDim_strong
        
        
        r_opto, l_opto = self.get_trace_matrix_multiple(self.good_neurons, opto=True)

        activityRL_opto= np.concatenate((r_opto, l_opto), axis=1)
        
        r_corr = np.where(self.R_correct + self.L_wrong)[0]
        l_corr = np.where(self.L_correct + self.R_wrong)[0]
        # Project for every opto trial
        r_trials = [i for i in r_corr if self.stim_ON[i] and not self.early_lick[i]]
        l_trials = [i for i in l_corr if self.stim_ON[i] and not self.early_lick[i]]
        print(len(r_trials), len(l_trials))
        shuffle(r_trials), shuffle(l_trials)
        
        r_proj_allDim_strong = []
        r_proj_allDim_weak = []
        for r in r_trials:
            activity = self.dff[0, r][self.good_neurons] 
            activity = activity - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activity.shape[1]))
            proj_allDim = np.dot(activity.T, orthonormal_basis)

            # plt.plot(x, proj_allDim[:len(self.T_cue_aligned_sel), i_pc], 'b', alpha = 0.5,  linewidth = 0.5)
            proj_point = np.mean(proj_allDim[self.delay-int(1/2*1/self.fs):self.delay, i_pc])
            if proj_point < r_median and proj_point < db:
                r_proj_allDim_weak += [[proj_allDim[:self.time_cutoff, i_pc]]]
                if len(r_proj_allDim_weak) > 3:
                    continue
                ax[0].plot(x, proj_allDim[:self.time_cutoff, i_pc], 'b', linewidth = 1)

            elif proj_point > r_median and proj_point < db:
                r_proj_allDim_strong += [[proj_allDim[:self.time_cutoff, i_pc]]]
                if len(r_proj_allDim_strong) > 3:
                    continue
                ax[1].plot(x, proj_allDim[:self.time_cutoff, i_pc], 'b', linewidth = 1)

                
        l_proj_allDim_strong = []
        l_proj_allDim_weak = []
        for l in l_trials:
            activity = self.dff[0, l][self.good_neurons]
            activity = activity - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activity.shape[1]))
            proj_allDim = np.dot(activity.T, orthonormal_basis)

            # plt.plot(x, proj_allDim[:len(self.T_cue_aligned_sel), i_pc], 'r', alpha = 0.5, linewidth = 0.5)
            proj_point = np.mean(proj_allDim[self.delay-int(1/2*1/self.fs):self.delay, i_pc])
            # print(proj_allDim.shape)
            if proj_point < l_median and proj_point > db:
                l_proj_allDim_weak += [proj_allDim[:self.time_cutoff, i_pc]]
                if len(l_proj_allDim_weak) > 3:
                    continue
                ax[0].plot(x, proj_allDim[:self.time_cutoff, i_pc], 'r', linewidth = 1)

            elif proj_point > l_median and proj_point > db:
                l_proj_allDim_strong += [proj_allDim[:self.time_cutoff, i_pc]]
                if len(l_proj_allDim_strong) > 3:
                    continue
                ax[1].plot(x, proj_allDim[:self.time_cutoff, i_pc], 'r', linewidth = 1)

            else:
                print('Error')
                print(proj_point, l_median, db)
                
                
        
        # if not return_traces:
        #     ax[0].plot(x, np.mean(r_proj_allDim_weak, axis=0)[0], 'b', linewidth = 1)
        #     ax[0].plot(x, np.mean(l_proj_allDim_weak, axis=0), 'r', linewidth = 1)


        #     ax[1].plot(x, np.mean(r_proj_allDim_strong, axis=0)[0], 'b', linewidth = 1)
        #     ax[1].plot(x, np.mean(l_proj_allDim_strong, axis=0), 'r', linewidth = 1)
   





        # Opto trials
        # activityRL_opto = activityRL_opto - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activityRL_opto.shape[1]))  # remove mean
        # proj_allDim = np.dot(activityRL_opto.T, orthonormal_basis)
        # if normalize:

        #     proj_allDim = (proj_allDim - meantrain) / meanstd
            
        # if return_traces:
            
        #     opto_traces =  proj_allDim[:len(self.T_cue_aligned_sel), i_pc], proj_allDim[len(self.T_cue_aligned_sel):, i_pc]
        #     error_bars = stats.sem(r_proj, axis=0), stats.sem(l_proj, axis=0)
        #     return control_traces, opto_traces, error_bars
        
        # plt.plot(x, proj_allDim[:len(self.T_cue_aligned_sel), i_pc], 'b', linewidth = 2)
        # plt.plot(x, proj_allDim[len(self.T_cue_aligned_sel):, i_pc], 'r', linewidth = 2)
        
        # plt.fill_between(x, proj_allDim[len(self.T_cue_aligned_sel):, i_pc] - stats.sem(l_proj, axis=0), 
        #          proj_allDim[len(self.T_cue_aligned_sel):, i_pc] +  stats.sem(l_proj, axis=0),
        #          color=['#ffaeb1'])
        # plt.fill_between(x, proj_allDim[:len(self.T_cue_aligned_sel), i_pc] - stats.sem(r_proj, axis=0), 
        #          proj_allDim[:len(self.T_cue_aligned_sel), i_pc] + stats.sem(r_proj, axis=0),
        #          color=['#b4b2dc'])
        
        ax[0].hlines(y=max(proj_allDim[:, i_pc]) + 0.5, xmin=self.delay, xmax=-2, linewidth=10, color='red')
        ax[1].hlines(y=max(proj_allDim[:, i_pc]) + 0.5, xmin=self.delay, xmax=-2, linewidth=10, color='red')

        if save is not None:
            plt.savefig(save)
            
        plt.show()

### ACROSS SESSION CODING (removed) ###
 
### STIM RELAETD VECTORS

    def get_stim_responsive_neurons(self):
        
        """
        Drops neurons that aren't responsive to opto stim in order to calculate
        the input vector
        
        Returns all neurons are responsive to stim
        
        """
        
        responsive_neurons, indices = [], []
        stimepoch = range(self.delay, self.delay+int(1*1/self.fs))
        alltrain = np.where(~self.stim_ON)[0]
        alloptotrain = np.where(self.stim_ON)[0]
        counter = 0
        for n in self.good_neurons:
            control_activity = np.mean([trial[n, stimepoch] for trial in self.dff[0, alltrain]], axis=0)
            opto_activity = np.mean([trial[n, stimepoch] for trial in self.dff[0, alloptotrain]], axis=0)
            
            res = stats.ttest_ind(control_activity, opto_activity)
            if res.pvalue < 0.01:
                responsive_neurons += [n]
                indices += [counter]
            counter += 1
        self.responsive_neurons = responsive_neurons
        
        return indices
    
    def input_vector(self, orthog=True, plot=False, return_opto=False, remove_unresponsive=True):
        """
        Get the input vector by subtracting the optogenetic stimulation CD projection
        from the control trial projection

        Returns
        -------
        orthonormal basis of the input vector

        """
        if remove_unresponsive:
            good_neurons = self.responsive_neurons
        else:
            good_neurons = self.good_neurons
        
        # Trial type
        r_corr = cat((np.where(self.R_correct)[0], np.where(self.R_wrong)[0]))
        l_corr = cat((np.where(self.L_correct)[0], np.where(self.L_wrong)[0]))
        
        # Lick dir
        if self.lickdir:
            r_corr = cat((np.where(self.R_correct)[0], np.where(self.L_wrong)[0]))
            l_corr = cat((np.where(self.L_correct)[0], np.where(self.R_wrong)[0]))
            
        r_trials_opto = [i for i in r_corr if i in self.i_good_stim_trials and not self.early_lick[i]]
        l_trials_opto = [i for i in l_corr if i in self.i_good_stim_trials and not self.early_lick[i]]

        r_corr = np.where(self.R_correct)[0]
        l_corr = np.where(self.L_correct)[0]
        
        r_trials = [i for i in r_corr if i in self.i_good_non_stim_trials and not self.early_lick[i]]
        l_trials = [i for i in l_corr if i in self.i_good_non_stim_trials and not self.early_lick[i]]
        
        
        alloptotrain = cat(([r_trials_opto[t] for t in self.r_train_opto_idx], [l_trials_opto[t] for t in self.l_train_opto_idx])) 
        alloptotest = cat(([r_trials_opto[t] for t in self.r_test_opto_idx], [l_trials_opto[t] for t in self.l_test_opto_idx])) 

        alltrain = cat(([r_trials[t] for t in self.r_train_idx], [l_trials[t] for t in self.l_train_idx])) 
        alltest = cat(([r_trials[t] for t in self.r_test_idx], [l_trials[t] for t in self.l_test_idx])) 

        
        CD_all = []
        for t in range(self.delay+3, self.delay+6):

            
            control_activity = np.mean([trial[good_neurons, t] for trial in self.dff[0, alltrain]], axis=0)
            opto_activity = np.mean([trial[good_neurons, t] for trial in self.dff[0, alloptotrain]], axis=0)
            # print(control_activity.shape)
            CD_all += [(control_activity - opto_activity) / 2]
            
            
        CD_input_mode = np.mean(CD_all, axis=0)        
        
        
        if orthog: # Process to orthogonalize
        
            allcontroloptotrain = cat((alloptotrain, alltrain))
            activityRL = []
            for n in good_neurons:
                
                activityRL += [np.mean([trial[n, range(self.delay+3, self.delay+6)] for trial in self.dff[0, allcontroloptotrain]], axis=0)]
                
            activityRL = activityRL - np.mean(activityRL, axis=1, keepdims=True) # remove?
            
            u, s, v = np.linalg.svd(activityRL.T)
            proj_allDim = activityRL.T @ v
        
        
        
            # Relevant choice dims
            # CD_choice_mode = [] # Late delay period
            
            CD_input_mode = CD_input_mode / np.linalg.norm(CD_input_mode)
            # return CD_choice_mode, 0
            # Reshape 
            
            CD_input_mode = np.reshape(CD_input_mode, (-1, 1)) 
        
    
            input_ = np.concatenate((CD_input_mode, v), axis=1)
            # orthonormal_basis = self.Gram_Schmidt_process(input_)
            orthonormal_basis, _ = np.linalg.qr(input_, mode='complete')  # lmao
            orthonormal_basis = orthonormal_basis[0]
            
            if return_opto: # Return the projection of every opto trial individually
            
                activity_optotest = []
                
                for t in alloptotest:
                    activity = self.dff[0, t][good_neurons] 
                    activity = activity - np.tile(np.mean(activityRL, axis=1)[:, None], (1, activity.shape[1]))
                    proj_allDim = np.dot(activity.T, orthonormal_basis)
                    activity_optotest += [proj_allDim[range(self.delay, self.delay+6)]]
                
                return np.array(activity_optotest)
                
            
            if plot:
                
                activityRL_testR = []
                for n in good_neurons:
                    
                    activityRL_testR += [np.mean([trial[n, :self.time_cutoff] for trial in self.dff[0, [r_trials[t] for t in self.r_test_idx]]], axis=0)]
                    
                activityRL_testL = []
                for n in good_neurons:
                    
                    activityRL_testL += [np.mean([trial[n, :self.time_cutoff] for trial in self.dff[0, [l_trials[t] for t in self.l_test_idx]]], axis=0)]
                    
                x = np.arange(-6.97,4,self.fs)[:self.time_cutoff]

                # orthonormal_basis = orthonormal_basis.reshape(-1,1)

                # Project for every trial
                for t in self.r_test_idx:
                    activity = self.dff[0, r_trials[t]][good_neurons] 
                    activity = activity - np.tile(np.mean(activityRL, axis=1)[:, None], (1, activity.shape[1]))
                    proj_allDim = np.dot(activity.T, orthonormal_basis)
                    plt.plot(x, proj_allDim[:len(self.T_cue_aligned_sel)], 'b', alpha = 0.1,  linewidth = 0.5)
                    
                for t in self.l_test_idx:
                    activity = self.dff[0, l_trials[t]][good_neurons]
                    activity = activity - np.tile(np.mean(activityRL, axis=1)[:, None], (1, activity.shape[1]))
                    proj_allDim = np.dot(activity.T, orthonormal_basis)
                    plt.plot(x, proj_allDim[:len(self.T_cue_aligned_sel)], 'r', alpha = 0.1, linewidth = 0.5)
                    
                # opto test trials
                activityRL_test = cat((activityRL_testR, activityRL_testL), axis=1)
                activityRL_test = activityRL_test - np.tile(np.mean(activityRL, axis=1)[:, None], (1, activityRL_test.shape[1]))  # remove mean
                proj_allDim = np.dot(activityRL_test.T, orthonormal_basis)
                # print(activityRL_test.shape)

                # print(proj_allDim.shape)
                
                plt.plot(x, proj_allDim[:len(self.T_cue_aligned_sel)], 'b', linewidth = 2)
                plt.plot(x, proj_allDim[len(self.T_cue_aligned_sel):], 'r', linewidth = 2)
                plt.title('Recovery decoder projections')
                plt.axvline(self.sample, color = 'grey', alpha=0.5, ls = '--')
                plt.axvline(self.delay, color = 'grey', alpha=0.5, ls = '--')
                plt.axvline(0, color = 'grey', alpha=0.5, ls = '--')
                plt.ylabel('CD_recovery projection (a.u.)')
                
                
            return orthonormal_basis
            
            
        return CD_input_mode

    def recovery_vector(self, orthog=True, plot=False, return_opto=False):
        """
        Get the recovery vector by subtracting the end of delay(3s) post stimulation CD projection
        from the beginning of post stim recovery (1.5s)

        Returns
        -------
        orthonormal basis of the input vector

        """
        # Trial type
        r_corr = cat((np.where(self.R_correct)[0], np.where(self.R_wrong)[0]))
        l_corr = cat((np.where(self.L_correct)[0], np.where(self.L_wrong)[0]))
        
        # Lick dir
        if self.lickdir:
            r_corr = cat((np.where(self.R_correct)[0], np.where(self.L_wrong)[0]))
            l_corr = cat((np.where(self.L_correct)[0], np.where(self.R_wrong)[0]))

        r_trials_opto = [i for i in r_corr if i in self.i_good_stim_trials and not self.early_lick[i]]
        l_trials_opto = [i for i in l_corr if i in self.i_good_stim_trials and not self.early_lick[i]]
        
        alloptotrain = cat(([r_trials_opto[t] for t in self.r_train_opto_idx], [l_trials_opto[t] for t in self.l_train_opto_idx])) 
        alloptotest = cat(([r_trials_opto[t] for t in self.r_test_opto_idx], [l_trials_opto[t] for t in self.l_test_opto_idx])) 
        
        recovered_activity = np.mean([trial[self.good_neurons, self.response] for trial in self.dff[0, alloptotrain]], axis=0)
        poststim_activity = np.mean([trial[self.good_neurons, self.delay+9] for trial in self.dff[0, alloptotrain]], axis=0)
        # print(control_activity.shape)
        # CD_all = [(recovered_activity - poststim_activity) / 2]
            
        CD_recovery_mode = (recovered_activity - poststim_activity) / 2    
        
        if orthog: # Process to orthogonalize
            activityRL = []
            for n in self.good_neurons:
                
                activityRL += [np.mean([trial[n, range(self.delay, self.response)] for trial in self.dff[0, alloptotrain]], axis=0)]
                
            activityRL = activityRL - np.mean(activityRL, axis=1, keepdims=True) # remove?
            
            u, s, v = np.linalg.svd(activityRL.T)
            proj_allDim = activityRL.T @ v
        
    
        
            # Relevant choice dims
            # CD_choice_mode = [] # Late delay period
            
            CD_recovery_mode = CD_recovery_mode / np.linalg.norm(CD_recovery_mode)
            # return CD_choice_mode, 0
            # Reshape 
            
            CD_recovery_mode = np.reshape(CD_recovery_mode, (-1, 1)) 
    
            start_time = time.time()
            input_ = np.concatenate((CD_recovery_mode, v), axis=1)
            # orthonormal_basis = self.Gram_Schmidt_process(input_)
            orthonormal_basis, _ = np.linalg.qr(input_, mode='complete')  # lmao
            orthonormal_basis = orthonormal_basis[0]
            
            if return_opto:

                activity_optotest = []
                
                for t in alloptotest:
                    activity = self.dff[0, t][self.good_neurons] 
                    activity = activity - np.tile(np.mean(activityRL, axis=1)[:, None], (1, activity.shape[1]))
                    proj_allDim = np.dot(activity.T, orthonormal_basis)
                    activity_optotest += [proj_allDim[range(self.delay+6, self.response)]]
                
                return np.array(activity_optotest)
                
                
            
            if plot:
                
                # activityRL_test= np.concatenate((self.PSTH_r_test_correct, 
                #                                 self.PSTH_l_test_correct), axis=1)
                
                activityRL_testR = []
                for n in self.good_neurons:
                    
                    activityRL_testR += [np.mean([trial[n, :self.time_cutoff] for trial in self.dff[0, [r_trials_opto[t] for t in self.r_test_opto_idx]]], axis=0)]
                    
                activityRL_testL = []
                for n in self.good_neurons:
                    
                    activityRL_testL += [np.mean([trial[n, :self.time_cutoff] for trial in self.dff[0, [l_trials_opto[t] for t in self.l_test_opto_idx]]], axis=0)]
                    
                x = np.arange(-6.97,4,self.fs)[:self.time_cutoff]

                # orthonormal_basis = orthonormal_basis.reshape(-1,1)

                # Project for every trial
                for t in self.r_test_opto_idx:
                    activity = self.dff[0, r_trials_opto[t]][self.good_neurons] 
                    activity = activity - np.tile(np.mean(activityRL, axis=1)[:, None], (1, activity.shape[1]))
                    proj_allDim = np.dot(activity.T, orthonormal_basis)
                    plt.plot(x, proj_allDim[:len(self.T_cue_aligned_sel)], 'b', alpha = 0.5,  linewidth = 0.5)
                    
                for t in self.l_test_opto_idx:
                    activity = self.dff[0, l_trials_opto[t]][self.good_neurons]
                    activity = activity - np.tile(np.mean(activityRL, axis=1)[:, None], (1, activity.shape[1]))
                    proj_allDim = np.dot(activity.T, orthonormal_basis)
                    plt.plot(x, proj_allDim[:len(self.T_cue_aligned_sel)], 'r', alpha = 0.5, linewidth = 0.5)
                    
                # opto test trials
                activityRL_test = cat((activityRL_testR, activityRL_testL), axis=1)
                activityRL_test = activityRL_test - np.tile(np.mean(activityRL, axis=1)[:, None], (1, activityRL_test.shape[1]))  # remove mean
                proj_allDim = np.dot(activityRL_test.T, orthonormal_basis)
                print(activityRL_test.shape)

                print(proj_allDim.shape)
                
                plt.plot(x, proj_allDim[:len(self.T_cue_aligned_sel)], 'b', linewidth = 2)
                plt.plot(x, proj_allDim[len(self.T_cue_aligned_sel):], 'r', linewidth = 2)
                plt.title('Recovery decoder projections')
                plt.axvline(self.sample, color = 'grey', alpha=0.5, ls = '--')
                plt.axvline(self.delay, color = 'grey', alpha=0.5, ls = '--')
                plt.axvline(0, color = 'grey', alpha=0.5, ls = '--')
                plt.ylabel('CD_recovery projection (a.u.)')
                
        
            return orthonormal_basis
        
        
        return CD_recovery_mode
    
    
    def modularity_proportion_by_CD(self, mode_input = 'choice', trials=None, 
                                    period=None, normalize=True, return_trials=False,
                                    applied=[]):
        """Returns the modularity as a proportion of control CD
        
        Define CD using all trials
        
        Return modularity as a delta from CD at given time points for left and right
        traces independently
                                        
        Parameters
        ----------
            
        trials : array, optional
            Trials used to calculate recovery for behavior state analysis
                        
        period : array, optional
            Time period used to calculate modularity (either during stim or at 
                                                      end of delay)
            
        applied : array, optional
            If not empty, use this orthonormal basis/mean to calculate robustness
        Returns
        -------
        Array of length corresponding to the number of states in states.npy object
        
        """
        
        idx_map = {'choice': 1, 'action':5, 'stimulus':0}
        idx = idx_map[mode_input]

        if len(applied) != 0:
            orthonormal_basis, mean = applied
        else:
            orthonormal_basis, mean = self.plot_behaviorally_relevant_modes(plot=False) # one method
            orthonormal_basis = orthonormal_basis[:, idx]
            
        activityRL_train= np.concatenate((self.PSTH_r_train_correct, 
                                        self.PSTH_l_train_correct), axis=1)

        activityRL_test= np.concatenate((self.PSTH_r_test_correct, 
                                        self.PSTH_l_test_correct), axis=1)
        
       
        x = np.arange(-6.97,4,self.fs)[:self.time_cutoff]
        if period is None: 
            period = range(self.response-int(1*1/self.fs),self.response) # Last second

        # orthonormal_basis = orthonormal_basis.reshape(-1,1)
        i_pc = 0

        # Project for every control trial
        # for t in self.r_test_idx:
        #     activity = self.dff[0, r_trials[t]][self.good_neurons] 
        #     activity = activity - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activity.shape[1]))
        #     proj_allDim = np.dot(activity.T, orthonormal_basis)
        #     # plt.plot(x, proj_allDim[:len(self.T_cue_aligned_sel), i_pc], 'b', alpha = 0.5,  linewidth = 0.5)
            
        # for t in self.l_test_idx:
        #     activity = self.dff[0, l_trials[t]][self.good_neurons]
        #     activity = activity - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activity.shape[1]))
        #     proj_allDim = np.dot(activity.T, orthonormal_basis)
        #     # plt.plot(x, proj_allDim[:len(self.T_cue_aligned_sel), i_pc], 'r', alpha = 0.5, linewidth = 0.5)
        if normalize:
            # Get mean and STD
            
            proj_allDim = np.dot(activityRL_train.T, orthonormal_basis)
            meantrain, meanstd = np.mean(proj_allDim), np.std(proj_allDim)

        # Correct trials
        activityRL_test = activityRL_test - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activityRL_test.shape[1]))  # remove mean
        proj_allDim = np.dot(activityRL_test.T, orthonormal_basis)
        if normalize:

            proj_allDim = (proj_allDim - meantrain) / meanstd
        
        # control_traces = proj_allDim[:len(self.T_cue_aligned_sel)], proj_allDim[len(self.T_cue_aligned_sel):]
        # if not return_traces:
        #     # Plot average control traces as dotted lines
        #     plt.plot(x, proj_allDim[:len(self.T_cue_aligned_sel)], 'b', ls = '--', linewidth = 0.5)
        #     plt.plot(x, proj_allDim[len(self.T_cue_aligned_sel):], 'r', ls = '--', linewidth = 0.5)
        #     plt.title("Choice decoder projections with opto")
        #     plt.axvline(self.sample, color = 'grey', alpha=0.5, ls = '--')
        #     plt.axvline(self.delay, color = 'grey', alpha=0.5, ls = '--')
        #     plt.axvline(0, color = 'grey', alpha=0.5, ls = '--')
        #     plt.ylabel('CD_delay projection (a.u.)')
        left_control_traces = proj_allDim[len(self.T_cue_aligned_sel):]
        right_control_traces = proj_allDim[:len(self.T_cue_aligned_sel)]
        
        

        r_opto, l_opto = self.get_trace_matrix_multiple(self.good_neurons, opto=True)

        # activityRL_opto= np.concatenate((r_opto, l_opto), axis=1)
        
        
        r_corr = np.where(self.R_correct + self.L_wrong)[0]
        l_corr = np.where(self.L_correct + self.R_wrong)[0]

        
        # Project for every opto trial
        r_trials = [i for i in r_corr if self.stim_ON[i] and not self.early_lick[i]]
        l_trials = [i for i in l_corr if self.stim_ON[i] and not self.early_lick[i]]
        
        r_proj_delta = []
        l_proj_delta = []
        for r in r_trials:
            activity = self.dff[0, r][self.good_neurons] 
            activity = activity - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activity.shape[1]))
            proj_allDim = np.dot(activity.T, orthonormal_basis)
            proj_allDim = (proj_allDim - meantrain) / meanstd
            r_proj_delta += [(right_control_traces[period] - proj_allDim[period]) / right_control_traces[period]]
            
        for l in l_trials:
            activity = self.dff[0, l][self.good_neurons]
            activity = activity - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activity.shape[1]))
            proj_allDim = np.dot(activity.T, orthonormal_basis)
            proj_allDim = (proj_allDim - meantrain) / meanstd
            l_proj_delta += [(left_control_traces[period] - proj_allDim[period]) / left_control_traces[period]]
            
        if return_trials:
            return r_trials, l_trials, r_proj_delta, l_proj_delta

        recovery = 0
        
        if len(r_proj_delta) != 0:
            recovery += np.abs(np.mean(r_proj_delta)) 
        if len(l_proj_delta) != 0:
            recovery += np.abs(np.mean(l_proj_delta))

        return recovery

  


    
    def modularity_proportion_by_stateCD(self, mode_input = 'choice', trials=None):
        """Returns the modularity as a proportion of control CD
        
        Define CD using all trials
        
        Return modularity as a delta from CD at end of delay for left and right
        traces independently
        
        Assume there are only two states
                                
        Parameters
        ----------
            
        trials : array, optional
            Trials used to calculate recovery for behavior state analysis
            
        Returns
        -------
        Array of length corresponding to the number of states in states.npy object
        
        """
        
        idx_map = {'choice': 1, 'action':5, 'stimulus':0}
        idx = idx_map[mode_input]

        orthonormal_basis, mean = self.plot_behaviorally_relevant_modes(plot=False) # one method
        orthonormal_basis = orthonormal_basis[:, idx]
            
        activityRL_train= np.concatenate((self.PSTH_r_train_correct, 
                                        self.PSTH_l_train_correct), axis=1)

        activityRL_test= np.concatenate((self.PSTH_r_test_correct, 
                                        self.PSTH_l_test_correct), axis=1)
        
       
        x = np.arange(-6.97,4,self.fs)[:self.time_cutoff]

        # orthonormal_basis = orthonormal_basis.reshape(-1,1)
        i_pc = 0

        # Project for every control trial
        # for t in self.r_test_idx:
        #     activity = self.dff[0, r_trials[t]][self.good_neurons] 
        #     activity = activity - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activity.shape[1]))
        #     proj_allDim = np.dot(activity.T, orthonormal_basis)
        #     # plt.plot(x, proj_allDim[:len(self.T_cue_aligned_sel), i_pc], 'b', alpha = 0.5,  linewidth = 0.5)
            
        # for t in self.l_test_idx:
        #     activity = self.dff[0, l_trials[t]][self.good_neurons]
        #     activity = activity - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activity.shape[1]))
        #     proj_allDim = np.dot(activity.T, orthonormal_basis)
        #     # plt.plot(x, proj_allDim[:len(self.T_cue_aligned_sel), i_pc], 'r', alpha = 0.5, linewidth = 0.5)
        if normalize:
            # Get mean and STD
            
            proj_allDim = np.dot(activityRL_train.T, orthonormal_basis)
            meantrain, meanstd = np.mean(proj_allDim), np.std(proj_allDim)

        # Correct trials
        activityRL_test = activityRL_test - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activityRL_test.shape[1]))  # remove mean
        proj_allDim = np.dot(activityRL_test.T, orthonormal_basis)
        if normalize:

            proj_allDim = (proj_allDim - meantrain) / meanstd
        
        # control_traces = proj_allDim[:len(self.T_cue_aligned_sel)], proj_allDim[len(self.T_cue_aligned_sel):]
        # if not return_traces:
        #     # Plot average control traces as dotted lines
        #     plt.plot(x, proj_allDim[:len(self.T_cue_aligned_sel)], 'b', ls = '--', linewidth = 0.5)
        #     plt.plot(x, proj_allDim[len(self.T_cue_aligned_sel):], 'r', ls = '--', linewidth = 0.5)
        #     plt.title("Choice decoder projections with opto")
        #     plt.axvline(self.sample, color = 'grey', alpha=0.5, ls = '--')
        #     plt.axvline(self.delay, color = 'grey', alpha=0.5, ls = '--')
        #     plt.axvline(0, color = 'grey', alpha=0.5, ls = '--')
        #     plt.ylabel('CD_delay projection (a.u.)')
        left_control_traces = proj_allDim[len(self.T_cue_aligned_sel):]
        right_control_traces = proj_allDim[:len(self.T_cue_aligned_sel)]
        
        states = np.load(r'{}\states.npy'.format(self.path))
            
        num_state = states.shape[1]
        
        
        all_recovery = []

        
        for i in range(num_state):
        
            r_opto, l_opto = self.get_trace_matrix_multiple(self.good_neurons, opto=True)
    
            # activityRL_opto= np.concatenate((r_opto, l_opto), axis=1)
            
            trials = self.find_bias_trials(state = i)
            
            r_corr = np.where(self.R_correct + self.L_wrong)[0]
            l_corr = np.where(self.L_correct + self.R_wrong)[0]
            r_corr = [i for i in r_corr if i in trials]
            l_corr = [i for i in l_corr if i in trials]
            
            # Project for every opto trial
            r_trials = [i for i in r_corr if self.stim_ON[i] and not self.early_lick[i]]
            l_trials = [i for i in l_corr if self.stim_ON[i] and not self.early_lick[i]]
            
            r_proj_delta = []
            l_proj_delta = []
            for r in r_trials:
                activity = self.dff[0, r][self.good_neurons] 
                activity = activity - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activity.shape[1]))
                proj_allDim = np.dot(activity.T, orthonormal_basis)
                proj_allDim = (proj_allDim - meantrain) / meanstd
                r_proj_delta += [right_control_traces[self.response-int(1*1/self.fs):self.response] - proj_allDim[self.response-int(1*1/self.fs):self.response]]
                # plt.plot(x, proj_allDim[:len(self.T_cue_aligned_sel), i_pc], 'b', alpha = 0.5,  linewidth = 0.5)
                
            for l in l_trials:
                activity = self.dff[0, l][self.good_neurons]
                activity = activity - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activity.shape[1]))
                proj_allDim = np.dot(activity.T, orthonormal_basis)
                proj_allDim = (proj_allDim - meantrain) / meanstd
                l_proj_delta += [left_control_traces[self.response-int(1*1/self.fs):self.response] - proj_allDim[self.response-int(1*1/self.fs):self.response]]
                # plt.plot(x, proj_allDim[:len(self.T_cue_aligned_sel), i_pc], 'r', alpha = 0.5, linewidth = 0.5)
                
                
            # # Opto trials
            # activityRL_opto = activityRL_opto - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activityRL_opto.shape[1]))  # remove mean
            # proj_allDim = np.dot(activityRL_opto.T, orthonormal_basis)
            # if normalize:
    
            #     proj_allDim = (proj_allDim - meantrain) / meanstd
                
            # if return_traces:
            #     opto_traces = proj_allDim[:len(self.T_cue_aligned_sel)], proj_allDim[len(self.T_cue_aligned_sel):]
            #     error_bars = stats.sem(r_proj, axis=0), stats.sem(l_proj, axis=0)
            #     if return_applied:
            #         return control_traces, opto_traces, error_bars, orthonormal_basis, np.mean(activityRL_train, axis=1)[:, None], meantrain, meanstd
            #     else:
    
            #         return control_traces, opto_traces, error_bars
            
            # plt.plot(x, proj_allDim[:len(self.T_cue_aligned_sel)], 'b', linewidth = 2)
            # plt.plot(x, proj_allDim[len(self.T_cue_aligned_sel):], 'r', linewidth = 2)
            
            # plt.fill_between(x, proj_allDim[len(self.T_cue_aligned_sel):] - stats.sem(l_proj, axis=0), 
            #          proj_allDim[len(self.T_cue_aligned_sel):] +  stats.sem(l_proj, axis=0),
            #          color=['#ffaeb1'])
            # plt.fill_between(x, proj_allDim[:len(self.T_cue_aligned_sel)] - stats.sem(r_proj, axis=0), 
            #          proj_allDim[:len(self.T_cue_aligned_sel)] + stats.sem(r_proj, axis=0),
            #          color=['#b4b2dc'])
            
            # plt.hlines(y=max(proj_allDim[:]) + 0.5, xmin=self.delay, xmax=-2, linewidth=10, color='red')
            
            recovery = 0
            
            if len(r_proj_delta) != 0:
                recovery += np.abs(np.mean(r_proj_delta)) 
            if len(l_proj_delta) != 0:
                recovery += np.abs(np.mean(l_proj_delta))

            all_recovery += [recovery]

        return all_recovery

  