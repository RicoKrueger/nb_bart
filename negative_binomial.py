from joblib import Parallel, delayed
import os
import sys
import time
import numpy as np
import pandas as pd
from numba import jit
import pypolyagamma as pypolyagamma
import h5py
from scipy.sparse.linalg import expm
from scipy.stats import invwishart
from scipy.special import loggamma
import bayestree as bt
from scipy.stats import rankdata

class Data:
    """ Holds the training data.
    
    Attributes:
        y (array): dependent count data.
        N (int): number of dependent data.
        x_fix (array): covariates pertaining to fixed link function parameters.
        n_fix (int): number of covariates pertaining to fixed link function 
        parameters.
        x_rnd (array): covariates pertaining to random link function parameters.
        n_rnd (int): number of covariates pertaining to random link function 
        parameters.
        W (array): row-normalised spatial weight matrix.
    """   
    
    def __init__(self, y, x_fix, x_rnd, W):
        self.y = y
        self.N = y.shape[0]
        
        self.x_fix = x_fix
        if x_fix is not None:
            self.n_fix = x_fix.shape[1]
        else:
            self.n_fix = 0
            
        self.x_rnd = x_rnd
        if x_rnd is not None:
            self.n_rnd = x_rnd.shape[1] 
        else:
            self.n_rnd = 0
            
        self.W = W
        

class Options:
    """ Contains options for MCMC algorithm.
    
    Attributes:
        model_name (string): name of the model to be estimated.
        nChain (int): number of Markov chains.
        nBurn (int): number of samples to be discarded for burn-in. 
        nSample (int): number of samples after burn-in period.
        nIter (int): total number of samples.
        nThin (int): thinning factors.
        nKeep (int): number of samples to be retained.
        nMem (int): number of samples to retain in memory.
        disp (int): number of samples after which progress is displayed. 
        mh_step_initial (float): Initial MH step size.
        mh_target (float): target MH acceptance rate.
        mh_correct (float): correction of MH step to reach desired target 
        acceptance rate.
        mh_window (int): number of samples after which to adjust MG step size.
        delete_draws (bool): Boolean indicating whether simulation draws should
        be deleted.
        seed (int): random seed.
    """   

    def __init__(
            self, 
            model_name='test',
            nChain=1, nBurn=500, nSample=500, nThin=2, nMem=None, disp=100, 
            mh_step_initial=0.1, mh_target=0.3, mh_correct=0.01, mh_window=50,
            delete_draws=True, seed=4711
            ):
        self.model_name = model_name
        self.nChain = nChain
        self.nBurn = nBurn
        self.nSample = nSample
        self.nIter = nBurn + nSample
        self.nThin = nThin
        self.nKeep = int(nSample / nThin)
        if nMem is None:
            self.nMem = self.nKeep
        else:
            self.nMem = nMem
        self.disp = disp
        
        self.mh_step_initial = mh_step_initial
        self.mh_target = mh_target
        self.mh_correct = mh_correct
        self.mh_window = mh_window
        
        self.delete_draws = delete_draws
        self.seed = seed
        
class Results:
    """ Holds simulation results.
    
    Attributes:
        options (Options): Options used for MCMC algorithm.
        bart_options (BartOptions): Options used for BART component.
        estimation time (float): Estimation time.
        lppd (float): Log pointwise predictive density.
        waic (float): Widely applicable information criterion.
        post_mean_lam (array): Posterior mean of the expected count of each 
        observation.
        rmse (float): Root mean squared error between the predicted count and 
        observed count.
        mae (float): Mean absolute error between the predicted count and the 
        observed count.
        rmsle (float): Mean squared logarithmic errror between the predicted 
        count and the observed count.
        post_mean_ranking (array): Posterior mean site rank.
        post_mean_ranking_top (array): Posterior mean probability that a site
        belongs to the top m most hazardous sites.
        ranking_top_m_list (list): List of m values used for calculating the 
        posterior mean probability that a site belongs to the top m most 
        hazardous sites.
        post_r (DataFrame): Posterior summary of the negative binomial success 
        rate.
        post_mean_f (array): Posterior mean of the residual value of the link
        function excluding fixed, random and spatial link function components
        (required for BART calibration).
        post_variable_inclusion_probs (DataFrame): Posterior summary of BART 
        variable inclusion proportions. 
        post_beta (DataFrame): Posterior summary of fixed link function 
        parameters.
        post_mu (DataFrame): Posterior summary of mean of random link function
        parameters.
        post_sigma (DataFrame): Posterior summary of standard deviation of 
        random link function parameters.
        post_Sigma (DataFrame): Posterior summary of covariance of random link
        function parameters. 
        post_sigma_mess: Posterior summary of spatial error scale.
        post_tau: Posterior summary of MESS association parameters. 
    """   
    
    def __init__(self, 
                 options, bart_options, toc, 
                 lppd, waic,
                 post_mean_lam, 
                 rmse, mae, rmsle,
                 post_mean_ranking, 
                 post_mean_ranking_top, ranking_top_m_list,
                 post_r, post_mean_f, 
                 post_beta,
                 post_mu, post_sigma, post_Sigma,
                 post_variable_inclusion_props,
                 post_sigma_mess, post_tau):
        self.options = options
        self.bart_options = bart_options
        self.estimation_time = toc
        
        self.lppd = lppd
        self.waic = waic
        self.post_mean_lam = post_mean_lam
        self.rmse = rmse
        self.mae = mae
        self.rmsle = rmsle
        
        self.post_mean_ranking = post_mean_ranking
        self.post_mean_ranking_top = post_mean_ranking_top
        self.ranking_top_m_list = ranking_top_m_list
        
        self.post_r = post_r
        self.post_mean_f = post_mean_f
        
        self.post_variable_inclusion_props = post_variable_inclusion_props
        
        self.post_beta = post_beta
        
        self.post_mu = post_mu
        self.post_sigma = post_sigma
        self.post_Sigma = post_Sigma
        
        self.post_sigma_mess = post_sigma_mess
        self.post_tau = post_tau
        
        
class NegativeBinomial:
    """ MCMC method for posterior inference in negative binomial model. """
    
    @staticmethod
    def _F_matrix(y):
        """ Calculates F matrix. """
        y_max = np.max(y)
        F = np.zeros((y_max, y_max))
        for m in np.arange(y_max):
            for j in np.arange(m+1):
                if m==0 and j==0:
                    F[m,j] = 1
                else:
                    F[m,j] = m/(m+1) * F[m-1,j] + (1/(m+1)) * F[m-1,j-1]
        return F
    
    def __init__(self, data, data_bart=None):
        self.data = data
        self.data_bart = data_bart
        self.F = self._F_matrix(data.y)
        self.N = data.N
        self.n_fix = data.n_fix
        self.n_rnd = data.n_rnd
        self.mess = data.W is not None

    ###
    #Convenience
    ###
    
    @staticmethod
    def _pg_rnd(a, b):
        """ Takes draws from Polya-Gamma distribution with parameters a, b. """
        ppg = pypolyagamma.PyPolyaGamma(np.random.randint(2**16))
        N = a.shape[0]
        r = np.zeros((N,))
        ppg.pgdrawv(a, b, r)
        return r
    
    @staticmethod
    def _mvn_prec_rnd(mu, P):
        """ Takes draws from multivariate normal with mean mu and precision P. 
        
        Keywords:
            mu (array): Mean vector.
            P (array): Precision matrix.
            
        Returns:
            r (array): Multivariate normal draw.
        """ 
        
        d = mu.shape[0]
        P_chT = np.linalg.cholesky(P).T
        r = mu + np.linalg.solve(P_chT, np.random.randn(d,1)).reshape(d,)
        return r 
    
    @staticmethod
    def _mvn_prec_rnd_arr(mu, P):
        """ Takes draws from multivariate normal with mean mu and precision P.
        
        Keywords:
            mu (array): Array of mean vectors with shape N x d.
            P (array): Array of precision matrices with shape N x d x d.
            
        Returns:
            r (array): Array of multivariate normal draws with shape N x d.
        """
        
        N, d = mu.shape
        P_chT = np.moveaxis(np.linalg.cholesky(P), [1,2], [2,1])
        r = mu + np.linalg.solve(P_chT, np.random.randn(N,d,1)).reshape(N,d)
        return r 
    
    @staticmethod
    def _nb_lpmf(y, psi, r):
        """ Calculates log pmf of negative binomial. """
        lc = loggamma(y + r) - loggamma(r) - loggamma(y + 1)
        lp = y * psi - (y + r) * np.log1p(np.exp(psi))
        lpmf = lc + lp
        return lpmf
    
    ###
    #Sampling
    ###
        
    def _next_omega(self, r, psi):
        """ Updates omega. """
        omega = self._pg_rnd(self.data.y + r, psi)
        Omega = np.diag(omega)
        z = (self.data.y - r) / 2 / omega
        return omega, Omega, z

    def _next_beta(self, z, psi_rnd, psi_bart, phi, 
                   Omega, beta_mu0, beta_Si0Inv):
        """ Updates beta. """
        beta_SiInv = self.data.x_fix.T @ Omega @ self.data.x_fix + beta_Si0Inv
        beta_mu = np.linalg.solve(beta_SiInv,
                                  self.data.x_fix.T @ Omega \
                                  @ (z - psi_rnd - psi_bart - phi) \
                                  + beta_Si0Inv @ beta_mu0)
        beta = self._mvn_prec_rnd(beta_mu, beta_SiInv)
        return beta
    
    def _next_gamma(self, z, psi_fix, psi_bart, phi, omega, mu, SigmaInv):
        """ Updates gamma. """
        gamma_SiInv = omega.reshape(self.N,1,1) * \
            self.data.x_rnd.reshape(self.N,self.n_rnd,1) \
            @ self.data.x_rnd.reshape(self.N,1,self.n_rnd) \
            + SigmaInv.reshape(1,self.n_rnd,self.n_rnd)
        gamma_mu_pre = (omega * (z - psi_fix - psi_bart - phi))\
            .reshape(self.N,1) \
            * self.data.x_rnd + (SigmaInv @ mu).reshape(1,self.n_rnd)
        gamma_mu = np.linalg.solve(gamma_SiInv, gamma_mu_pre)
        gamma = self._mvn_prec_rnd_arr(gamma_mu, gamma_SiInv)            
        return gamma
 
    def _next_mu(self, gamma, SigmaInv, mu_mu0, mu_Si0Inv):
        """ Updates mu. """
        mu_SiInv = self.N * SigmaInv + mu_Si0Inv
        mu_mu = np.linalg.solve(mu_SiInv,
                                SigmaInv @ np.sum(gamma, axis=0)
                                + mu_Si0Inv @ mu_mu0)
        mu = self._mvn_prec_rnd(mu_mu, mu_SiInv)
        return mu

    def _next_Sigma(self, gamma, mu, a, nu):    
        """ Updates Sigma. """
        diff = gamma - mu
        Sigma = (invwishart.rvs(nu + self.N + self.n_rnd - 1, 
                                2 * nu * np.diag(a) + diff.T @ diff))\
            .reshape((self.n_rnd, self.n_rnd))
        SigmaInv = np.linalg.inv(Sigma)
        return Sigma, SigmaInv

    def _next_a(self, SigmaInv, nu, A):
        """ Updates a. """
        a = np.random.gamma((nu + self.n_rnd) / 2, 
                            1 / (1 / A**2 + nu * np.diag(SigmaInv)))
        return a

    def _next_phi(self, z, psi_fix, psi_rnd, psi_bart, Omega, Omega_tilde):
        """ Updates phi. """
        phi_SiInv = Omega + Omega_tilde
        phi_mu = np.linalg.solve(phi_SiInv, 
                                 Omega @ (z - psi_fix - psi_rnd - psi_bart))
        phi = self._mvn_prec_rnd(phi_mu, phi_SiInv)
        return phi      
    
    def _next_sigma2(self, phi, S, sigma2_b0, sigma2_c0):
        """ Updates sigma_mess**2. """
        b = sigma2_b0 + self.data.N / 2
        c = sigma2_c0 + phi.T @ S.T @ S @ phi / 2
        sigma2 = 1 / np.random.gamma(b, 1 / c)
        return sigma2
    
    def _log_target_tau(self, tau, S, phi, sigma2, tau_mu0, tau_si0):
        """ Calculates target density for MH. """
        if S is None:
            S = expm(tau * self.data.W)
        Omega_tilde = S.T @ S / sigma2
        lt = - phi.T @ Omega_tilde @ phi / 2 \
             - (tau - tau_mu0)**2 / 2 / tau_si0**2
        return lt, S
    
    def _next_tau(self, tau, S, phi, sigma2, tau_mu0, tau_si0, mh_step):
        """ Updates tau. """
        lt_tau, S = self._log_target_tau(tau, S, phi, sigma2, tau_mu0, tau_si0)
        tau_star = tau + np.sqrt(mh_step) * np.random.randn()
        lt_tau_star, S_star = self._log_target_tau(
            tau_star, None, phi, sigma2, tau_mu0, tau_si0
            )
        log_r = np.log(np.random.rand())
        log_alpha = lt_tau_star - lt_tau
        if log_r <= log_alpha:
            tau = tau_star
            S = np.array(S_star)
            mh_tau_accept = True
        else:
            mh_tau_accept = False
        return tau, S, mh_tau_accept
    
    @staticmethod
    def _next_h(r, r0, b0, c0):
        """ Updates h. """
        h = np.random.gamma(r0 + b0, 1/(r + c0))
        return h
    
    @staticmethod
    @jit
    def _next_L(y, r, F):
        """ Updates L. """
        N = y.shape[0]
        L = np.zeros((N,))
        for n in np.arange(N):
            if y[n]:
                numer = np.zeros((y[n],))
                for j in np.arange(y[n]):
                    numer[j] = F[y[n]-1,j] * r**(j+1)
                L_p = numer / np.sum(numer)
                L[n] = np.searchsorted(np.cumsum(L_p), np.random.rand()) + 1
        return L
    
    @staticmethod
    def _next_r(r0, L, h, psi):
        """ Updates r. """
        sum_p = np.sum(np.log1p(np.exp(psi)))
        r = np.random.gamma(r0 + np.sum(L), 1 / (h + sum_p))
        return r
    
    @staticmethod
    def _rank(lam, ranking_top_m_list):
        """ Computes the rank of each site.
        
        Keywords:
            lam (array): Expected counts.
            ranking_top_m_list (list): List of m values used for extracting 
            whether a site belongs to the top m most hazardous sites.
            
        Returns:
            ranking (array): Ranks 
            ranking_top (array): Booleans indicating whether a site belongs to 
            the top m most hazardous sites. 
        """
        
        ranking = rankdata(-lam, method='min')
        ranking_top = np.zeros((lam.shape[0], len(ranking_top_m_list)))
        for j, m in enumerate(ranking_top_m_list):
            ranking_top[:,j] = ranking <= m
        return ranking, ranking_top
    
    def _mcmc_chain(
            self,
            chainID, 
            options, bart_options,
            r0, b0, c0,
            beta_mu0, beta_Si0Inv,
            mu_mu0, mu_Si0Inv, nu, A,
            sigma2_b0, sigma2_c0, tau_mu0, tau_si0,
            r_init, beta_init, mu_init, Sigma_init,
            ranking_top_m_list):
        """ Markov chain for MCMC simulation for negative binomial model.
        
        Keywords:
            chainID (int): ID of Markov chain.
            options (Options): Simulation options.
            bart_options (BartOptions): Options for BART component.
            r0 (float): Hyperparameter of prior on r; r ~ Gamma(r0, h).
            b0 (float): Hyperparameter of prior on h; h ~ Gamma(b0, c0).
            c0 (float): Hyperparameter of prior on h; h ~ Gamma(b0, c0).
            beta_mu0 (array): Hyperparameter of prior on beta; 
            beta ~ N(beta_mu0, beta_Si0)
            beta_Si0Inv (array): Inverse of hyperparameter of prior on beta; 
            beta ~ N(beta_mu0, beta_Si0)
            mu_mu0 (array): Hyperparameter of prior on mu; 
            mu ~ N(mu_mu0, mu_Si0)
            mu_Si0Inv (array): Inverse of hyperparameter of prior on mu; 
            mu ~ N(mu_mu0, mu_Si0).
            nu (float) ~ Hyperparameter of prior on a; a ~ Gamma(nu, A_k)
            A (array): Hyperparameter of prior on a; a ~ Gamma(nu, A_k)
            sigma2_b0 (float): Hyperparameter of prior on sigma2; 
            sigma2 ~ Gamma(sigma2_b0, sigma2_c0).
            sigma2_c0 (float): Hyperparameter of prior on sigma2; 
            sigma2 ~ Gamma(sigma2_b0, sigma2_c0).
            tau_mu0 (float): Hyperparameter of prior on tau; 
            tau ~ N(tau_mu0, tau_si0**2).
            tau_si0 (float): Hyperparameter of prior on tau; 
            tau ~ N(tau_mu0, tau_si0**2).
            r_init (float): Initial value of r.
            beta_init (array): Initial value of beta.
            mu_init (array): Initial value of mu.
            Sigma_init (array): Initial value of Sigma.
            ranking_top_m_list (list): List of m values used for extracting 
            whether a site belongs to the top m most hazardous sites.
        """
        
        ###
        #Storage
        ###
        
        file_name = options.model_name + '_draws_chain' + str(chainID + 1) \
        + '.hdf5'
        if os.path.exists(file_name):
            os.remove(file_name) 
        file = h5py.File(file_name, 'a')    
        
        lp_store = file.create_dataset('lp_store', \
        (options.nKeep,self.N), dtype='float64')
        lp_store_tmp = np.zeros((options.nMem,self.N))
        
        lam_store = file.create_dataset('lam_store', \
        (options.nKeep,self.N), dtype='float64')
        lam_store_tmp = np.zeros((options.nMem,self.N))
        
        ranking_store = file.create_dataset('ranking_store', \
        (options.nKeep,self.N), dtype='float64')
        ranking_store_tmp = np.zeros((options.nMem,self.N))
        
        ranking_top_store = file.create_dataset('ranking_top_store', \
        (options.nKeep,self.N,len(ranking_top_m_list)), dtype='float64')
        ranking_top_store_tmp = np.zeros((options.nMem, self.N, 
                                          len(ranking_top_m_list)))
        r_store = file.create_dataset('r_store', \
        (options.nKeep,), dtype='float64')
        r_store_tmp = np.zeros((options.nMem,))
        
        f_store = file.create_dataset('f_store', \
        (options.nKeep,self.N), dtype='float64')
        f_store_tmp = np.zeros((options.nMem,self.N))
        
        if self.n_fix:
            beta_store = file.create_dataset('beta_store', \
            (options.nKeep, self.data.n_fix), dtype='float64')
            beta_store_tmp = np.zeros((options.nMem, self.data.n_fix))
            
        if self.n_rnd:
            mu_store = file.create_dataset('mu_store', \
            (options.nKeep, self.data.n_rnd), dtype='float64')
            mu_store_tmp = np.zeros((options.nMem, self.data.n_rnd))
            
            sigma_store = file.create_dataset('sigma_store', \
            (options.nKeep, self.data.n_rnd), dtype='float64')
            sigma_store_tmp = np.zeros((options.nMem, self.data.n_rnd))
            
            Sigma_store = file.create_dataset('Sigma_store', \
            (options.nKeep, self.data.n_rnd, self.data.n_rnd), dtype='float64')
            Sigma_store_tmp \
                = np.zeros((options.nMem, self.data.n_rnd, self.data.n_rnd))
                
        if self.data_bart is not None:
            avg_tree_acceptance_store = np.zeros((options.nIter,))
            avg_tree_depth_store = np.zeros((options.nIter,))
                
            variable_inclusion_props_store \
                = file.create_dataset('variable_inclusion_props_store', \
                                      (options.nKeep, self.data_bart.J), 
                                      dtype='float64')
            variable_inclusion_props_store_tmp \
                = np.zeros((options.nMem, self.data_bart.J))              
        
        if self.mess:
            sigma_mess_store = file.create_dataset('sigma_mess_store', \
            (options.nKeep,), dtype='float64')
            sigma_mess_store_tmp = np.zeros((options.nMem,))

            tau_store = file.create_dataset('tau_store', \
            (options.nKeep,), dtype='float64')
            tau_store_tmp = np.zeros((options.nMem,))
            
            mh_tau_accept_store = np.zeros((options.nIter,))
        
        ###
        #Initialise
        ###
        
        r = max(r_init - 0.5 + 1.0 * np.random.rand(), 0.25)
        
        if self.n_fix:
            beta = beta_init - 0.5 + 1 * np.random.rand(self.n_fix,)
            psi_fix = self.data.x_fix @ beta
        else:
            beta = None
            psi_fix = 0
            
        if self.n_rnd:
            mu = mu_init - 0.5 + 1 * np.random.rand(self.n_rnd,)
            Sigma = Sigma_init.copy()
            SigmaInv = np.linalg.inv(Sigma)
            a = np.random.gamma(1/2, 1/A**2)
            gamma = mu + (np.linalg.cholesky(Sigma) \
                     @ np.random.randn(self.n_rnd,self.N)).T
            psi_rnd = np.sum(self.data.x_rnd * gamma, axis=1)
        else:
            psi_rnd = 0
            
        if self.data_bart is not None:
            forest = bt.Forest(bart_options, self.data_bart)
            psi_bart = self.data_bart.unscale(forest.y_hat)
        else:
            forest = None
            psi_bart = 0
        
        if self.mess:
            sigma2 = np.sqrt(0.4)
            tau = -float(0.5 + np.random.rand())
            S = expm(tau * self.data.W)
            Omega_tilde = S.T @ S / sigma2         
            eps = np.sqrt(sigma2) * np.random.randn(self.data.N,)
            phi = np.linalg.solve(S, eps)
            mh_step = options.mh_step_initial
        else:
            sigma2 = None
            tau = None       
            S = None
            Omega_tilde = None
            phi = 0
            mh_step = None
  
        psi = psi_fix + psi_rnd + psi_bart + phi
        
        ###
        #Sampling
        ###
    
        j = -1
        ll = 0
        sample_state = 'burn in'    
        
        for i in np.arange(options.nIter):
            omega, Omega, z = self._next_omega(r, psi)
            
            if self.n_fix:
                beta = self._next_beta(z, psi_rnd, psi_bart, phi, 
                                       Omega, beta_mu0, beta_Si0Inv)
                psi_fix = self.data.x_fix @ beta
                
            if self.n_rnd:
                gamma = self._next_gamma(z, psi_fix, psi_bart, phi, 
                                         omega, mu, SigmaInv)
                mu = self._next_mu(gamma, SigmaInv, mu_mu0, mu_Si0Inv)
                Sigma, SigmaInv = self._next_Sigma(gamma, mu, a, nu)
                a = self._next_a(SigmaInv, nu, A)
                psi_rnd = np.sum(self.data.x_rnd * gamma, axis=1) 
                
            if self.data_bart is not None:
                sigma_weights = np.sqrt(1/omega)
                f = z - psi_fix - psi_rnd - phi
                self.data_bart.update(f, sigma_weights)
                avg_tree_acceptance_store[i], avg_tree_depth_store[i] = \
                    forest.update(self.data_bart)
                psi_bart = self.data_bart.unscale(forest.y_hat)
                
            if self.mess:
                phi = self._next_phi(z, psi_fix, psi_rnd, psi_bart,
                                     Omega, Omega_tilde)
            
            psi = psi_fix + psi_rnd + psi_bart + phi 
            
            if self.mess:
                sigma2 = self._next_sigma2(phi, S, sigma2_b0, sigma2_c0)
                tau, S, mh_tau_accept_store[i] = self._next_tau(
                    tau, S, phi, sigma2, tau_mu0, tau_si0, mh_step
                    )
                Omega_tilde = S @ S.T / sigma2
            
            h = self._next_h(r, r0, b0, c0)
            L = self._next_L(self.data.y, r, self.F)
            r = self._next_r(r0, L, h, psi)
            
            #Adjust MH step size
            if self.mess and ((i+1) % options.mh_window) == 0:
                sl = slice(max(i+1-options.mh_window,0), i+1)
                mean_accept = mh_tau_accept_store[sl].mean()
                if mean_accept >= options.mh_target:
                    mh_step += options.mh_correct
                else:
                    mh_step -= options.mh_correct
                     
            if ((i+1) % options.disp) == 0:  
                if (i+1) > options.nBurn:
                    sample_state = 'sampling'
                verbose = 'Chain ' + str(chainID + 1) \
                + '; iteration: ' + str(i + 1) + ' (' + sample_state + ')'
                if self.data_bart is not None:
                    sl = slice(max(i+1-100,0),i+1)
                    ravg_depth = np.round(
                        np.mean(avg_tree_depth_store[sl]), 2)
                    ravg_acceptance = np.round(
                        np.mean(avg_tree_acceptance_store[sl]), 2)
                    verbose = verbose \
                    + '; avg. tree depth: ' + str(ravg_depth) \
                    + '; avg. tree acceptance: ' + str(ravg_acceptance)
                if self.mess:
                    verbose = verbose \
                    + '; avg. tau acceptance: ' \
                    + str(mh_tau_accept_store[sl].mean())
                print(verbose)
                sys.stdout.flush()
                
            if (i+1) > options.nBurn:                  
                if ((i+1) % options.nThin) == 0:
                    j+=1
                    
                    lp_store_tmp[j,:] = self._nb_lpmf(self.data.y, psi, r)
                    lam = np.exp(psi + np.log(r))
                    lam_store_tmp[j,:] = lam
                    ranking_store_tmp[j,:], ranking_top_store_tmp[j,:,:] \
                        = self._rank(lam, ranking_top_m_list)
                    r_store_tmp[j] = r
                    f_store_tmp[j,:] = z - phi
                    
                    if self.n_fix:
                        beta_store_tmp[j,:] = beta
                        
                    if self.n_rnd:
                        mu_store_tmp[j,:] = mu
                        sigma_store_tmp[j,:] = np.sqrt(np.diag(Sigma))
                        Sigma_store_tmp[j,:,:] = Sigma
                        
                    if self.data_bart is not None:
                        variable_inclusion_props_store_tmp[j,:] \
                            = forest.variable_inclusion()
                        
                    if self.mess:
                        sigma_mess_store_tmp[j] = np.sqrt(sigma2)
                        tau_store_tmp[j] = tau
                        
                if (j+1) == options.nMem:
                    l = ll
                    ll += options.nMem
                    sl = slice(l, ll)
                    
                    print('Storing chain ' + str(chainID + 1))
                    sys.stdout.flush()
                    
                    lp_store[sl,:] = lp_store_tmp
                    lam_store[sl,:] = lam_store_tmp
                    ranking_store[sl,:] = ranking_store_tmp
                    ranking_top_store[sl,:,:] = ranking_top_store_tmp
                    r_store[sl] = r_store_tmp
                    f_store[sl,:] = f_store_tmp
                    
                    if self.n_fix:
                        beta_store[sl,:] = beta_store_tmp
                        
                    if self.n_rnd:
                        mu_store[sl,:] = mu_store_tmp
                        sigma_store[sl,:] = sigma_store_tmp
                        Sigma_store[sl,:,:] = Sigma_store_tmp
                        
                    if self.data_bart is not None:
                        variable_inclusion_props_store[sl,:] \
                            = variable_inclusion_props_store_tmp
                    
                    if self.mess:
                        sigma_mess_store[sl] = sigma_mess_store_tmp
                        tau_store[sl] = tau_store_tmp
                    
                    j = -1 
        
        if self.data_bart is not None:
            file.create_dataset('avg_tree_acceptance_store', 
                                data=avg_tree_acceptance_store)
            file.create_dataset('avg_tree_depth_store', 
                                data=avg_tree_depth_store)
                    
    ###
    #Posterior summary
    ###  
    
    @staticmethod
    def _posterior_summary(options, param_name, nParam, nParam2, verbose):
        """ Returns summary of posterior draws of parameters of interest. """
        headers = ['mean', 'std. dev.', '2.5%', '97.5%', 'Rhat']
        q = (0.025, 0.975)
        nSplit = 2
        
        draws = np.zeros((options.nChain, options.nKeep, nParam, nParam2))
        for c in range(options.nChain):
            file = h5py.File(options.model_name + '_draws_chain' + str(c + 1) \
                             + '.hdf5', 'r')
            draws[c,:,:,:] = np.array(file[param_name + '_store'])\
            .reshape((options.nKeep, nParam, nParam2))
            
        mat = np.zeros((nParam * nParam2, len(headers)))
        post_mean = np.mean(draws, axis=(0,1))
        mat[:, 0] = np.array(post_mean).reshape((nParam * nParam2,))
        mat[:, 1] = np.array(np.std(draws, axis=(0,1)))\
        .reshape((nParam * nParam2,))
        mat[:, 2] = np.array(np.quantile(draws, q[0], axis=(0,1)))\
        .reshape((nParam * nParam2,))
        mat[:, 3] = np.array(np.quantile(draws, q[1], axis=(0,1)))\
        .reshape((nParam * nParam2,))
        
        m = int(options.nChain * nSplit)
        n = int(options.nKeep / nSplit)
        draws_split = np.zeros((m, n, nParam, nParam2))
        draws_split[:options.nChain,:,:,:] = draws[:,:n,:,:]
        draws_split[options.nChain:,:,:,:] = draws[:,n:,:,:]
        mu_chain = np.mean(draws_split, axis=1, keepdims=True)
        mu = np.mean(mu_chain, axis=0, keepdims=True)
        B = (n / (m - 1)) * np.sum((mu_chain - mu)**2, axis=(0,1))
        ssq = (1 / (n - 1)) * np.sum((draws_split - mu_chain)**2, axis=1)
        W = np.mean(ssq, axis=0)
        varPlus = ((n - 1) / n) * W + B / n
        Rhat = np.empty((nParam, nParam2)) * np.nan
        W_idx = W > 0
        Rhat[W_idx] = np.sqrt(varPlus[W_idx] / W[W_idx])
        mat[:,4] = np.array(Rhat).reshape((nParam * nParam2,))
            
        df = pd.DataFrame(mat, columns=headers) 
        if verbose:
            print(' ')
            print(param_name + ':')
            print(df)
        return df  
    
    @staticmethod
    def _posterior_mean(options, param_name, nParam, nParam2): 
        """ Calculates mean of posterior draws of parameter of interest. """
        draws = np.zeros((options.nChain, options.nKeep, nParam, nParam2))
        for c in range(options.nChain):
            file = h5py.File(options.model_name + '_draws_chain' + str(c + 1) \
                             + '.hdf5', 'r')
            draws[c,:,:,:] = np.array(file[param_name + '_store'])\
            .reshape((options.nKeep, nParam, nParam2))
        post_mean = draws.mean(axis=(0,1))
        return post_mean
    
    def _posterior_fit(self, options, verbose):
        """ Calculates LPPD and WAIC. """
        lp_draws = np.zeros((options.nChain, options.nKeep, self.N))
        for c in range(options.nChain):
            file = h5py.File(options.model_name + '_draws_chain' + str(c + 1) \
                             + '.hdf5', 'r')
            lp_draws[c,:,:] = np.array(file['lp' + '_store'])
        
        p_draws = np.exp(lp_draws)
        lppd = np.log(p_draws.mean(axis=(0,1))).sum()
        p_waic = lp_draws.var(axis=(0,1)).sum()
        waic = -2 * (lppd - p_waic)
        
        if verbose:
            print(' ')
            print('LPPD: ' + str(lppd))
            print('WAIC: ' + str(waic))
        return lppd, waic
    
    ###
    #Estimate
    ###            
    
    def estimate(
            self,
            options, bart_options,
            r0, b0, c0,
            beta_mu0, beta_Si0Inv,
            mu_mu0, mu_Si0Inv, nu, A,
            sigma2_b0, sigma2_c0, tau_mu0, tau_si0,
            r_init, beta_init, mu_init, Sigma_init,
            ranking_top_m_list):
        """ Performs MCMC simulation for negative binomial model. 
        
        Keywords:
            options (Options): Simulation options.
            bart_options (BartOptions): Options for BART component.
            r0 (float): Hyperparameter of prior on r; r ~ Gamma(r0, h).
            b0 (float): Hyperparameter of prior on h; h ~ Gamma(b0, c0).
            c0 (float): Hyperparameter of prior on h; h ~ Gamma(b0, c0).
            beta_mu0 (array): Hyperparameter of prior on beta; 
            beta ~ N(beta_mu0, beta_Si0)
            beta_Si0Inv (array): Inverse of hyperparameter of prior on beta; 
            beta ~ N(beta_mu0, beta_Si0)
            mu_mu0 (array): Hyperparameter of prior on mu; 
            mu ~ N(mu_mu0, mu_Si0)
            mu_Si0Inv (array): Inverse of hyperparameter of prior on mu; 
            mu ~ N(mu_mu0, mu_Si0).
            nu (float) ~ Hyperparameter of prior on a; a ~ Gamma(nu, A_k)
            A (array): Hyperparameter of prior on a; a ~ Gamma(nu, A_k)
            sigma2_b0 (float): Hyperparameter of prior on sigma2; 
            sigma2 ~ Gamma(sigma2_b0, sigma2_c0).
            sigma2_c0 (float): Hyperparameter of prior on sigma2; 
            sigma2 ~ Gamma(sigma2_b0, sigma2_c0).
            tau_mu0 (float): Hyperparameter of prior on tau; 
            tau ~ N(tau_mu0, tau_si0**2).
            tau_si0 (float): Hyperparameter of prior on tau; 
            tau ~ N(tau_mu0, tau_si0**2).
            r_init (float): Initial value of r.
            beta_init (array): Initial value of beta.
            mu_init (array): Initial value of mu.
            Sigma_init (array): Initial value of Sigma.
            ranking_top_m_list (list): List of m values used for extracting 
            whether a site belongs to the top m most hazardous sites.
        """
        
        np.random.seed(options.seed)
        
        ###
        #Posterior sampling
        ###
        
        tic = time.time()
        
        """
        for c in range(options.nChain):
            self._mcmc_chain(
                c, options, bart_options,
                r0, b0, c0,
                beta_mu0, beta_Si0Inv,
                mu_mu0, mu_Si0Inv, nu, A,
                sigma2_b0, sigma2_c0, tau_mu0, tau_si0,
                r_init, beta_init, mu_init, Sigma_init,
                ranking_top_m_list) 
        """    
        
        Parallel(n_jobs = options.nChain)(delayed(self._mcmc_chain)(
                c, options, bart_options,
                r0, b0, c0,
                beta_mu0, beta_Si0Inv,
                mu_mu0, mu_Si0Inv, nu, A,
                sigma2_b0, sigma2_c0, tau_mu0, tau_si0,
                r_init, beta_init, mu_init, Sigma_init,
                ranking_top_m_list) 
        for c in range(options.nChain))
        
        toc = time.time() - tic
        
        print(' ')
        print('Estimation time [s]: ' + str(toc))
            
        ###
        #Posterior summary
        ###
        
        lppd, waic = self._posterior_fit(options, verbose=True)
        
        post_mean_lam = self._posterior_mean(options, 'lam', self.N, 1) \
            .reshape(self.N,)
            
        rmse = np.sqrt(np.mean((post_mean_lam - self.data.y)**2))
        mae = np.mean(np.abs(post_mean_lam - self.data.y))
        rmsle = np.sqrt(np.mean((np.log1p(post_mean_lam) \
                                - np.log1p(self.data.y))**2))
        print(' ')
        print('RMSE: ' + str(rmse))
        print('MAE: ' + str(mae))
        print('RMSLE: ' + str(rmsle))
        
        post_mean_ranking = self._posterior_mean(options, 'ranking', 
                                                 self.N, 1).reshape(self.N,)
        
        post_mean_ranking_top = self._posterior_mean(options, 'ranking_top', 
                                                     self.N, 
                                                     len(ranking_top_m_list))
        
        post_r = self._posterior_summary(options, 'r', 1, 1, verbose=True)
        post_mean_f = self._posterior_mean(options, 'f', self.N, 1)\
            .reshape(self.N,)
        
        if self.n_fix:
            post_beta = self._posterior_summary(options, 'beta', 
                                                self.data.n_fix, 1,
                                                verbose=True)
        else:
            post_beta = None
            
        if self.n_rnd:
            post_mu = self._posterior_summary(options, 'mu', 
                                              self.data.n_rnd, 1,
                                              verbose=True) 
            post_sigma = self._posterior_summary(options, 'sigma', 
                                                 self.data.n_rnd, 1,
                                                 verbose=True) 
            post_Sigma = self._posterior_summary(options, 'Sigma', 
                                                 self.data.n_rnd,
                                                 self.data.n_rnd,
                                                 verbose=True) 
        else:
            post_mu = None
            post_sigma = None
            post_Sigma = None
            
        if self.data_bart is not None:
            post_variable_inclusion_props \
                = self._posterior_summary(options, 'variable_inclusion_props', 
                                          self.data_bart.J, 1,
                                          verbose=False) 
        else:
            post_variable_inclusion_props = None
        
        if self.mess:
            post_sigma_mess = self._posterior_summary(options, 'sigma_mess', 
                                                      1, 1,
                                                      verbose=True)
            post_tau = self._posterior_summary(options, 'tau', 1, 1,
                                               verbose=True)
        else:
            post_sigma_mess = None
            post_tau = None
        
        ###
        #Delete draws
        ###
        
        if options.delete_draws:
            for c in range(options.nChain):
                os.remove(options.model_name + '_draws_chain' + str(c+1) \
                          + '.hdf5')          
        
        ###
        #Results
        ###
        
        results = Results(options, bart_options, toc,
                          lppd, waic,
                          post_mean_lam, 
                          rmse, mae, rmsle,
                          post_mean_ranking, 
                          post_mean_ranking_top, ranking_top_m_list,
                          post_r, post_mean_f,
                          post_beta,
                          post_mu, post_sigma, post_Sigma,
                          post_variable_inclusion_props,
                          post_sigma_mess, post_tau)
        
        return results  
    
class SyntheticData:
    """ Generates synthetic data to test MCMC method. """
    
    def __init__(self, N=200):
        self.N = N
       
    def _draw_W_matrix(self):   
        """ Creates spatial weight matrix. """
        C = np.zeros((self.N, self.N))   
        for i in np.arange(self.N-1):
            for j in np.arange(i+1, self.N):
                C[i,j] = np.random.rand() < (8 / self.N)
        C += C.T
        W = C / C.sum(axis=1, keepdims=True)   
        return W
        
    def generate(self, fixed, random, mess):
        """ Generates synthetic data. """
        
        #Fixed parameters
        if fixed:
            beta = np.array([0.25, -0.25, 0.25, -0.25])
            n_fix = beta.shape[0]
            x_fix = -3 + 6 * np.random.rand(self.N, n_fix)
            psi_fix = x_fix @ beta        
        else:
            x_fix = None
            psi_fix = 0
            
        #Random parameters
        if random:
            gamma_mu = np.array([0.25, -0.25, 0.25, -0.25])
            n_rnd = gamma_mu.shape[0]
            gamma_sd = np.sqrt(0.5 * np.abs(gamma_mu))
            gamma_corr = 0.4 * np.array(
                [[0, 1, 0, 1],
                 [1, 0, 0, 0],
                 [0, 0, 0, 1],
                 [1, 0, 1, 0]]
                ) + np.eye(n_rnd)
            gamma_cov = np.diag(gamma_sd) @ gamma_corr @ np.diag(gamma_sd)
            gamma_ch = np.linalg.cholesky(gamma_cov)
            gamma = gamma_mu.reshape(1,n_rnd) \
                + (gamma_ch @ np.random.randn(n_rnd, self.N)).T
            x_rnd = -3 + 6 * np.random.rand(self.N, n_rnd)
            x_rnd[:,0] = 1
            psi_rnd = np.sum(x_rnd * gamma, axis=1)
        else:
            x_rnd = None
            psi_rnd = 0
        
        #Spatial error
        if mess:
            tau = -0.9
            W = self._draw_W_matrix()
            S = expm(tau * W)
            sigma = 0.3
            eps = sigma * np.random.randn(self.N)
            phi = np.linalg.solve(S, eps)
        else:
            W = None
            phi = 0
        
        #Link function
        psi = psi_fix + psi_rnd + phi
          
        #Success rate
        r = 6.0
        
        #Generate synthetic observations
        p_rng = 1 / (1 + np.exp(psi))
        y = np.random.negative_binomial(r, p_rng)
        
        #Create data object
        data = Data(y, x_fix, x_rnd, W)
        
        return data

###
#If main: test
###
    
if __name__ == "__main__":
    
    import pickle
    
    ###
    #Generate data
    ###   
    
    data = SyntheticData(N=600).generate(fixed=True, 
                                         random=False, 
                                         mess=True)
    
    ###
    #Estimate model via MCMC
    ###
    
    nb_model = NegativeBinomial(data, data_bart=None)
    
    options = Options(
            model_name='fixed',
            nChain=1, nBurn=1000, nSample=1000, nThin=2, nMem=None, 
            mh_step_initial=0.1, mh_target=0.3, mh_correct=0.01, 
            mh_window=100,
            disp=100, delete_draws=False, seed=4711
            )
    bart_options = None
    
    r0 = 1e-2
    b0 = 1e-2
    c0 = 1e-2    
    beta_mu0 = np.zeros((data.n_fix,))
    beta_Si0Inv = 1e-2 * np.eye(data.n_fix)
    mu_mu0 = np.zeros((data.n_rnd,))
    mu_Si0Inv = 1e-2 * np.eye(data.n_rnd)
    nu = 2
    A = 1e3 * np.ones(data.n_rnd)
    sigma2_b0 = 1e-2 
    sigma2_c0 = 1e-2
    tau_mu0 = 0 
    tau_si0 = 2
    
    r_init = 5.0
    beta_init = np.zeros((data.n_fix,))
    mu_init = np.zeros((data.n_rnd,))
    Sigma_init = np.eye(data.n_rnd,)
    
    ranking_top_m_list = [10, 25, 50, 100]
    
    filename = 'results_fixed'
    
    if True:
        results0 = nb_model.estimate(
            options, bart_options,
            r0, b0, c0,
            beta_mu0, beta_Si0Inv,
            mu_mu0, mu_Si0Inv, nu, A,
            sigma2_b0, sigma2_c0, tau_mu0, tau_si0,
            r_init, beta_init, mu_init, Sigma_init,
            ranking_top_m_list)
        
        if os.path.exists(filename): 
            os.remove(filename) 
        outfile = open(filename, 'wb')
        pickle.dump(results0, outfile)
        outfile.close()
    else:
        infile = open(filename, 'rb')
        results0 = pickle.load(infile)
        infile.close()
    
    """
    ######
    #With BART component
    ######

    data2 = Data(np.array(data.y), 
                 x_fix=None, x_rnd=None, W=None)
    data_bart = bt.DataTrain(
        y_raw=np.array(results0.post_mean_f), 
        X=np.array(data.x_fix[:,1:]),
        weights=np.ones((data.N,)),
        offset=results0.post_beta.loc[0, 'mean']
        )
    
    options = Options(
            model_name='bart',
            nChain=1, nBurn=1000, nSample=1000, nThin=2, nMem=None, 
            disp=100, delete_draws=False, seed=4711
            )
    bart_options = bt.BartOptions(
        data_bart,
        nTrees=50, 
        k=3.0, nu=3.0, q=0.9,
        alpha=0.95, beta=2.0,
        p_grow=0.28, p_prune=0.28, p_change=0.44,
        estimate_sigma=False, heter=True
        )
    
    nb_model = NegativeBinomial(data2, data_bart) 
    
    results = nb_model.estimate(
        options, bart_options,
        r0, b0, c0,
        beta_mu0, beta_Si0Inv,
        mu_mu0, mu_Si0Inv, nu, A,
        sigma2_b0, sigma2_c0, tau_mu0, tau_si0,
        r_init, beta_init, mu_init, Sigma_init,
        ranking_top_m_list)

    ###
    #Plot ranks
    ###
    
    import matplotlib.pyplot as plt

    observed_ranking = rankdata(-y, method='min')
    observed_ranking_top_sites = [np.nonzero(observed_ranking <= m) \
                                  for m in ranking_top_m_list]
        
    fig, ax = plt.subplots()
    ax.scatter(
        #observed_ranking,
        results0.post_mean_ranking,
        results.post_mean_ranking
               )
    ax.grid()
    plt.show()
    
    fig, ax = plt.subplots()
    ax.scatter(
        #y,
        results0.post_mean_lam,
        results.post_mean_lam
               )
    ax.grid()
    plt.show()
    """
    ###
    #Posterior draws
    ###

    param_name = 'f'
    
    file = h5py.File('fixed' + '_draws_chain' + str(0+1) + '.hdf5', 'r')
    draws_fixed = np.array(file[param_name + '_store'])
    """
    file = h5py.File('bart' + '_draws_chain' + str(0+1) + '.hdf5', 'r')
    draws_bart = np.array(file[param_name + '_store']) 
    """
