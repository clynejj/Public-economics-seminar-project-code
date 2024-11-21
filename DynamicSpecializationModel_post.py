from types import SimpleNamespace

import numpy as np
from scipy.optimize import minimize

from EconModel import EconModelClass
from consav.grids import nonlinspace
from consav.linear_interp_2d import interp_2d

import matplotlib.pyplot as plt
plt.rcParams.update({"axes.grid":True,"grid.color":"black","grid.alpha":"0.25","grid.linestyle":"--"})
plt.rcParams.update({'font.size': 12})

# global gender identifiers
woman = 1
man = 2

class DynamicSpecializationModelClass(EconModelClass):
    #########
    # Setup #
    def settings(self):
        """ fundamental settings """
        pass

    def setup(self):
        """ setup model """

        # a. unpack
        par = self.par 

        par.T = 8

        # b. preferences
        par.beta = 0.98
        par.rho = 2.0
        par.nu_f_l = 0.001
        par.nu_f_h = 0.001
        par.nu_m_l = 0.001
        par.nu_m_h = 0.001
        par.epsilon_f_l = 1.0
        par.epsilon_m_l = 1.0
        par.epsilon_f_h = 1.0
        par.epsilon_m_h = 1.0
        par.epsilon_f_l_kids = -0.1
        par.epsilon_m_l_kids = -0.1
        par.epsilon_f_h_kids = -0.1
        par.epsilon_m_h_kids = -0.1
        par.omega = 0.5 
        par.omega_n = -0.2
        par.power = 0.5 # power level of women

        # c. household production
        par.alpha = 0.5
        par.alpha_n = 0.0

        par.sigma = 0.3 
        par.sigma_n = 0.0

        # d. wages and income
        par.wage_const_f = 3.0
        par.wage_humcap_f = 0.1
        par.wage_const_m = 3.0
        par.wage_humcap_m = 0.1

        par.X = 0.0 # unearned income child transfer

        par.delta = 0.1 # human capital depreciation

        # fertility
        par.prob_birth = 0.1
        par.num_n = 2

        # e. grids
        par.num_K = 10
        par.max_K = 15.0

        # f. simulation
        par.simT = par.T
        par.simN = 10000

    def allocate(self):
        # a. unpack
        par = self.par 
        sol = self.sol 
        sim = self.sim

        # b. setup grids 
        par.grid_K = nonlinspace(0.0,par.max_K,par.num_K,1.1)
        par.grid_n = np.arange(par.num_n)

        # c. solution
        shape = (par.T,par.num_n,par.num_K,par.num_K)
        sol.V = np.nan + np.zeros(shape)
        sol.labor_f = np.nan + np.zeros(shape)
        sol.home_f = np.nan + np.zeros(shape)
        sol.labor_m = np.nan + np.zeros(shape)
        sol.home_m = np.nan + np.zeros(shape)

        # d. simulation
        shape = (par.simN,par.simT)
        sim.labor_f = np.nan + np.zeros(shape)
        sim.home_f = np.nan + np.zeros(shape)
        sim.labor_m = np.nan + np.zeros(shape)
        sim.home_m = np.nan + np.zeros(shape)

        sim.wage_f = np.nan + np.zeros(shape)
        sim.wage_m = np.nan + np.zeros(shape)

        sim.kf = np.nan + np.zeros(shape)
        sim.km = np.nan + np.zeros(shape)
        
        sim.birth = np.zeros(shape,dtype=np.int_)
        sim.n = np.zeros(shape,dtype=np.int_)
        sim.draws_uniform = np.random.uniform(0,1,size=shape)

        # initial values
        np.random.seed(2023)
        sim.init_kf = np.random.uniform(0,3,size=par.simN) 
        sim.init_km = np.random.uniform(0,3,size=par.simN) 
        sim.init_n = np.zeros(par.simN,dtype=np.int_)
    
    ############
    # Solution #
    def solve(self):

        # a. unpack
        par = self.par
        sol = self.sol

        # b. loop backwards (over all periods)
        for t in reversed(range(par.T)):
            if t==par.T-1:
                V_next = None
            else:
                V_next = sol.V[t+1]

            # c. loop over state variables: presence of child and human capital for each household member
            for i_n,kids in enumerate(par.grid_n):
                for i_kf,capital_f in enumerate(par.grid_K):
                    for i_km,capital_m in enumerate(par.grid_K):
                        idx = (t,i_n,i_kf,i_km)
                        idx_last = (t+1,i_n,i_kf,i_km)
                        
                        # d. find optimal time-allocation at this level of human capital.
                        obj = lambda x: self.value_of_choice(x,capital_f,capital_m,kids,V_next)  

                        bounds = [(0,24) for i in range(4)]

                        # e. initial values: use solution from future, if possible
                        if t<par.T-1:
                            init = np.array([getattr(sol,name)[idx_last] for name in ('labor_f','home_f','labor_m','home_m')])
                            
                        else:
                            if i_kf>0 & i_km>0:
                                init = res.x
                            else:
                                init = np.ones(4)

                        # f. estimate and store results
                        res = minimize(obj,init,bounds=bounds,method='L-BFGS-B') 

                        sol.labor_f[idx] = res.x[0]
                        sol.home_f[idx] = res.x[1]
                        sol.labor_m[idx] = res.x[2]
                        sol.home_m[idx] = res.x[3]
                        sol.V[idx] = -res.fun
    
    def clip_and_penalty(self,labor_f,home_f,labor_m,home_m):

        # penalty for bounds
        penalty = 0.0
        if labor_f>24.0:
            penalty += (labor_f-24)*1000.0
            labor_f = 24.0
        if home_f>24.0:
            penalty += (home_f-24)*1000.0
            home_f = 24.0
        if labor_m>24.0:
            penalty += (labor_m-24)*1000.0
            labor_m = 24.0
        if home_m>24.0:
            penalty += (home_m-24)*1000.0
            home_m = 24.0

        # penalty for inequality constraint
        if (labor_f+home_f)>24:
            penalty += (labor_f+home_f-24)*1000.0
            labor_f = 24 - home_f
        if (labor_m+home_m)>24:
            penalty += (labor_m+home_m-24)*1000.0
            labor_m = 24 - home_m

        return labor_f,home_f,labor_m,home_m,penalty

    def value_of_choice(self,x,capital_f,capital_m,kids,V_next=None):

        # a. unpack
        par = self.par
        labor_f,home_f,labor_m,home_m = x

        # b. penalty for bounds and inequality constraint
        labor_f,home_f,labor_m,home_m,penalty = self.clip_and_penalty(labor_f,home_f,labor_m,home_m)

        # c. current utility
        utility = self.util(labor_f,home_f,labor_m,home_m,capital_f,capital_m,kids)
        
        # c. Expected continuation value
        if V_next is not None:
            kf_next = self.human_capital_next(capital_f,labor_f)
            km_next = self.human_capital_next(capital_m,labor_m)

            # birth-specific values
            V_interp_no_birth = interp_2d(par.grid_K,par.grid_K,V_next[kids],kf_next,km_next)
            if kids<(par.num_n-1): # can have kids next period
                V_interp_birth = interp_2d(par.grid_K,par.grid_K,V_next[kids+1],kf_next,km_next)
            else:
                V_interp_birth = V_interp_no_birth # this ensures that I can always weight with the associated probabilities below

            EV_interp = par.prob_birth*V_interp_birth + (1.0-par.prob_birth)*V_interp_no_birth
                
        else: # last period of life
            EV_interp = 0.0

        # d. return negative value of choice (plus penalty)
        return -(utility + par.beta*EV_interp) + penalty

    def util(self,labor_f,home_f,labor_m,home_m,capital_f,capital_m,kids):
        # a. unpack
        par = self.par

        # b. labor income and consumption expenditures
        wage_f = self.wage_func(capital_f,woman)
        wage_m = self.wage_func(capital_m,man)

        C = wage_f*labor_f + wage_m*labor_m + self.child_transfer(kids)

        # c. home production
        alpha = par.alpha + par.alpha_n*kids
        sigma = par.sigma + par.sigma_n*kids
        F_term = alpha*np.fmax(home_f,1e-8)**((sigma-1)/sigma)
        M_term = (1-alpha)*np.fmax(home_m,1e-8)**((sigma-1)/sigma)
        H = np.fmax(M_term + F_term,1e-8)**(sigma/(sigma-1))

        # d. composite good
        omega = par.omega + par.omega_n*kids
        Q = (C**omega)  * (H**(1.0-omega))

        # e. utility of consumption
        util_Q = np.fmax(Q,1e-8)**(1.0-par.rho) / (1.0-par.rho)

        # f. dis-utility from work
        T_f = labor_f + home_f
        T_m = labor_m + home_m
        power_f_l = 1+1/(par.epsilon_f_l + par.epsilon_f_l_kids*kids)
        power_m_l = 1+1/(par.epsilon_m_l + par.epsilon_m_l_kids*kids)
        power_f_h = 1+1/(par.epsilon_f_h + par.epsilon_f_h_kids*kids)
        power_m_h = 1+1/(par.epsilon_m_h + par.epsilon_m_h_kids*kids)

        util_f_T = par.nu_f_l* (labor_f**power_f_l / power_f_l) + par.nu_f_h*(home_f**power_f_h / power_f_h)
        util_m_T = par.nu_m_l* (labor_m**power_m_l / power_m_l) + par.nu_m_h*(home_m**power_m_h / power_m_h)

        util_T = par.power * util_f_T + (1-par.power) * util_m_T
        
        # g. return total utility
        return util_Q - util_T

    def wage_func(self,capital,gender):
        # a. unpack
        par = self.par

        # b. determine relevant parameters
        const = par.wage_const_f
        humcap = par.wage_humcap_f
        if gender==man:
            const = par.wage_const_m
            humcap = par.wage_humcap_m

        # c. return wage rate
        return np.exp(const + humcap*capital)

    def human_capital_next(self,capital,labor):
        return (1.0-self.par.delta)*capital + labor/24.0
    
    def child_transfer(self,kids):
        if kids ==1:
            return self.par.X
        else:
            return 0.0
    
    ##############
    # Simulation #
    def simulate(self):

        # a. unpack
        par = self.par
        sol = self.sol
        sim = self.sim

        # b. loop over individuals and time
        for i in range(par.simN):

            # c. initialize states
            sim.kf[i,0] = sim.init_kf[i]
            sim.km[i,0] = sim.init_km[i]
            sim.n[i,0] = sim.init_n[i]

            for t in range(par.simT):

                # d. interpolate optimal allocation
                idx_sol = (t,sim.n[i,t])
                sim.labor_f[i,t] = interp_2d(par.grid_K,par.grid_K,sol.labor_f[idx_sol],sim.kf[i,t],sim.km[i,t])
                sim.home_f[i,t] = interp_2d(par.grid_K,par.grid_K,sol.home_f[idx_sol],sim.kf[i,t],sim.km[i,t])
                sim.labor_m[i,t] = interp_2d(par.grid_K,par.grid_K,sol.labor_m[idx_sol],sim.kf[i,t],sim.km[i,t])
                sim.home_m[i,t] = interp_2d(par.grid_K,par.grid_K,sol.home_m[idx_sol],sim.kf[i,t],sim.km[i,t])

                # e. store wage
                sim.wage_f[i,t] = self.wage_func(sim.kf[i,t],woman)
                sim.wage_m[i,t] = self.wage_func(sim.km[i,t],man)

                # f. store next-period states
                if t<par.simT-1:
                    sim.kf[i,t+1] = self.human_capital_next(sim.kf[i,t],sim.labor_f[i,t])
                    sim.km[i,t+1] = self.human_capital_next(sim.km[i,t],sim.labor_m[i,t])

                    # childbirth
                    sim.birth[i,t+1] = 0
                    if (sim.draws_uniform[i,t+1]<par.prob_birth) & (sim.n[i,t]<(par.num_n-1)):
                        sim.birth[i,t+1] = 1
                    sim.n[i,t+1] = sim.n[i,t] + sim.birth[i,t+1]
        
        # g. specialization index
        sim.specialization = sim.home_f/(sim.home_f+sim.home_m)-sim.labor_f/(sim.labor_f+sim.labor_m)

    ##############                
    # Estimation #                     
    def regress(self):
        """ run regression """
        # a. unpack
        sim = self.sim

        # b. run regression
        if (np.min(sim.wage_f)>0.0) & (np.min(sim.wage_m)>0.0) & (np.min(sim.home_f)>0.0) & (np.min(sim.home_m)>0.0):
            x = np.log(sim.wage_f/sim.wage_m).ravel()
            y = np.log(sim.home_f/sim.home_m).ravel()
            A = np.vstack([np.ones(x.size),x]).T
            constant,slope = np.linalg.lstsq(A,y,rcond=None)[0]
        else:
            constant = slope = np.nan
        
        # c. return constant and slope parameters
        return constant,slope
    
    def regress_female(self):
        """ run regression. First we must transfer daily hours to weekly hours """
        # a. unpack
        sim = self.sim

        # transfer daily hours to weekly hours
        weekly_sim_home_f = sim.home_f * 7

        # b. run regression
        if (np.min(sim.wage_f)>0.0) & (np.min(sim.wage_m)>0.0) & (np.min(sim.home_f)>0.0) & (np.min(sim.home_m)>0.0):
            x = np.log(sim.wage_f/sim.wage_m).ravel()
            y = np.log(weekly_sim_home_f).ravel()
            A = np.vstack([np.ones(x.size),x]).T
            constant,slope = np.linalg.lstsq(A,y,rcond=None)[0]
        else:
            constant = slope = np.nan
        
        # c. return constant and slope parameters
        return constant,slope
    
    def regress_male(self):
        """ run regression. First we must transfer daily hours to weekly hours """
        # a. unpack
        sim = self.sim

        # transfer daily hours to weekly hours
        weekly_sim_home_m = sim.home_m * 7

        # b. run regression
        if (np.min(sim.wage_f)>0.0) & (np.min(sim.wage_m)>0.0) & (np.min(sim.home_f)>0.0) & (np.min(sim.home_m)>0.0):
            x = np.log(sim.wage_f/sim.wage_m).ravel()
            y = np.log(weekly_sim_home_m).ravel()
            A = np.vstack([np.ones(x.size),x]).T
            constant,slope = np.linalg.lstsq(A,y,rcond=None)[0]
        else:
            constant = slope = np.nan
        
        # c. return constant and slope parameters
        return constant,slope

    def plot_old(self,add_regression=True):
        # a. unpack
        sim = self.sim

        # b. log relative wage of women
        rel_wage = np.log(sim.wage_f/sim.wage_m)
        rel_wage_grid = np.linspace(np.min(rel_wage.ravel()),np.max(rel_wage.ravel()),2)

        # c. log relative home production 
        rel_home = np.log(sim.home_f/sim.home_m)

        # d. plot relationship 
        fig, ax = plt.subplots()
        #ax.scatter(rel_wage,rel_home,label='simulated scatter')
        ax.plot(rel_wage_grid,0.4-0.1*rel_wage_grid,label='empirical target',color='orange',linewidth=2)
        if add_regression:
            constant,slope = self.regress()
            ax.plot(rel_wage_grid,constant+slope*rel_wage_grid,label='simulated relationship',color='red',linewidth=2)

        ax.set(title='Female domestic work share',xlabel='$log(w_f/w_m)$',ylabel='$log(h_f/h_m)$')
        ax.legend()

    def plot(self, add_regression=True):
        # a. unpack
        sim = self.sim

        # b. log relative wage of women
        rel_wage = np.log(sim.wage_f/sim.wage_m)
        
        # Extend rel_wage_grid beyond data range to include intercepts
        rel_wage_grid = np.linspace(np.min(rel_wage.ravel()) - 0.5, np.max(rel_wage.ravel()) + 0.5, 100)

        # c. log relative home production 
        rel_home = np.log(sim.home_f/sim.home_m)

        # d. plot relationship 
        fig, ax = plt.subplots()
        ax.plot(rel_wage_grid, 0.4 - 0.1 * rel_wage_grid, label='Data', color='orange', linewidth=2)
        if add_regression:
            constant, slope = self.regress()
            ax.plot(rel_wage_grid, constant + slope * rel_wage_grid, label='Model', color='blue', linewidth=2)

        ax.set(title='Female domestic work share', xlabel='$log(w_f/w_m)$', ylabel='$log(h_f/h_m)$')

        # Set limits for x and y axes to include intercepts properly
        ax.set_xlim([np.min(rel_wage.ravel()) - 0.5, np.max(rel_wage.ravel()) + 0.5])
        ax.set_ylim([0, 0.8])  # You can adjust this as needed based on intercept location

        ax.legend()
        plt.show()



    def plot_female_hours(self,add_regression=True):
        # a. unpack
        sim = self.sim

        # b. log relative wage of women
        rel_wage = np.log(sim.wage_f/sim.wage_m)
        rel_wage_grid = np.linspace(np.min(rel_wage.ravel()),np.max(rel_wage.ravel()),2)

        # c. log female home production
        log_home_f = np.log(sim.home_f*7)

        # d. plot relationship 
        fig, ax = plt.subplots()
        #ax.scatter(rel_wage,log_home_f,label='simulated scatter')
        ax.plot(rel_wage_grid,3.255-0.094*rel_wage_grid,label='data',color='orange',linewidth=2)
        if add_regression:
            constant,slope = self.regress_female()
            ax.plot(rel_wage_grid,constant+slope*rel_wage_grid,label='model',color='blue',linewidth=2)

        # define labels
        ax.set(title='Female domestic work hours (per week)',xlabel='$log(w_f/w_m)$',ylabel='$log(h_f)$')

        # Set x-axis limits to reduce zoom and show a broader range of values
        #ax.set_ylim(2.5,3.5)  # Adjust this range as needed to achieve the desired zoom level
        ax.legend()

    def plot_male_hours(self,add_regression=True):
        # a. unpack
        sim = self.sim

        # b. log relative wage of women
        rel_wage = np.log(sim.wage_f/sim.wage_m)
        rel_wage_grid = np.linspace(np.min(rel_wage.ravel()),np.max(rel_wage.ravel()),2)

        # c. log female home production
        log_home_m = np.log(sim.home_m*7)

        # d. plot relationship 
        fig, ax = plt.subplots()
        #ax.scatter(rel_wage,log_home_m,label='simulated scatter')
        ax.plot(rel_wage_grid,2.811-0.003*rel_wage_grid,label='data',color='orange',linewidth=2)
        if add_regression:
            constant,slope = self.regress_male()
            ax.plot(rel_wage_grid,constant+slope*rel_wage_grid,label='model',color='blue',linewidth=2)

        ax.set(title='Male domestic work hours',xlabel='$log(w_f/w_m)$',ylabel='$log(h_m)$')
        ax.legend()