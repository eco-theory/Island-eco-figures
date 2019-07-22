import numpy as np
import math
import scipy.signal
import pdb
import sys
import scipy.optimize as opt

def nearAntisymRun(K,gamma,seed,time = 4):
    if K % 2 == 0:
        K = K+1

    # find fixed pt types
    AD = AntisymDynamics(K,seed=seed)
    AD.RunAntisymDynamics()
    AD.FindAntisymFixedPt()
    
    n_fp = np.reshape(K*AD.n_fp,(K,1)) # vector(K,1) with fixed pt abundances
    fp_index = AD.fp_index

    # run near antisym dynamics
    # gamma = -0.99
    M = np.inf
    AD.ChangeInteractions(gamma)
    AD.RunNearAntisymDynamics(M,time,fixedPtOnly=True)
    
    tvec = AD.tvec*np.sqrt(K)
    x_traj = AD.x_traj + np.log(K)
    n_traj = np.exp(x_traj)
    
    V = AD.V/np.sqrt(K)
    lambd = np.einsum('it,ij,jt->t',n_traj,V,n_traj)/K
    temp = -np.mean(n_fp[fp_index,:] * (x_traj[fp_index,:]-np.log(n_fp[fp_index,:])),axis=0)
    
    return tvec, lambd, temp


def nullspace(A, atol=1e-13, rtol=0):
    A = np.atleast_2d(A)
    u, s, vh = np.linalg.svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns

class AntisymDynamics:
    
    def __init__(self, K, seed=None, A = None):
        self.K = K
        if K % 2 != 1:   #K should be odd so that there is null vector.
            self.K = K+1
        
        if seed is None:
            seed = np.random.randint(10000)
        self.seed = seed
        np.random.seed(self.seed)
        
        if A is None:  
            A = generate_interactions_with_diag(self.K,-1)
        self.A = A
        
    def RunAntisymDynamics(self,x0=None,dt = None,total_time=None,sample_time = None):
        K = self.K
        A = self.A
        
        
        if x0 is None:
            N = 1
            x0 = generate_init_values(K,N,'equal')
        else:
            N = np.sum(np.exp(x0))
        deriv = define_antisym_deriv(A)
        step_forward = step_rk4
        
        if dt is None:
            dt = 0.5*np.sqrt(K)
        if total_time is None:
            total_time = 50*K
        if sample_time is None:
            sample_time = dt
        
        tvec, x_traj = simple_dynamics(x0,N,dt,total_time,sample_time,deriv,step_forward)
        self.tvec = tvec
        self.x_traj = x_traj
        self.x_end = x_traj[:,-1]
        
    def FindAntisymFixedPt(self,x0 = None):
        if x0 is None:
            self.loop_counter = 1
            
        self.RunAntisymDynamics(x0 = x0)
        
        A = self.A
        K = self.K
        x_end = self.x_end
        
        thresh = -50
        fp_index = x_end > thresh
        if np.sum(fp_index) % 2 == 0:
            # find closest to x=-50 and include if x<-50 and exclude if x>-50
            close_ind= np.argmin(np.abs(x_end-thresh))
            x_close = x_end[close_ind]
            if x_close>thresh:
                fp_index = x_end>x_close
            else:
                fp_index = x_end>=x_close
                 
        ext_index = np.logical_not(fp_index)

        A_ss = A[fp_index,:][:,fp_index]
        A_es = A[ext_index,:][:,fp_index]

        n_s_fp = nullspace(A_ss)
        
        if n_s_fp.size > 0:
            n_s_fp = np.reshape(n_s_fp,(len(n_s_fp)))/np.sum(n_s_fp)
            ext_bias = A_es @ n_s_fp
            
            if np.all(n_s_fp>0) and np.all(ext_bias<0):
                n_fp = np.zeros((K))
                n_fp[fp_index] = n_s_fp
                
                self.n_s = n_s_fp
                self.n_fp = n_fp
                self.A_ss = A_ss
                self.fp_index = fp_index
                self.ext_index = ext_index
                
                print('\r Saturated fixed pt found in {} loops'.format(self.loop_counter))
                
                self.FindAntisymBias()
            else:
                self.loop_counter += 1
                print('\r Fixed pt not saturated: {} loops'.format(self.loop_counter),end='')
                sys.stdout.flush()
                
                self.FindAntisymFixedPt(x0 = x_end)
                      
        else:
            print('No null vector found')
        
    def FindAntisymBias(self):
        K = self.K
        V = self.A/np.sqrt(K)
        n_fp = self.n_fp*K
        n_s = self.n_s*K
        ext_bias = (V @ n_fp)[self.ext_index]
        
        #Estimate chi to ensure self consistency ( E[bias**2] = E[n_fp**2] )
        chi = np.sqrt(1-np.sum(ext_bias**2)/np.sum(n_s**2))
        surv_bias = n_s*chi
        
        bias = np.zeros((K))
        bias[self.fp_index] = surv_bias
        bias[self.ext_index] = ext_bias
        
        self.bias = bias

    def ChangeInteractions(self,gamma):
        # perturbs antisymmetric interactions to get matrix with order one entries and symmetry parameter = gamma.
        
        self.gamma = gamma
        np.random.seed(2*self.seed)
        
        B = generate_interactions_with_diag(self.K,1)
        a = np.sqrt((1-gamma)/2)
        b = np.sqrt((1+gamma)/2)
        self.V = a*self.A + b*B
        
    def RunNearAntisymDynamics(self, M,time,sample_time=1,fixedPtOnly = False):
        
        K = self.K
        self.M = M
        
        N = 1
        u = 0
        nbar = 1/K
        normed = True
        m = np.exp(-M)
        
        deriv = define_deriv_mainland(self.V,N,u,m,nbar,normed)
        step_forward = step_rk4
        
        if fixedPtOnly == False:
            x0 = generate_init_values(K,N,'equal')
        else: 
            x0 = generate_init_values(K,N,'equal')
            x0[self.ext_index] =  -np.inf
            x0 = x0 + np.log(N/np.sum(np.exp(x0)))

        dt = .1
        total_time = time*K**(3/2)
        sample_time = 1
        
        tvec, x_traj = simple_dynamics(x0,N,dt,total_time,sample_time,deriv,step_forward)
        self.tvec = tvec
        self.x_traj = x_traj


class IslandSimWithExtinction:
    # Class object to run simulation and keep abundance trajectories. 

    def __init__(self,file_name,D,K,M,gamma,thresh,time,dt,sample_time,seed,n0 = None, V = None,K0 = None):
        
        self.file_name = file_name
        self.D = D
        self.K = K
        self.M = M
        self.gamma = gamma
        self.thresh = thresh   # Equal to -log(N). Should be greater than M. log(Nm) = log(N) - M
        self.time = time  # in units of K0**(3/2)
        self.dt = dt
        self.sample_time = sample_time
        self.seed = seed
        self.n0 = n0
        self.V = V

        if K0 == None:  #K0 used for normalizations
            self.K0 = K
        else:
            self.K0 = K0

        self.set_params()

        self.run_island_sim()


    def set_params(self):

        D = self.D
        K = self.K
        K0 = self.K0
        dt = self.dt
        n0 = self.n0
        V = self.V

        self.N = 1
        self.m = np.exp(-self.M)
    
        t0 = K0**(3/2)
        self.total_time = self.time*t0
        self.total_steps = int(round(self.total_time/dt))


        # self.sample_time = 1
        self.start_time_short = 0
        self.sample_num_short, self.sample_start_short, self.sample_end_short, self.tvec_short = sample_calculations(dt,self.sample_time, self.start_time_short,self.total_time)
        self.tvec_short_len = len(self.tvec_short)


        np.random.seed(seed=self.seed)


        if n0 is None:    
            self.n0 = islands_initialization(D,K,self.N,self.m,'flat')
        else:
            nhat = n0
            Nhat = np.sum(nhat,axis=1,keepdims=1)
            self.n0 = nhat/Nhat


        if V is None:
            self.V = generate_interactions_with_diag(K,self.gamma)


    def run_island_sim(self):

        file_name = self.file_name
        K = self.K
        K0 = self.K0
        D = self.D
        dt = self.dt
        m = self.m
        N = self.N
        n0 = self.n0
        thresh = self.thresh
        M = self.M
        V = self.V
        Normalize = self.Normalize
        Extinction = self.Extinction
        
        sample_start_short = self.sample_start_short
        sample_num_short = self.sample_num_short

        u = 0
        deriv = define_deriv_islands(V,N,u,m)
        step_forward = step_runge_kutta4

        nbar = np.mean(n0,axis=0,keepdims=True)
        ext_ind = nbar==0
        nbar[ext_ind] = 1

        # Define abundances as n = y*np.exp(-xbar)
        xbar0 = np.log(nbar)
        y0 = n0/nbar

        xbar0[ext_ind] = -np.inf

        
        n_traj = np.zeros((self.tvec_short_len,D,K))

        for step in range(self.total_steps):

            # Record abundances every sample_time when step > sample_start_short
            if step >= sample_start_short and (step-sample_start_short) % sample_num_short == 0:
                ind = int((step-sample_start_short)//sample_num_short)
                n0 = K0 * y0*np.exp(xbar0)  # Normalize so that sum of abundances is K0.
                n_traj[ind,:,:] = n0


            # Step abundances forward
            y1, xbar1 = step_forward(y0,xbar0,m,dt,deriv)
            y1, xbar1 = Extinction(y1,xbar1,thresh,M)
            y1, xbar1 = Normalize(y1,xbar1,N)


            # Prep for next time step
            y0 = y1
            xbar0 = xbar1

        # Save data
        self.n_traj = n_traj
        self.tvec = self.tvec_short/K0**(3/2)

        np.savez(file_name, IslandSim = vars(self))


    def Normalize(self,y,xbar,N):
        n = np.exp(xbar)*y
        Nhat = np.sum(n,axis=1,keepdims=True)
        Yhat = np.mean(y*N/Nhat,axis=0,keepdims=True)

        Yhat[Yhat==0] = 1

        y1 = (y*N/Nhat)/Yhat
        xbar1 = xbar + np.log(Yhat)

        return y1, xbar1

    def Extinction(self,y,xbar,thresh,M):
        
        local_ext_ind = xbar+np.log(y)<thresh
        y[local_ext_ind] = 0

        global_ext_ind = np.all(y==0,axis=0)
        xbar[:,global_ext_ind] = -np.inf
        y[:,global_ext_ind] = 0
        
        return y, xbar


#######################################################

def sample_calculations(dt,sample_time,sample_time_start,sample_time_end):

    sample_num = int(round(sample_time/dt))
    sample_start = int(round(sample_time_start/dt))
    sample_end = int(round(sample_time_end/dt))
    sample_tvec = dt*np.arange(sample_start,sample_end,sample_num)

    return sample_num, sample_start, sample_end, sample_tvec

def islands_initialization(D,K,N,m,mode):
    #:input K: number of species
    #:input N: normalization of abundances
    #:input D: number of demes
    #:input mode: type of initialization
    #:output x: drawn from normal distribution with mean N/K and var 
    
    if mode=='equal':
        nbar_temp=N/K
        nhat = np.random.exponential(nbar_temp,(D,K))
        Nhat = np.sum(nhat,axis=1,keepdims=1)
        n = N*nhat/Nhat
    elif mode=='flat':
        nbar = N/K
        random_vars = np.random.rand(D,K)
        xhat = (np.log(N) - np.log(m*nbar))*random_vars + np.log(m*nbar)
        nhat = np.exp(xhat)
        Nhat = np.sum(nhat,axis=1,keepdims=1)
        n = nhat/Nhat

    return n

def generate_interactions_with_diag(K,gamma):
    # :param K: number of types
    # :param gamma: correlation parameter. Need gamma in [-1,1]
    # :output V: interaction matrix. Entries have correlation E[V_{i,j} V_{m,n}] = \delta_{i,m}\delta_{j,n} + \gamma \delta_{i,n} \delta_{j,m}

    V = np.zeros((K,K))
    upperInd = np.triu_indices(K,1)
    
    if gamma < 1 and gamma > -1:
        #generate list of correlated pairs.
        mean = [0,0]
        cov = [[1, gamma], [gamma, 1]]
        x = np.random.multivariate_normal(mean, cov, (K*(K-1)//2))

        #put correlated pairs into matrix
        V[upperInd] = x[:,0]
        V = np.transpose(V)
        V[upperInd] = x[:,1]

        diag = np.random.normal(scale=np.sqrt(1+gamma),size=K)
        V = V + np.diag(diag)
    
    elif gamma == 1 or gamma == -1:
        x = np.random.normal(0,1,size=(K*(K-1)//2))

        V[np.triu_indices(K,1)] = x
        V = np.transpose(V)
        V[np.triu_indices(K,1)] = gamma*x

        diag = np.random.normal(scale=np.sqrt(1+gamma),size=K)
        V = V + np.diag(diag)

    else:
        print('Correlation parameter invalid')
    
    return V




def define_deriv_islands(V,N,u,m):
    # :input V: interaction matrix
    # :input u: self interaction strength
    # :input m: migration rate
    # :input xbar: (D,K) log of nbar abundance
    # :input y: (D,K) vector describing abundances of the form n = y np.exp(xbar)
    # :output deriv: function that computes derivative

    def deriv(y,xbar,m):
        n = np.exp(xbar)*y
        D = np.shape(n)[0]
        growth_rate = -u*(n) + np.einsum('dj,ij',n,V)
        
        norm_factor = np.einsum('di,di->d',n,growth_rate)/N  #normalization factor
        norm_factor = np.reshape(norm_factor,(D,1))

        y_dot0 = y*(growth_rate-norm_factor)
        
        if m==0:
            y_dot = y_dot0 
        else:
            y_dot = y_dot0  + m*(1-y)

        return y_dot

    return deriv



def step_runge_kutta4(y0,xbar0,m,dt,deriv):
    # :input x0: (K,) vec of current log variables
    # :param dt: step size
    # :function deriv: gives derivative of x. Function of x.
    # :output x1: (K,) vec of log variables after one time step.
    

    k1 = deriv(y0,xbar0,m)
    k2 = deriv(y0+(dt/2)*k1,xbar0,m)
    k3 = deriv(y0+(dt/2)*k2,xbar0,m)
    k4 = deriv(y0+dt*k3,xbar0,m)
    
    y1 = y0 + (dt/6)*(k1 + 2*k2 + 2*k3+k4)
    return y1, xbar0


def antisym_generate_init_equilibrium(q,T,K0):
    #input q: (K,) vec of q values. 
    #input T: temperature. 
    #output x0: (K,) vec of initial values v = log(n)

    K = len(q)
    N = np.sum(q)

    #draw x from a beta distribution with parameters a and b
    a = q/T
    b = (N-q)/T

    x0 = np.zeros(K)
    n_rand = np.random.beta(a,b)
    n_rand = n_rand*N/np.sum(n_rand)
    zero_index = n_rand==0
    nonzero_index = n_rand>0
    num_zeros = np.sum(zero_index)
    x0[zero_index] = -np.random.rand(num_zeros)/a[zero_index]
    x0[nonzero_index] = np.log(n_rand[nonzero_index])

    x0 = x0 + np.log(N/np.sum(np.exp(x0)))

    return x0

def define_antisym_deriv(V):
    # :input V: interaction matrix
    # :input u: self interaction strength
    # :input m: migration rate
    # :input nbar: mainland abundance
    # :input normed: logical bit for whether to normalize derivative or not
    # :output deriv: function of x that computes derivative

    def deriv(x):

        xdot = np.dot(V,np.exp(x))
        return xdot

    return deriv

def generate_init_values(K,N,method,logrange=0):
    # :param K: number of types
    # :param N: normalization, \sum n = N 
    # :input method: string that defines method of generating initial values
    # :output x: (K) vec of initial values for x = log(n)

    # 'equal' method: x = log(N/K)
    if method is 'equal':
        x = np.log(N/K)*np.ones(K)

    if method is 'flat':
        x = -np.random.rand(K)*logrange
        x = normalize(x,N)

    return x

#simulation dynamics    
def simple_dynamics(x0,N,dt,total_time,sample_time,deriv,step_forward):
    # :input x0: (K,) vector of initial log variables x = log(n)
    # :param dt: step size.
    # :param total_time: total time for dynamics. 
    # :param sample_time: time between samplings for output
    # :function deriv: gives derivative of x
    # :function step_forward: returns x for next timestep

    # :return time_vec: vec of times
    # :return x_traj: trajectories of x

    K = np.shape(x0)[0]

    sample_num = int(round(sample_time/dt))
    sample_time = dt*sample_num
    time_vec = np.arange(total_time,step=sample_time)
    L = len(time_vec)
    x_traj = np.zeros((K,L))

    x0 = normalize(x0,N)
    
    for ii in range(L):
        x_traj[:,ii] = x0
        
        for jj in range(sample_num):

            x1 = step_forward(x0,dt,deriv)
            x1 = normalize(x1,N)
            x0 = x1

        # if ii%int(1e4/sample_time) == 0:
            # print('t = '+str(time_vec[ii]))

    return time_vec, x_traj

def normalize(x,N):
    # :input x: (K,) vec of current log variables
    # :param N: normalization, \sum exp(x) = N 
    # :output xnorm: correctly normalized x values.

    y = x - np.max(x)
    xnorm = y + np.log(N) - np.log(np.sum(np.exp(y)))
    return xnorm

def step_rk4(x0,dt,deriv):
    # :input x0: (K,) vec of current log variables
    # :param dt: step size
    # :function deriv: gives derivative of x. Function of x.
    # :output x1: (K,) vec of log variables after one time step.
    

    k1 = deriv(x0)
    k2 = deriv(x0+(dt/2)*k1)
    k3 = deriv(x0+(dt/2)*k2)
    k4 = deriv(x0+dt*k3)
    
    x1 = x0 + (dt/6)*(k1 + 2*k2 + 2*k3+k4)
    return x1

def define_deriv_mainland(V,N,u,m,nbar,normed):
    # :input V: interaction matrix
    # :input u: self interaction strength
    # :input m: migration rate
    # :input nbar: mainland abundance
    # :input normed: logical bit for whether to normalize derivative or not
    # :output deriv: function of x that computes derivative

    def deriv(x):

        n = np.exp(x)
        growth_rate = -u*n + np.dot(V,n)
        if normed is True:
            norm_factor = np.dot(n,growth_rate)/N  #normalization factor
        else:
            norm_factor = 0
        if m==0:
            xdot = growth_rate - norm_factor
        else:
            not_extinct_bool = n>0
            xdot = np.zeros(np.shape(n))
            xdot[not_extinct_bool] = growth_rate[not_extinct_bool] - norm_factor + m*(nbar/n[not_extinct_bool] - 1)
        return xdot

    return deriv


def lv_to_lp(A):
    """
    Generate parameters for lp to find fixed points
    :param A: antisymmetric matrix
    :return c, A_ub, b_ub, A_eq, b_eq: linear program parameters for linprog from scipy.optimize.
    """
    N = np.shape(A)[0]
    # augment value v of "game" to end of variables. Trying to minimize v.
    c = np.zeros(N + 1)
    c[-1] = 1

    # inequality constraints

    A_ub = np.zeros((N, N + 1))
    b_ub = np.zeros(N)

    # max fitness bounded by v

    A_ub[:, :-1] = A
    A_ub[:, -1] = -1

    # equality constraint: probabilities sum to 1
    A_eq = np.ones((1, N + 1))
    A_eq[0, -1] = 0
    b_eq = np.ones(1)

    return c, A_ub, b_ub, A_eq, b_eq

def find_fixed_pt(A):
    """
    Find saturated fixed point of antisymmetric LV model using linear programming
    :param A: Interaction matrix
    :return: saturated fixed point
    """
    N = np.shape(A)[0]
    maxiter = int(np.max(((N**2)/25,1000)))
    c, A_ub, b_ub, A_eq, b_eq = lv_to_lp(A)
    result = opt.linprog(c,A_ub,b_ub,A_eq,b_eq,options={'maxiter':maxiter},method='interior-point')
    if isinstance(result.x, float):
        print(result)
        print(result.message)
    return result.x[:-1]

def count_alive(fs,min_freq=None):
    """
    Count number of types with non-zero abundance
    :param fs: array of frequencies
    :param min_freq: cutoff frequency for measuring alive types
    :return: Number of types with non-zero abundance
    """
    if min_freq==None:
        min_freq = 1.0/len(fs)**2
    return np.sum([f>min_freq for f in fs])

