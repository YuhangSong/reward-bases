import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(context="talk")

def xavier (Nout,Nin):
    return np.random.rand(Nout,Nin) * np.sqrt(6/(Nout+Nin))

#Setting seed to produce the same figure as in the paper
np.random.seed(0)

#Setting network size
Nx = 5       #number of cortical neurons
Ns = 50      #number of striatal neurons
Nd = 10      #number of dopamine neurons
Nr = 2       #number of reward dimensions

#Simulation parameters
rewards = np.array([[1, 0.5, 0.1, 0, 0],
                    [0, 0, 0, 0.7, 0.05]])
alpha = 0.05   #learning rate
ITER = 100000  #number of training iterations
REP = 10       #number of repetitions of the whole experimwnt 
RECEVERY = 100 #interval between timepoints visualized in the plots

#initialization of variables used for plotting
E = np.zeros((REP*int(ITER/RECEVERY),3))
rew_con = ['Thirsty', 'Hungry']
stim_con = ['0.9ml juice','0.5ml juice','0.2ml juice','1.5g banana','0.3g banana']
Vdf = pd.DataFrame(columns=['Striatal output','Motivation','Stimulus'])

for ri in range(REP):
    
    #Initializing weights
    wCS = xavier(Ns,Nx)
    wSD = xavier (Nd,Ns)
    wDS = xavier (Ns,Nd)
    #weights from reward to dopamine initialized such that add up to 1 across neurons
    wRD = np.random.randn(Nd,Nr)
    tot = np.sum(wRD, axis=0)
    for i in range(Nr):
        wRD[:,i] = wRD[:,i] / tot[i]
    
    for it in range(ITER):
        
        #initializing network state on the beginning of a trial
        m = np.random.rand(Nr)
        stim = np.random.randint(Nx)
        x = np.zeros(Nx)
        x[stim] = 1
        r = m * rewards[:,stim]
        
        #simulating the network
        g = wDS @ wRD @ m  #tonic dopamine on striatal neurons
        s = g * (wCS @ x)     
        d = wRD @ r - wSD @ s
        wSD = wSD + alpha * np.outer (d, s) 
        wCS = wCS + alpha * np.outer (g * (wDS @ d), x)
        
        #recording error and weight alignment for plotting
        if it % RECEVERY == 0:
            E[int((ri*ITER+it)/RECEVERY),0] = it
            E[int((ri*ITER+it)/RECEVERY),1] = np.sum(d**2)
            r = np.corrcoef(wSD.flatten(), wDS.transpose().flatten())
            E[int((ri*ITER+it)/RECEVERY),2] = r[0,1]
    
    #recording outputs genenerated by the network at the end of training
    print('.')
    for rewarded in range(2):
        m = np.zeros(2)
        m[rewarded] = 1
        for stim in range(Nx):
            x = np.zeros(Nx)
            x[stim] = 1
            s = (wDS @ wRD @ m) * (wCS @ x)
            Vdf.loc[len(Vdf)] = [np.sum(wSD@s), rew_con[rewarded], stim_con[stim]]
    
Error = pd.DataFrame(E, columns=['Trial','Loss','r(wSD,wDS)'])
plt.figure(figsize=(3,3))
sns.lineplot (data=Error, x='Trial', y='Loss')
plt.yscale('log')
plt.savefig('Trial-Loss.pdf', format='pdf')

plt.figure(figsize=(3,3))
sns.lineplot (data=Error, x='Trial', y='r(wSD,wDS)')
plt.savefig('Trial-r.pdf', format='pdf')

plt.figure(figsize=(7,3))
sns.barplot (data=Vdf, x='Stimulus', y='Striatal output', hue='Motivation', errorbar='sd')
plt.savefig('Stimulus-Striatal_output.pdf', format='pdf')
            
