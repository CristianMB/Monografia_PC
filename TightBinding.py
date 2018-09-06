import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la


# Hamiltonian 

t = 0.8               #Hopping and Site Potnential values
U = 1.0

# Distribution Parameters

# N - Normal Distribution 
sigma = 0.5
mu = 0.5
# L - Log Normal Distribution
sigma_l = 0.5
mu_l = 0.5 


def H(MODE, N):            
    H=np.zeros(shape=(N,N))		     
    tn = 0
    Un = 0
    
    if(MODE=="CC"):     #Molecular potential and hopping are constant                       
        tn = t                       
        Un = U                    					
        for i in range(N-1):
            H[i,i+1] = -1.0*tn
            H[i+1,i] = -1.0*tn
            H[i,i] = Un + 2.0*tn
            H[N-1,N-1] = Un + 2.0*tn                
        return H

# Uniform Distribution

    if(MODE=="CR_U"):    #Molecular potential is random (Uniform) but hopping is constant
        tn = t
        Un= np.random.random(N)     
        for i in range(N-1):    
            H[i,i+1] = -1.0*tn
            H[i+1,i] = -1.0*tn
            H[i,i] = Un[i] + 2.0*tn
            H[N-1,N-1] = Un[N-1] + 2.0*tn
        return H
    if(MODE=="RrC_U"):    #Molecular potential is constant but all hopping terms are random (Uniform), H is not Hermitian
        Un= U                    
        for i in range(N-1):        
            H[i,i+1] = -1.0*np.random.random()
            H[i+1,i] = -1.0*np.random.random()
            H[i,i] = Un + 2.0*np.random.random()
            H[N-1,N-1] = Un + 2.0*np.random.random()
        return H
    if(MODE=="RoC_U"):    #Molecular potential is constant. All hopping terms are random (Uniform) [H is Hermitian]
        Un= U                    
        for i in range(N-1):        
            tn=np.random.random()
            H[i,i+1] = -1.0*tn
            H[i+1,i] = -1.0*tn
            H[i,i] = Un + 2.0*tn
            H[N-1,N-1] = Un + 2.0*tn
        return H
    if(MODE=="RrR_U"):     #Both, Molecular potential and all hopping terms are random (Uniform) [H is not Hermitian]
        Un= np.random.random(N)         
        for i in range(N-1):
            H[i,i+1] = -1.0*np.random.random()
            H[i+1,i] = -1.0*np.random.random()
            H[i,i] = Un[i] + 2.0*np.random.random()
            H[N-1,N-1] = Un[N-1] + 2.0*np.random.random()
        return H
    if(MODE=="RoR_U"):     #Both, Molecular potential and all hopping terms are random (Uniform) [H is Hermitian]
        Un= np.random.random(N)         
        for i in range(N-1):
            tn = np.random.random()
            H[i,i+1] = -1.0*tn
            H[i+1,i] = -1.0*tn
            H[i,i] = Un[i] + 2.0*tn
            H[N-1,N-1] = Un[N-1] + 2.0*tn
        return H

# Normal Distribution

    if(MODE=="CR_N"):    #Molecular potential is random (Normal) but hopping is constant
        tn = t
        Un= np.random.normal(sigma, mu, N)     
        for i in range(N-1):    
            H[i,i+1] = -1.0*tn
            H[i+1,i] = -1.0*tn
            H[i,i] = Un[i] + 2.0*tn
            H[N-1,N-1] = Un[N-1] + 2.0*tn
        return H
    if(MODE=="RrC_N"):    #Molecular potential is constant but all hopping terms are random (Normal), H is not Hermitian
        Un= U                    
        for i in range(N-1):        
            H[i,i+1] = -1.0*np.random.normal(sigma, mu)
            H[i+1,i] = -1.0*np.random.normal(sigma, mu)
            H[i,i] = Un + 2.0*np.random.normal(sigma, mu)
            H[N-1,N-1] = Un + 2.0*np.random.normal(sigma, mu)
        return H
    if(MODE=="RoC_N"):    #Molecular potential is constant. All hopping terms are random (Normal) [H is Hermitian]
        Un= U                    
        for i in range(N-1):        
            tn=np.random.normal(sigma, mu)
            H[i,i+1] = -1.0*tn
            H[i+1,i] = -1.0*tn
            H[i,i] = Un + 2.0*tn
            H[N-1,N-1] = Un + 2.0*tn
        return H
    if(MODE=="RrR_N"):     #Both, Molecular potential and all hopping terms are random (Normal) [H is not Hermitian]
        Un= np.random.normal(sigma, mu, N)      
        for i in range(N-1):
            H[i,i+1] = -1.0*np.random.normal(sigma, mu)
            H[i+1,i] = -1.0*np.random.normal(sigma, mu)
            H[i,i] = Un[i] + 2.0*np.random.normal(sigma, mu)
            H[N-1,N-1] = Un[N-1] + 2.0*np.random.normal(sigma, mu)
        return H
    if(MODE=="RoR_N"):     #Both, Molecular potential and all hopping terms are random (Normal) [H is Hermitian]
        Un= np.random.normal(sigma, mu, N)         
        for i in range(N-1):
            tn = np.random.normal(sigma, mu)
            H[i,i+1] = -1.0*tn
            H[i+1,i] = -1.0*tn
            H[i,i] = Un[i] + 2.0*tn
            H[N-1,N-1] = Un[N-1] + 2.0*tn
        return H
    
# Log-Normal Distribution

    if(MODE=="CR_L"):    #Molecular potential is random (LogNormal) but hopping is constant
        tn = t
        Un= np.random.lognormal(sigma_l, mu_l, N)     
        for i in range(N-1):    
            H[i,i+1] = -1.0*tn
            H[i+1,i] = -1.0*tn
            H[i,i] = Un[i] + 2.0*tn
            H[N-1,N-1] = Un[N-1] + 2.0*tn
        return H
    if(MODE=="RrC_L"):    #Molecular potential is constant but all hopping terms are random (LogNormal), H is not Hermitian
        Un= U                    
        for i in range(N-1):        
            H[i,i+1] = -1.0*np.random.lognormal(sigma_l, mu_l)
            H[i+1,i] = -1.0*np.random.lognormal(sigma_l, mu_l)
            H[i,i] = Un + 2.0*np.random.lognormal(sigma_l, mu_l)
            H[N-1,N-1] = Un + 2.0*np.random.lognormal(sigma, mu)
        return H
    if(MODE=="RoC_L"):    #Molecular potential is constant. All hopping terms are random (LogNormal) [H is Hermitian]
        Un= U                    
        for i in range(N-1):        
            tn=np.random.lognormal(sigma_l, mu_l)
            H[i,i+1] = -1.0*tn
            H[i+1,i] = -1.0*tn
            H[i,i] = Un + 2.0*tn
            H[N-1,N-1] = Un + 2.0*tn
        return H
    if(MODE=="RrR_L"):     #Both, Molecular potential and all hopping terms are random (LogNormal) [H is not Hermitian]
        Un= np.random.lognormal(sigma_l, mu_l, N)      
        for i in range(N-1):
            H[i,i+1] = -1.0*np.random.lognormal(sigma_l, mu_l)
            H[i+1,i] = -1.0*np.random.lognormal(sigma_l, mu_l)
            H[i,i] = Un[i] + 2.0*np.random.lognormal(sigma_l, mu_l)
            H[N-1,N-1] = Un[N-1] + 2.0*np.random.lognormal(sigma_l, mu_l)
        return H
    if(MODE=="RoR_L"):     #Both, Molecular potential and all hopping terms are random (LogNormal) [H is Hermitian]
        Un= np.random.lognormal(sigma_l, mu_l, N)   
        for i in range(N-1):
            tn = np.random.lognormal(sigma_l, mu_l)
            H[i,i+1] = -1.0*tn
            H[i+1,i] = -1.0*tn
            H[i,i] = Un[i] + 2.0*tn
            H[N-1,N-1] = Un[N-1] + 2.0*tn
        return H
# Exponential Distribution

    if(MODE=="CR_E"):    #Molecular potential is random (Exp) but hopping is constant
        tn = t
        Un= np.random.exponential(1, N)     
        for i in range(N-1):    
            H[i,i+1] = -1.0*tn
            H[i+1,i] = -1.0*tn
            H[i,i] = Un[i] + 2.0*tn
            H[N-1,N-1] = Un[N-1] + 2.0*tn
        return H
    if(MODE=="RrC_E"):    #Molecular potential is constant but all hopping terms are random (Exp), H is not Hermitian
        Un= U                    
        for i in range(N-1):        
            H[i,i+1] = -1.0*np.random.exponential()
            H[i+1,i] = -1.0*np.random.exponential()
            H[i,i] = Un + 2.0*np.random.exponential()
            H[N-1,N-1] = Un + 2.0*np.random.exponential()
        return H
    if(MODE=="RoC_E"):    #Molecular potential is constant. All hopping terms are random (Exp) [H is Hermitian]
        Un= U                    
        for i in range(N-1):        
            tn=np.random.exponential()
            H[i,i+1] = -1.0*tn
            H[i+1,i] = -1.0*tn
            H[i,i] = Un + 2.0*tn
            H[N-1,N-1] = Un + 2.0*tn
        return H
    if(MODE=="RrR_E"):     #Both, Molecular potential and all hopping terms are random (Exp) [H is not Hermitian]
        Un= np.random.exponential(1, N)      
        for i in range(N-1):
            H[i,i+1] = -1.0*np.random.exponential()
            H[i+1,i] = -1.0*np.random.exponential()
            H[i,i] = Un[i] + 2.0*np.random.exponential()
            H[N-1,N-1] = Un[N-1] + 2.0*np.random.exponential()
        return H
    if(MODE=="RoR_E"):     #Both, Molecular potential and all hopping terms are random (Exp) [H is Hermitian]
        Un= np.random.exponential(1, N)   
        for i in range(N-1):
            tn = np.random.exponential()
            H[i,i+1] = -1.0*tn
            H[i+1,i] = -1.0*tn
            H[i,i] = Un[i] + 2.0*tn
            H[N-1,N-1] = Un[N-1] + 2.0*tn
        return H

# EigenValues Calculation

def Eigen(M):
    val, vec = la.eig(M)
    idx = np.argsort(val)
    val = val[idx]
    vec = (vec[:,idx]).T    
    return(val, vec) 
    
# Molecular Potentials Calculated from Hamiltonian 
    
def U_t(M):
    U = []
    for i in range(len(M)):
        U.append(M[i,i]+ 2.0*M[0,1])
    return (-M[0,1], U)

# Labels for plotting
    
def Label(MODE, N, Iter):
    L = 0
    
    #Uniform Labels
    
    if (MODE=="CC"):
        L = "t=" + str(t) + ", "+ "Uo=" + str(U) + ", "+ "L=" + str(N)
    if (MODE=="CR_U"):
        L = "t=" + str(t) + ", "+ "Uo= Rand.(Uniform)" + ", "+ "L=" + str(N)+" ," +"Iter.=" +str(Iter)
    if (MODE=="RrC_U"):
        L = "t= Rand.(Uniform)" + ", "+ "Uo=" + str(U) + ", "+ "L=" + str(N) +", " + "Iter.=" +str(Iter)
    if (MODE=="RoC_U"):
        L = "t= Rand.(Uniform)" + ", "+ "Uo=" + str(U) + ", "+ "L=" + str(N)+", " + "Iter.=" +str(Iter)
    if (MODE=="RrR_U"):
        L = "t, Uo = Rand.(Uniform)" + ", "+ "L=" + str(N)+", " + "Iter.=" +str(Iter)
    if (MODE=="RoR_U"):
        L = "t, Uo = Rand.(Uniform)" + ", "+ "L=" + str(N)+ ", " + "Iter.=" +str(Iter)
        
    #Normal Labels    
        
    if (MODE=="CR_N"):
        L = "t=" + str(t) + ", "+ "Uo= Rand.(Normal_mu,sigma=" + str(mu) +","+str(sigma)+ "), "+ "L=" + str(N)+" ," +"Iter.=" +str(Iter)
    if (MODE=="RrC_N"):
        L = "t = Rand.(Normal_mu,sigma=" + str(mu) +","+str(sigma) + ")" + ", "+ "Uo=" + str(U) + ", "+ "L=" + str(N) +", " + "Iter.=" +str(Iter)
    if (MODE=="RoC_N"):
        L = "t = Rand.(Normal_mu,sigma=" + str(mu) +","+str(sigma)+ ")" + ", "+ "Uo=" + str(U) + ", "+ "L=" + str(N)+", " + "Iter.=" +str(Iter)
    if (MODE=="RrR_N"):
        L = "t, Uo = Rand.(Normal_mu,sigma=" + str(mu) +","+str(sigma)+ ")" + ", "+ "L=" + str(N)+", " + "Iter.=" +str(Iter)
    if (MODE=="RoR_N"):
        L = "t, Uo = Rand.(Normal_mu,sigma=" + str(mu) +","+str(sigma)+ ")" + ", "+ "L=" + str(N)+", " + "Iter.=" +str(Iter)

    #Log-Normal Labels    
        
    if (MODE=="CR_L"):
        L = "t= " + str(t) + ", "+ "Uo= Rand.(LogNormal_mu,sigma= " + str(mu_l) +","+str(sigma_l)+ "), "+ "L=" + str(N)+" ," +"Iter.=" +str(Iter)
    if (MODE=="RrC_L"):
        L = "t= Rand.(LogNormal_mu,sigma= " + str(mu_l) +","+str(sigma_l) + ")" + ", "+ "Uo=" + str(U) + ", "+ "L=" + str(N) +", " + "Iter.=" +str(Iter)
    if (MODE=="RoC_L"):
        L = "t= Rand.(LogNormal_mu,sigma= " + str(mu_l) +","+str(sigma_l)+ ")" + ", "+ "Uo=" + str(U) + ", "+ "L=" + str(N)+", " + "Iter.=" +str(Iter)
    if (MODE=="RrR_L"):
        L = "t, Uo= Rand.(LogNormal_mu,sigma= " + str(mu_l) +","+str(sigma_l)+ ")" + ", "+ "L=" + str(N)+", " + "Iter.=" +str(Iter)
    if (MODE=="RoR_L"):
        L = "t, Uo= Rand.(LogNormal_mu,sigma= " + str(mu_l) +","+str(sigma_l)+  ")" + ", "+ "L=" + str(N)+", " + "Iter.=" +str(Iter)

    #Exponential Labels    
        
    if (MODE=="CR_E"):
        L = "t= " + str(t) + ", "+ "Uo= Rand.(Exponential),"+ "L=" + str(N)+" ," +"Iter.=" +str(Iter)
    if (MODE=="RrC_E"):
        L = "t= Rand.(Exponential)" + ", "+ "Uo=" + str(U) + ", "+ "L=" + str(N) +", " + "Iter.=" +str(Iter)
    if (MODE=="RoC_E"):
        L = "t= Rand.(Exponential)" +  ", "+ "Uo=" + str(U) + ", "+ "L=" + str(N)+", " + "Iter.=" +str(Iter)
    if (MODE=="RrR_E"):
        L = "t, Uo= Rand.(Exponential)" + ", "+ "L=" + str(N)+", " + "Iter.=" + str(Iter)
    if (MODE=="RoR_E"):
        L = "t, Uo= Rand.(Exponential), L=" + str(N) +", " + "Iter.=" + str(Iter)

    return L




# Plots of DoS and Energy Spectrum

def Plot(MODE, N, It):
    k = np.linspace(0, np.pi, N)
    E_E0 = []
    N_e = []
    
    if (MODE == "CC"):
        M = H(MODE, N)
        En = Eigen(M)[0]
        E0 = U + 2.0*t
        E_E0 = En - E0
        
        for i in (E_E0):
            N_e.append(1.0/(4*np.pi*t*np.sqrt(1.0-((i)/(2*t))**2)))    
        
        
        fig = plt.figure(figsize = (9, 6))
        fig.subplots_adjust(hspace=0.5)
        ax = fig.add_subplot(111)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)        
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')


        Diff = np.zeros(k.shape,np.float)
        Diff[0:-1] = np.diff(k)/np.diff(E_E0)
        Diff[-1] = (k[-1] - k[-2])/(E_E0[-1] - E_E0[-2])
        Diff = Diff/(2*np.pi)
        
          
        ax1.set_title("ENERGY SPECTRUM", fontweight='bold', fontsize=10)
        ax1.scatter(k, En, color = "darkred", label = "Numerical: " + Label(MODE, N, It), s=10, marker = 'x')
        ax1.plot(k, (U + 2*t*(1.0-np.cos(k))),linewidth = 1 ,color = "blue", label = "Analytical: Uo + 2t(1-cos(ka)), a=1")
        ax1.legend(loc = 2)
        ax1.set_xlabel("k",fontweight = "bold")
        ax1.set_ylabel("Energy",fontweight = "bold")      

        ax2.set_title("DENSITY OF STATES", fontweight='bold', fontsize=10)
        ax2.plot(E_E0, N_e, label = "Analytical: " + Label(MODE, N, It)+ ", Eo=" + str(E0) + " (Eo = Uo+2t)", linewidth = 1, color = "darkred")
        ax2.scatter(E_E0, Diff, s=10, color="darkblue", marker="*", label = "Numerical")        
        ax2.legend(loc = 9)
        ax2.set_xlabel("E-E0",fontweight = "bold")
        ax2.set_ylabel("N(E)",fontweight = "bold")

        plt.savefig("EDOS(" + str(MODE) + ")"+ str(N) +"( t="+str(t)+", "+"U="+str(U)+ ").png", dpi=300)
         
    else:
        Mean = []
        E_bank = []
        V_bank = []
        for i in range(It):
            Matrix = H(MODE, N)
            E_bank.append(Eigen(Matrix)[0])
            V_bank.append(Eigen(Matrix)[1])
    
        Mean = np.mean(E_bank, axis=0)        
        
        fig = plt.figure(figsize = (9, 6))
        fig.subplots_adjust(hspace=0.5)
        ax = fig.add_subplot(111)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)        
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
                
        
        Diff = np.zeros(k.shape,np.float)
        Diff[0:-1] = np.diff(k)/np.diff(Mean)
        Diff[-1] = (k[-1] - k[-2])/(Mean[-1] - Mean[-2])
        Diff = Diff/(2*np.pi)
        
        
        ax1.set_title("ENERGY SPECTRUM", fontweight='bold', fontsize=10)
        for i in E_bank:
            ax1.plot(k, i, linewidth = 0.5)
        ax1.plot(k, Mean, color="black", linewidth = 2, label = "Average ("+Label(MODE, N, It)+")")
        ax1.legend(loc = 2)
        ax1.set_xlabel("k",fontweight = "bold")
        ax1.set_ylabel("Energy",fontweight = "bold")
        
        ax2.set_title("DENSITY OF STATES", fontweight = "bold", fontsize=10)
        ax2.plot(Mean, Diff, linewidth=0.6, color="black", label = Label(MODE, N, It))
        ax2.legend(loc = "best")
        ax2.set_xlabel("E",fontweight = "bold")
        ax2.set_ylabel("N(E)",fontweight = "bold")

        plt.savefig("EDOS(" + str(MODE) + ")"+ str(N) +"_"+str(It)+ ".png", dpi=300) 



print("Hopping = " + str(t))
print("Potencial = " + str(U))

l = 10
n = 100

M = ["CC", "CR_U", "CR_N", "CR_E", "CR_L", "RrC_U", "RrC_N", "RrC_L","RrC_E","RoC_U", "RoC_N", "RoC_L","RoC_E", "RrR_U", "RrR_N", "RrR_L","RrR_E","RoR_U", "RoR_N", "RoR_L", "RoR_E"]
for m in M: 
    Plot(m, l, n)

    