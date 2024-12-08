from pde import FieldCollection, PDEBase, PlotTracker, ScalarField, UnitGrid,CallbackTracker
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

class SIRPDE(PDEBase):
    """SEIR model with diffusion and migration"""

    def __init__(
        self, beta1=0.1,beta2=0.2,beta3=0.15, sigma1=0.2,sigma2=0.3,sigma3=0.4,
        delta1=0.05,delta2=0.1,delta3=0.15,diffusivity1=0.1,diffusivity2=0.3,diffusivity3=0.4,
        vs=0.1,ve1=0.2,ve2=0.25,ve3=0.3,vi1=0.22,vi2=0.27,vi3=0.32,
        bc="auto_periodic_neumann"
    ):
        super().__init__()
        self.beta1 = beta1  # transmission rate
        self.beta2 = beta2
        self.beta3 = beta3
        self.sigma1 = sigma1 # incubation period conversion rate
        self.sigma2 = sigma2
        self.sigma3 = sigma3
        self.delta1=delta1  # fatality rate
        self.delta2=delta2
        self.delta3=delta3
        self.diffusivity1 = diffusivity1  # spatial mobility
        self.diffusivity2 = diffusivity2
        self.diffusivity3 = diffusivity3
        self.vs=vs
        self.ve1=ve1
        self.ve2=ve2
        self.ve3=ve3
        self.vi1=vi1
        self.vi2=vi2
        self.vi3=vi3
        
        self.bc = bc  # boundary condition

    def get_state(self, s, e1,e2,e3,i1,i2,i3):
        """Generate a suitable initial state"""
        norm = (s + e1+e2+e3+i1+i2+i3).data.max()  # maximal density
        if norm > 1:
            s.data[0,0] /= norm  
            e1.data[0,0] /= norm
            e2.data[0,0] /= norm
            e3.data[0,0] /= norm
            i1.data[0,0] /= norm
            i2.data[0,0] /= norm
            i3.data[0,0] /= norm
        s.label = "S"  
        e1.label = "E1" 
        e2.label = "E2" 
        e3.label = "E3" 
        i1.label = "I1"  
        i2.label = "I2"  
        i3.label = "I3"  

        # create R field
        r = ScalarField(s.grid, data=1 - s - e1-e2-e3-i1-i2-i3, label="R")
        return FieldCollection([s,e1,e2,e3, i1,i2,i3, r])  

    def evolution_rate(self, state, t=0):
        """定义状态场的演化速率。"""
        s,e1,e2,e3, i1,i2,i3, r= state  
        diff1 = self.diffusivity1  
        diff2 = self.diffusivity2
        diff3 = self.diffusivity3 
        gradx1,grady1=np.gradient(self.vs*s.data)
        gradx2,grady2=np.gradient(self.ve1*e1.data)
        gradx3,grady3=np.gradient(self.ve2*e2.data)
        gradx4,grady4=np.gradient(self.ve3*e3.data)
        gradx5,grady5=np.gradient(self.vi1*i1.data)
        gradx6,grady6=np.gradient(self.vi2*i2.data)
        gradx7,grady7=np.gradient(self.vi3*i3.data)
        ds_dt =-(gradx1+grady1)+diff1 * s.laplace(self.bc) - (self.beta1 * i1+self.beta2 * i2+self.beta3 * i3 )* s  
        de1_dt =-(gradx2+grady2)+ diff2 * e1.laplace(self.bc) + self.beta1 * i1* s -self.sigma1*e1 
        de2_dt =-(gradx3+grady3)+ diff2 * e2.laplace(self.bc) + self.beta2 * i2* s -self.sigma2*e2 
        de3_dt =-(gradx4+grady4)+ diff2 * e3.laplace(self.bc) + self.beta3 * i3* s -self.sigma3*e3 
        di1_dt =-(gradx5+grady5)+ diff3 * i1.laplace(self.bc) + self.sigma1 * e1  - self.delta1 * i1  
        di2_dt =-(gradx6+grady6)+ diff3 * i2.laplace(self.bc) + self.sigma2 * e2  - self.delta2 * i2  
        di3_dt =-(gradx7+grady7)+ diff3 * i3.laplace(self.bc) + self.sigma3 * e3  - self.delta3 * i3  
        dr_dt = self.delta1 * i1 +self.delta2 * i2+self.delta3 * i3 
        
        return FieldCollection([ds_dt, de1_dt, de2_dt,de3_dt,di1_dt,di2_dt,di3_dt,dr_dt])  # 返回演化速率场集合

eq = SIRPDE()

# initialize state
grid = UnitGrid([64, 64])  
s = ScalarField(grid, 1)  
e1=ScalarField(grid, 0)   
e2=ScalarField(grid, 0)   
e3=ScalarField(grid, 0)   
i1 = ScalarField(grid, 0)  
i2 = ScalarField(grid, 0)  
i3 = ScalarField(grid, 0)  

e1.data[0, 0] = 0.05
e2.data[0, 0] = 0.04
e3.data[0, 0] = 0.03
i1.data[0, 0] = 0.03  
i2.data[0, 0] = 0.02
i3.data[0, 0] = 0.01


state = eq.get_state(s, e1,e2,e3,i1,i2,i3)  
# simulate the pde
tracker = PlotTracker(interrupts=40, plot_args={"vmin": 0, "vmax": 1, "cmap": cm.Blues})  
sol = eq.solve(state, t_range=120, dt=0.01, tracker=["progress", tracker])  

ss_final=sol[0] 
ee1_final=sol[1] 
ee2_final=sol[2] 
ee3_final=sol[3] 
ii1_final=sol[4] 
ii2_final=sol[5] 
ii3_final=sol[6] 
rr_final=sol[7]  
# visualization
data = np.array(ss_final.data)
x = np.linspace(0, len(ss_final.data[0])-1, data.shape[1])  
y = np.linspace(0, len(ss_final.data[0])-1, data.shape[0])  
final_data = [ss_final, ee1_final, ee2_final, ee3_final, ii1_final, ii2_final, ii3_final, rr_final]
titles = ['S', 'E1', 'E2', 'E3', 'I1', 'I2', 'I3', 'R']
plt.figure(figsize=(18, 8))
for i, data in enumerate(final_data, 1):
    plt.subplot(2, 4, i)
    if titles[i-1] in ['S', 'R']:  
        cax = plt.imshow(np.array(data.data), extent=[x.min(), x.max(), y.min(), y.max()],
                         origin='lower', cmap='Blues', vmin=0, vmax=1)
    else:  
        cax = plt.imshow(np.array(data.data), extent=[x.min(), x.max(), y.min(), y.max()],
                         origin='lower', cmap='Blues')

    plt.title(titles[i-1])
    plt.xlabel('x')
    plt.ylabel('y')
    
    if i == 4 or i == 8:  
       plt.colorbar(cax, label="Fraction of Population")
    else:
       plt.colorbar(cax)
plt.suptitle('Time=80-120', fontsize=16)
plt.show()

#%%
eq = SIRPDE()
state = eq.get_state(s, e1,e2,e3,i1,i2,i3)  
results = []

def save_callback(state, t):
    results.append((t, state.copy()))

tracker = CallbackTracker(save_callback)
sol = eq.solve(state, t_range=120, dt=0.01, tracker=["progress", tracker])

s_final=[]
e1_final=[]
e2_final=[]
e3_final=[]
i1_final=[]
i2_final=[] 
i3_final=[]
r_final=[]
for i in range(121):
    s_final.append(results[i][1][0].data[0,0])
    e1_final.append(results[i][1][1].data[0,0])
    e2_final.append(results[i][1][2].data[0,0])
    e3_final.append(results[i][1][3].data[0,0])
    i1_final.append(results[i][1][4].data[0,0])
    i2_final.append(results[i][1][5].data[0,0])
    i3_final.append(results[i][1][6].data[0,0])
    r_final.append(results[i][1][7].data[0,0])

e_final = [e1 + e2 + e3 for e1, e2, e3 in zip(e1_final, e2_final, e3_final)]
i_final = [i1 + i2 + i3 for i1, i2, i3 in zip(i1_final, i2_final, i3_final)]

plt.figure(figsize=(12, 4))
plt.subplot(1,2,1)
plt.plot(s_final,label='S')
plt.plot(r_final,label='R')
plt.plot(e1_final,label='E1')
plt.plot(e2_final,label='E2')
plt.plot(e3_final,label='E3')
plt.plot(e_final,label='E')
plt.plot(i1_final,label='I1')
plt.plot(i2_final,label='I2')
plt.plot(i3_final,label='I3')
plt.plot(i_final,label='I')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5)) 
plt.xlabel('Time')
plt.ylabel('Fraction of Population')
plt.tight_layout()  
plt.subplot(1,2,2)
plt.plot(e1_final,label='E1')
plt.plot(e2_final,label='E2')
plt.plot(e3_final,label='E3')
plt.plot(e_final,label='E')
plt.plot(i1_final,label='I1')
plt.plot(i2_final,label='I2')
plt.plot(i3_final,label='I3')
plt.plot(i_final,label='I')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5)) 
plt.xlabel('Time')
plt.ylabel('Fraction of Population')
plt.tight_layout()  
plt.show()

#%%
# alter the transmission rate
eq = SIRPDE(beta1=0.07,beta2=0.17,beta3=0.12, sigma1=0.2,sigma2=0.3,sigma3=0.4,
delta1=0.05,delta2=0.1,delta3=0.15)
state = eq.get_state(s, e1,e2,e3,i1,i2,i3)  
results = []
tracker = CallbackTracker(save_callback)
sol = eq.solve(state, t_range=120, dt=0.01, tracker=["progress", tracker])

s1_final=[]
e11_final=[]
e12_final=[]
e13_final=[]
i11_final=[]
i12_final=[] 
i13_final=[]
r1_final=[]
for i in range(121):
    s1_final.append(results[i][1][0].data[0,0])
    e11_final.append(results[i][1][1].data[0,0])
    e12_final.append(results[i][1][2].data[0,0])
    e13_final.append(results[i][1][3].data[0,0])
    i11_final.append(results[i][1][4].data[0,0])
    i12_final.append(results[i][1][5].data[0,0])
    i13_final.append(results[i][1][6].data[0,0])
    r1_final.append(results[i][1][7].data[0,0])

ee_final = [e1 + e2 + e3 for e1, e2, e3 in zip(e11_final, e12_final, e13_final)]
ii_final = [i1 + i2 + i3 for i1, i2, i3 in zip(i11_final, i12_final, i13_final)]

plt.figure(figsize=(16, 4))
plt.subplot(1,3,1)
plt.plot(s_final,label='S')
plt.plot(r_final,label='R')
plt.plot(s1_final,label='Modified S')
plt.plot(r1_final,label='Modified R')
plt.legend() 
plt.xlabel('Time')
plt.ylabel('Fraction of Population')
plt.tight_layout()  
plt.subplot(1,3,2)
plt.plot(e1_final,label='E1')
plt.plot(e2_final,label='E2')
plt.plot(e3_final,label='E3')
plt.plot(e_final,label='E')
plt.plot(e11_final,label='Modified E1')
plt.plot(e12_final,label='Modified E2')
plt.plot(e13_final,label='Modified E3')
plt.plot(ee_final,label='Modified E')
plt.legend() 
plt.xlabel('Time')
plt.ylabel('Fraction of Population')
plt.tight_layout()  
plt.subplot(1,3,3)
plt.plot(i1_final,label='I1')
plt.plot(i2_final,label='I2')
plt.plot(i3_final,label='I3')
plt.plot(i_final,label='I')
plt.plot(i11_final,label='Modified I1')
plt.plot(i12_final,label='Modified I2')
plt.plot(i13_final,label='Modified I3')
plt.plot(ii_final,label='Modified I')
plt.legend() 
plt.xlabel('Time')
plt.ylabel('Fraction of Population')
plt.tight_layout()  
plt.show()

#%%
# alter the incubation period conversion rate
eq = SIRPDE(beta1=0.1,beta2=0.2,beta3=0.15, sigma1=0.25,sigma2=0.35,sigma3=0.45,
delta1=0.05,delta2=0.1,delta3=0.15)
state = eq.get_state(s, e1,e2,e3,i1,i2,i3)  
results = []
tracker = CallbackTracker(save_callback)
sol = eq.solve(state, t_range=120, dt=0.01, tracker=["progress", tracker])

s1_final=[]
e11_final=[]
e12_final=[]
e13_final=[]
i11_final=[]
i12_final=[] 
i13_final=[]
r1_final=[]
for i in range(121):
    s1_final.append(results[i][1][0].data[0,0])
    e11_final.append(results[i][1][1].data[0,0])
    e12_final.append(results[i][1][2].data[0,0])
    e13_final.append(results[i][1][3].data[0,0])
    i11_final.append(results[i][1][4].data[0,0])
    i12_final.append(results[i][1][5].data[0,0])
    i13_final.append(results[i][1][6].data[0,0])
    r1_final.append(results[i][1][7].data[0,0])

ee_final = [e1 + e2 + e3 for e1, e2, e3 in zip(e11_final, e12_final, e13_final)]
ii_final = [i1 + i2 + i3 for i1, i2, i3 in zip(i11_final, i12_final, i13_final)]

plt.figure(figsize=(16, 4))
plt.subplot(1,3,1)
plt.plot(s_final,label='S')
plt.plot(r_final,label='R')
plt.plot(s1_final,label='Modified S')
plt.plot(r1_final,label='Modified R')
plt.legend() 
plt.xlabel('Time')
plt.ylabel('Fraction of Population')
plt.tight_layout()  
plt.subplot(1,3,2)
plt.plot(e1_final,label='E1')
plt.plot(e2_final,label='E2')
plt.plot(e3_final,label='E3')
plt.plot(e_final,label='E')
plt.plot(e11_final,label='Modified E1')
plt.plot(e12_final,label='Modified E2')
plt.plot(e13_final,label='Modified E3')
plt.plot(ee_final,label='Modified E')
plt.legend() 
plt.xlabel('Time')
plt.ylabel('Fraction of Population')
plt.tight_layout()  
plt.subplot(1,3,3)
plt.plot(i1_final,label='I1')
plt.plot(i2_final,label='I2')
plt.plot(i3_final,label='I3')
plt.plot(i_final,label='I')
plt.plot(i11_final,label='Modified I1')
plt.plot(i12_final,label='Modified I2')
plt.plot(i13_final,label='Modified I3')
plt.plot(ii_final,label='Modified I')
plt.legend() 
plt.xlabel('Time')
plt.ylabel('Fraction of Population')
plt.tight_layout()  
plt.show()