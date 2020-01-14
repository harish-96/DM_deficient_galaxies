
# coding: utf-8

# In[1]:


import illustris_python as il
import numpy as np
import h5py
import time
from numba import njit

basePath = '../sims.TNG/TNG100-1/output/'
fields = ['SubhaloMassType','SubhaloParent','SubhaloHalfmassRadType','SubhaloPos', 'SubhaloLenType', 'SubhaloGrNr', 'SubhaloCM','SubhaloStarMetallicity', 'SubhaloGasMetallicity']
subhalos = il.groupcat.loadSubhalos(basePath, 99, fields=fields)

f = h5py.File('../sims.TNG/TNG100-1/output/snapdir_099/snap_099.0.hdf5')
header = dict(f['Header'].attrs)

h = header['HubbleParam']
L = header['BoxSize'] / h
dm_part_mass = header['MassTable'][1] / h


# In[2]:


dm_mass_full = subhalos['SubhaloMassType'][:,1] / h
star_mass_full = subhalos['SubhaloMassType'][:,4] / h
hf_sm_rad_full = subhalos['SubhaloHalfmassRadType'][:,4] / h
parent_prop_full = subhalos['SubhaloParent']
group_num_full = subhalos['SubhaloGrNr']
pos_full = subhalos['SubhaloPos'] / h
num_part_full = subhalos['SubhaloLenType']
gas_met = subhalos['SubhaloGasMetallicity']
star_met = subhalos['SubhaloStarMetallicity']
ID_full = np.arange(len(dm_mass_full))


# In[5]:


bools = (parent_prop_full==0) & (star_mass_full>0.001) & (num_part_full[:,4]>20)
dm_mass = dm_mass_full[bools]
hf_sm_rad = hf_sm_rad_full[bools]
group_num = group_num_full[bools]
pos = pos_full[bools]
star_mass = star_mass_full[bools]
num_part = num_part_full[bools]
gas_met = gas_met[bools]
star_met = star_met[bools]
ID = ID_full[bools]
print(pos.shape)

groups = [0]
for i in range(len(group_num)-1):
    if group_num[i] != group_num[i+1]:
        groups.append(i+1)

bools = (group_num == -1)
for i in range(len(groups)-1):
    pos_grp = pos[groups[i]:groups[i+1]]
    hst_index = np.argmax(star_mass[groups[i]:groups[i+1]])
    pos_host = pos_grp[hst_index]
    shmr = hf_sm_rad[groups[i]+hst_index]
    dist = np.minimum(np.abs(pos_grp-pos_host), L-np.abs(pos_grp-pos_host))**2
    dist = np.sum(dist, axis=1)**0.5
    bools[groups[i]:groups[i+1]] = dist > 2*shmr
    bools[groups[i]+hst_index] = True

dm_mass = dm_mass[bools]
star_mass = star_mass[bools]
hf_sm_rad = hf_sm_rad[bools]
pos = pos[bools]
group_num = group_num[bools]
num_part = num_part[bools]
gas_met = gas_met[bools]
star_met = star_met[bools]
ID = ID[bools]
print(pos.shape)

idx = (dm_mass/star_mass < 1)
dm_mass_low = dm_mass[idx]
star_mass_low = star_mass[idx]
hf_sm_rad_low = hf_sm_rad[idx]
pos_low = pos[idx]
group_num_low = group_num[idx]
num_part_low = num_part[idx]
gas_met_low = gas_met[idx]
star_met_low = star_met[idx]
ID_low = ID[idx]
print(pos_low.shape)


# In[6]:


groups = [0]
for i in range(len(group_num_low)-1):
    if group_num_low[i] != group_num_low[i+1]:
        groups.append(i+1)
print(len(groups))


# In[7]:


@njit
def subh_bool(bools, ratio, dm_pos, star_pos, star_mass, pos_low, subs, dm_part_mass, hf_sm_rad_low, h, L):
    for subh in subs:
        dist = np.sum(np.minimum(np.abs(dm_pos/h-pos_low[subh]), L - np.abs(dm_pos/h-pos_low[subh]))**2, axis=1)**0.5
        dist2 = np.sum(np.minimum(np.abs(star_pos/h-pos_low[subh]), L-np.abs(star_pos/h-pos_low[subh]))**2, axis=1)**0.5
        shmr3 = 3*hf_sm_rad_low[subh]
        r = np.sum(dm_part_mass*(dist<=shmr3))/np.sum(star_mass[dist2<=shmr3]/h)
        ratio[subh] = r
        bools[subh] = (r<1)


# In[8]:


fields_star = ['Masses','Coordinates']
fields_dm = ['Coordinates']
bools = (dm_mass_low == -100)
ratio = np.zeros_like(dm_mass_low)
for i in range(len(groups)-1):
    dm_pos = il.snapshot.loadHalo(basePath,99, group_num_low[groups[i]], 'dm', fields_dm)
    stars = il.snapshot.loadHalo(basePath,99, group_num_low[groups[i]], 'stars', fields_star)
    subh_bool(bools, ratio, dm_pos, stars['Coordinates'], stars['Masses'], pos_low, list(range(groups[i], groups[i+1])),dm_part_mass, hf_sm_rad_low, h, L)
    print(i, groups[i], np.sum(bools))


# In[16]:


with open('subhalos_lowdm.txt', 'w+') as f:
    for i in ID_low[bools]:
        f.write(str(i) + '\n')


# In[37]:


import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = [15, 13]
hf_sm2 = hf_sm_rad_low[bools]
ratio2 = ratio[bools]
plt.loglog(ratio2[ratio2!=0], hf_sm2[ratio2!=0], '.')
plt.loglog(1e-4*np.ones_like(hf_sm2[ratio2==0]), hf_sm2[ratio2==0], 'r.')
plt.xlabel(r'$\frac{M_{dm}}{M_{star}}$', fontsize=30)
plt.ylabel('Half stellar mass radius (kpc)', fontsize=20)
plt.savefig('DMdef.png')


# In[42]:


dm_mass_low2 = dm_mass_low[bools]
star_mass_low2 = star_mass_low[bools]

x = np.linspace(10**(-3), 10**2, 100)

plt.loglog(star_mass[dm_mass!=0], dm_mass[dm_mass!=0], '.')
plt.loglog(star_mass[dm_mass==0], 0.0002*np.ones_like(star_mass[dm_mass==0]), 'r.')

plt.loglog(x, 10*x + 10**(-2), '--r', label='$M_{dm}/M_{star}$ = 10')
plt.loglog(x, 1*x + 10**(-3), '--g', label='$M_{dm}/M_{star}$ = 1')
plt.loglog(x, 0.1*x + 10**(-4), '--b', label='$M_{dm}/M_{star}$ = 0.1')
plt.loglog(x, 0.01*x + 10**(-5), '--k', label='$M_{dm}/M_{star}$ = 0.01')

plt.legend(fontsize='x-large')
plt.ylabel('Stellar Mass [$M_\odot$]', fontsize=20)
plt.xlabel('Dark Matter Mass [$M_\odot$]', fontsize=20)
plt.xlim(10**(-3),10**3)
plt.ylim(10**(-4),10**5)

plt.savefig('dmstar.png')


# In[1]:


np.sum(ratio==0)


# In[2]:


plt.plot(star_mass_low[bools], star_met_low[bools])

