
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('pylab', 'inline')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from matplotlib.colors import LogNorm
import illustris_python as il
import numpy as np
import h5py
from numba import njit
import pathlib

basePath = '../sims.TNG/TNG100-1/output/'
fields = ['SubhaloMassType','SubhaloParent','SubhaloHalfmassRadType','SubhaloPos', 'SubhaloLenType', 'SubhaloGrNr', 'SubhaloCM','SubhaloStarMetallicity', 'SubhaloGasMetallicity']
subhalos = il.groupcat.loadSubhalos(basePath, 99, fields=fields)

f = h5py.File('../sims.TNG/TNG100-1/output/snapdir_099/snap_099.0.hdf5')
header = dict(f['Header'].attrs)

h = header['HubbleParam']
L = header['BoxSize'] / h
dm_part_mass = header['MassTable'][1] / h


# In[10]:


rcParams['figure.figsize']=(14, 12)
rcParams['lines.linewidth']=2
rcParams['axes.titlesize']=40
rcParams['axes.labelsize']=25
rcParams['xtick.labelsize']=15
rcParams['ytick.labelsize']=15
rcParams['legend.fontsize']=25


# In[2]:


dm_mass_full = subhalos['SubhaloMassType'][:,1] / h
star_mass_full = subhalos['SubhaloMassType'][:,4] / h
gas_mass_full = subhalos['SubhaloMassType'][:,0] / h
hf_sm_rad_full = subhalos['SubhaloHalfmassRadType'][:,4] / h
parent_prop_full = subhalos['SubhaloParent']
group_num_full = subhalos['SubhaloGrNr']
pos_full = subhalos['SubhaloPos'] / h
num_part_full = subhalos['SubhaloLenType']
gas_met_full = subhalos['SubhaloGasMetallicity']
star_met_full = subhalos['SubhaloStarMetallicity']
ID_full = np.arange(len(dm_mass_full))


# ## Haslbauer distance criterion and SubhaloMassType

# In[3]:


bools = (star_mass_full>0)
dm_mass = dm_mass_full[bools]
hf_sm_rad = hf_sm_rad_full[bools]
group_num = group_num_full[bools]
pos = pos_full[bools]
star_mass = star_mass_full[bools]
gas_mass = gas_mass_full[bools]
num_part = num_part_full[bools]
gas_met = gas_met_full[bools]
star_met = star_met_full[bools]
ID = ID_full[bools]
print(pos.shape)

dmc = (dm_mass != 0)
dmf = (dm_mass == 0)

print('DMCs: ', np.sum(dmc), '\nDMFs: ', np.sum(dmf))

groups = [0]
for i in range(len(group_num)-1):
    if group_num[i] != group_num[i+1]:
        groups.append(i+1)

bools = (star_mass == -1)#(star_mass > 5e-3)
s = np.zeros_like(star_mass)
hst_shmr = np.zeros_like(star_mass)
hst_cands = np.where(star_mass > 0.1)[0]
for i in range(len(dm_mass)):
    hst_cands_i = hst_cands[star_mass[hst_cands] > 10*star_mass[i]]
    dist = np.minimum(np.abs(pos[i]-pos[hst_cands_i]), L-np.abs(pos[i]-pos[hst_cands_i]))**2
    dist = np.sum(dist, axis=1)**0.5
    if len(dist) == 0:
        bools[i] = False
        continue
    idx = np.argmin(dist)
    hst_index = hst_cands_i[idx]
    s[i] = dist[idx]
    hst_shmr[i] = hf_sm_rad[hst_index]
    bools[i] = (s[i] > 10*hst_shmr[i]) & (s[i] < 100*hst_shmr[i])

print('All Gals: ', np.sum(bools))

dm_mass_cands = dm_mass[bools]
star_mass_cands = star_mass[bools]
gas_mass_cands = gas_mass[bools]
hf_sm_rad_cands = hf_sm_rad[bools]
pos_cands = pos[bools]
group_num_cands = group_num[bools]
num_part_cands = num_part[bools]
gas_met_cands = gas_met[bools]
star_met_cands = star_met[bools]
ID_cands = ID[bools]

TDGC = (dm_mass_cands == 0) & (star_mass_cands > 5e-3)
rich = (dm_mass_cands/star_mass_cands >= 1) & (star_mass_cands < 0.1) & (star_mass_cands > 5e-3) & (dm_mass_cands > 0)
poor = (dm_mass_cands/star_mass_cands < 1) & (star_mass_cands < 0.1) & (star_mass_cands > 5e-3) & (dm_mass_cands > 0)

print('TDGCs: ',np.sum(TDGC), '\nDM-rich: ', np.sum(rich),'\nDM-poor: ', np.sum(poor))


# In[13]:


base_path = './haslbauer_subhalomasstype'
pathlib.Path(base_path).mkdir(parents=True, exist_ok=True)

plt.loglog(hst_shmr[hst_shmr!=0], s[hst_shmr!=0], '.')
x = np.linspace(10**(-2), 10**4, 100)

plt.loglog(x, 2*x, '--b', label='s/shmr = 2')
plt.loglog(x, 5*x, '--k', label='s/shmr = 5')
plt.loglog(x, 10*x, '--r', label='s/shmr = 10')
plt.loglog(x, 100*x, '--g', label='s/shmr = 100')

plt.legend(fontsize='x-large')
plt.xlabel('host stellar half mass', fontsize=20)
plt.ylabel('Distance to host', fontsize=20)
plt.savefig(base_path+'/dist_crit.png')


# In[16]:


plt.hist2d(10 + np.log10(star_mass[dmc]), gas_met[dmc], bins=30, cmap='Blues', norm=LogNorm())
plt.colorbar(orientation='horizontal', label='counts in each bin', cmap='Blues')
plt.scatter(10+np.log10(star_mass_cands[TDGC]), gas_met_cands[TDGC], marker='^', color='orangered', edgecolor='red', alpha=0.7, label='TDGCs')
plt.scatter(10+np.log10(star_mass_cands[poor]), gas_met_cands[poor], marker=',', color='violet', edgecolor='m', alpha=0.7, label='DM-poor DGs')
plt.scatter(10+np.log10(star_mass_cands[rich]), gas_met_cands[rich], marker='.', color='forestgreen', edgecolor='forestgreen', alpha=0.5, s=42, label='DM-rich DGs')
plt.xlabel(r'log$_{10}$(M$_{stellar}$ [$M_\odot$])')
plt.ylabel(r'$z_{gas}$')
plt.title('Stellar Metallicity for Haslbauer+SubhaloMassType')
plt.legend()
plt.savefig(base_path+'/star_met.png')


# In[17]:


plt.hist2d(10 + np.log10(star_mass[dmc]), gas_met[dmc], bins=30, cmap='Blues', norm=LogNorm())
plt.colorbar(orientation='horizontal', label='counts in each bin', cmap='Blues')
plt.scatter(10+np.log10(star_mass_cands[TDGC]), gas_met_cands[TDGC], marker='^', color='orangered', edgecolor='red', alpha=0.7, label='TDGCs')
plt.scatter(10+np.log10(star_mass_cands[poor]), gas_met_cands[poor], marker=',', color='violet', edgecolor='m', alpha=0.7, label='DM-poor DGs')
plt.scatter(10+np.log10(star_mass_cands[rich]), gas_met_cands[rich], marker='.', color='forestgreen', edgecolor='forestgreen', alpha=0.5, s=42, label='DM-rich DGs')
plt.xlabel(r'log$_{10}$(M$_{stellar}$ [$M_\odot$])')
plt.ylabel(r'$z_{gas}$')
plt.title('Gas Metallicity for Haslbauer+SubhaloMassType')
plt.legend()
plt.savefig(base_path+'/gas_met.png')


# ## Haslbauer distance criterion and radial profiles

# In[6]:


@njit
def subh_bool(bools, ratio, dm_pos, star_pos, star_mass, pos, subs, dm_part_mass, hf_sm_rad, h, L):
    for subh in subs:
        dist = np.sum(np.minimum(np.abs(dm_pos/h-pos[subh]), L - np.abs(dm_pos/h-pos[subh]))**2, axis=1)**0.5
        dist2 = np.sum(np.minimum(np.abs(star_pos/h-pos[subh]), L-np.abs(star_pos/h-pos[subh]))**2, axis=1)**0.5
        shmr3 = 3*hf_sm_rad[subh]
        star_radial_mass = np.sum(star_mass[dist2<=shmr3]/h)
        if star_radial_mass == 0:
            bools[subh] = 0
            ratio[subh] = -1
        else:
            r = np.sum(dm_part_mass*(dist<=shmr3))/star_radial_mass
            ratio[subh] = r
            bools[subh] = (r<1)


# In[19]:


idx = (dm_mass_cands/star_mass_cands < 1)
dm_mass_low = dm_mass_cands[idx]
star_mass_low = star_mass_cands[idx]
hf_sm_rad_low = hf_sm_rad_cands[idx]
pos_low = pos_cands[idx]
group_num_low = group_num_cands[idx]
num_part_low = num_part_cands[idx]
gas_met_low = gas_met_cands[idx]
star_met_low = star_met_cands[idx]
ID_low = ID_cands[idx]
print(pos_low.shape)

groups_rad = [0]
for i in range(len(group_num_low)-1):
    if group_num_low[i] != group_num_low[i+1]:
        groups_rad.append(i+1)

fields_star = ['Masses','Coordinates']
fields_dm = ['Coordinates']
lowdm_rad = (dm_mass_low == -100)
ratio = np.zeros_like(dm_mass_low)

for i in range(len(groups_rad)-1):
    dm_pos = il.snapshot.loadHalo(basePath,99, group_num_low[groups_rad[i]], 'dm', fields_dm)
    stars = il.snapshot.loadHalo(basePath,99, group_num_low[groups_rad[i]], 'stars', fields_star)
    subh_bool(lowdm_rad, ratio, dm_pos, stars['Coordinates'], stars['Masses'], pos_low, list(range(groups_rad[i], groups_rad[i+1])),dm_part_mass, hf_sm_rad_cands, h, L)

print('Low DM within 3r_0.5: ', np.sum(lowdm_rad))


# In[22]:


base_path = './haslbauer_radial'
pathlib.Path(base_path).mkdir(parents=True, exist_ok=True)

plt.loglog(hst_shmr[hst_shmr!=0], s[hst_shmr!=0], '.')
x = np.linspace(10**(-2), 10**4, 100)

plt.loglog(x, 2*x, '--b', label='s/shmr = 2')
plt.loglog(x, 5*x, '--k', label='s/shmr = 5')
plt.loglog(x, 10*x, '--r', label='s/shmr = 10')
plt.loglog(x, 100*x, '--g', label='s/shmr = 100')

plt.legend(fontsize='x-large')
plt.xlabel('host stellar half mass', fontsize=20)
plt.ylabel('Distance to host', fontsize=20)
plt.savefig(base_path+'/dist_crit.png')


# In[23]:


plt.hist2d(10 + np.log10(star_mass[dmc]), gas_met[dmc], bins=30, cmap='Blues', norm=LogNorm())
plt.colorbar(orientation='horizontal', label='counts in each bin', cmap='Blues')
plt.scatter(10+np.log10(star_mass_low[lowdm_rad]), gas_met_low[lowdm_rad], marker='^', color='orangered', edgecolor='red', alpha=0.7, label='Low DM')
# plt.scatter(10+np.log10(star_mass_cands[poor]), gas_met_cands[poor], marker=',', color='violet', edgecolor='m', alpha=0.7, label='DM-poor DGs')
# plt.scatter(10+np.log10(star_mass_cands[rich]), gas_met_cands[rich], marker='.', color='forestgreen', edgecolor='forestgreen', alpha=0.5, s=42, label='DM-rich DGs')
plt.xlabel(r'log$_{10}$(M$_{stellar}$ [$M_\odot$])')
plt.ylabel(r'$z_{gas}$')
plt.title('Stellar Metallicity Distribution for Haslbauer+Radial Profile')
plt.legend()
plt.savefig(base_path+'/star_met.png')


# In[25]:


plt.hist2d(10 + np.log10(star_mass[dmc]), gas_met[dmc], bins=30, cmap='Blues', norm=LogNorm())
plt.colorbar(orientation='horizontal', label='counts in each bin', cmap='Blues')
plt.scatter(10+np.log10(star_mass_low[lowdm_rad]), gas_met_low[lowdm_rad], marker='^', color='orangered', edgecolor='red', alpha=0.7, label='Low DM')
# plt.scatter(10+np.log10(star_mass_cands[poor]), gas_met_cands[poor], marker=',', color='violet', edgecolor='m', alpha=0.7, label='DM-poor DGs')
# plt.scatter(10+np.log10(star_mass_cands[rich]), gas_met_cands[rich], marker='.', color='forestgreen', edgecolor='forestgreen', alpha=0.5, s=42, label='DM-rich DGs')
plt.xlabel(r'log$_{10}$(M$_{stellar}$ [$M_\odot$])')
plt.ylabel(r'$z_{gas}$')
plt.title('Gas Metallicity Distribution for Haslbauer+Radial Profile')
plt.legend()
plt.savefig(base_path+'/gas_met.png')


# ## Hai and SubhaloMasstype

# In[27]:


hai_subh = (parent_prop_full==0) & (star_mass_full>0.001) & (num_part_full[:,4]>20)
dm_mass = dm_mass_full[hai_subh]
hf_sm_rad = hf_sm_rad_full[hai_subh]
group_num = group_num_full[hai_subh]
pos = pos_full[hai_subh]
star_mass = star_mass_full[hai_subh]
num_part = num_part_full[hai_subh]
gas_met = gas_met_full[hai_subh]
star_met = star_met_full[hai_subh]
ID = ID_full[hai_subh]
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

