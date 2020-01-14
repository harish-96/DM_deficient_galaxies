
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('pylab', 'inline')
import illustris_python as il
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import h5py


# In[3]:


f = h5py.File('../sims.illustris/Illustris-1/output/snapdir_135/snap_135.0.hdf5')
header = dict(f['Header'].attrs)

h = header['HubbleParam']
L = header['BoxSize'] / h
dm_part_mass = header['MassTable'][1] / h


# In[4]:


basePath = '../sims.illustris/Illustris-1/output'
fields = ['SubhaloMassType','SubhaloParent','SubhaloHalfmassRadType','SubhaloPos', 'SubhaloLenType', 'SubhaloGrNr']
subhalos = il.groupcat.loadSubhalos(basePath,135,fields=fields)
GroupFirstSub = il.groupcat.loadHalos(basePath, 135, fields='GroupFirstSub')
# subhalo_halfmassrad = il.groupcat.loadSubhalos(basePath,135,fields='SubhaloHalfmassRadType')
print(len(GroupFirstSub))


# In[5]:


h = 0.6774
dm_mass_full = subhalos['SubhaloMassType'][:,1]/h
star_mass_full = subhalos['SubhaloMassType'][:,4]/h
hf_sm_rad_full = subhalos['SubhaloHalfmassRadType'][:,4]/h
parent_prop_full = subhalos['SubhaloParent']
group_num_full = subhalos['SubhaloGrNr']
pos_full = subhalos['SubhaloPos']/h
num_part_full = subhalos['SubhaloLenType']
L = 750000/h

bools = (star_mass_full!=0) & (parent_prop_full==0) & (star_mass_full>0.001) & (num_part_full[:,4]>20)
dm_mass = dm_mass_full[bools]
hf_sm_rad = hf_sm_rad_full[bools]
group_num = group_num_full[bools]
pos = pos_full[bools]
star_mass = star_mass_full[bools]
num_part = num_part_full[bools]

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
print(pos.shape)


# In[6]:


idx = (dm_mass/star_mass < 1)
dm_mass_low = dm_mass[idx]
star_mass_low = star_mass[idx]
hf_sm_rad_low = hf_sm_rad[idx]
pos_low = pos[idx]
group_num_low = group_num[idx]
num_part_low = num_part[idx]

print(pos_low.shape)


# In[8]:


groups = [0]
for i in range(len(group_num_low)-1):
    if group_num_low[i] != group_num_low[i+1]:
        groups.append(i+1)
print(len(groups))


# In[69]:


from numba import njit

@njit
def subh_bool(bools, dm_pos, star_pos, star_mass, pos_low, subs, dm_part_mass, hf_sm_rad_low, h, L):
    for subh in subs:
        dist = np.sum(np.minimum(np.abs(dm_pos/h-pos_low[subh]), L - np.abs(dm_pos/h-pos_low[subh]))**2, axis=1)**0.5
        dist2 = np.sum(np.minimum(np.abs(star_pos/h-pos_low[subh]), L-np.abs(star_pos/h-pos_low[subh]))**2, axis=1)**0.5
        shmr3 = 50 #3*hf_sm_rad_low[subh]
        ratio = np.sum(dm_part_mass*(dist<shmr3))/np.sum(star_mass[dist2<shmr3]/h)
        bools[subh] = (ratio<1)


# In[70]:


fields_star = ['Masses','Coordinates']
fields_dm = ['Coordinates']
bools = (dm_mass_low == -100)
for i in range(len(groups)-1):
    dm_pos = il.snapshot.loadHalo(basePath,135, group_num_low[groups[i]], 'dm', fields_dm)
    stars = il.snapshot.loadHalo(basePath,135, group_num_low[groups[i]], 'stars', fields_star)
    subh_bool(bools, dm_pos, stars['Coordinates'], stars['Masses'], pos_low, list(range(groups[i],groups[i+1])), dm_part_mass, hf_sm_rad_low, h, L)
    print(i, np.sum(bools))


# In[61]:


fields_star = ['Masses','Coordinates']
fields_dm = ['Coordinates']
bools = (dm_mass_low == -100)
tot = 0
for i in range(len(groups)-1):
    dm_pos = il.snapshot.loadHalo(basePath,135, group_num_low[groups[i]], 'dm', fields_dm)
    stars = il.snapshot.loadHalo(basePath,135, group_num_low[groups[i]], 'stars', fields_star)
    for subh in range(groups[i], groups[i+1]):
        dist = np.minimum(np.abs(dm_pos/h-pos_low[subh]), L - np.abs(dm_pos/h-pos_low[subh]))**2
        dist = np.sum(dist, axis=1)**0.5
        dist2 = np.minimum(np.abs(stars['Coordinates']/h-pos_low[subh]), L-np.abs(stars['Coordinates']/h-pos_low[subh]))**2
        dist2 = np.sum(dist2, axis=1)**0.5
        ratio = np.sum(dm_part_mass*(dist<3*hf_sm_rad_low[subh]))/np.sum(stars['Masses'][dist2<3*hf_sm_rad_low[subh]]/h)
#         ratio = np.sum(dm_part_mass*(dist<50))/np.sum(stars['Masses'][dist2<50]/h)
        bools[subh] = (ratio<1)
        tot += 1
        print(ratio, bools[subh], np.sum(bools))
    #print(i, tot, np.sum(bools))


# In[62]:


np.sum(bools)


# In[56]:


subh = 2
dist = np.minimum(np.abs(dm_pos/h-pos_low[subh]), L - np.abs(dm_pos/h-pos_low[subh]))**2
dist = np.sum(dist, axis=1)**0.5
dist2 = np.minimum(np.abs(stars['Coordinates']/h-pos_low[subh]), L-np.abs(stars['Coordinates']/h-pos_low[subh]))**2
dist2 = np.sum(dist2, axis=1)**0.5
ratio = np.sum(dm_part_mass*(dist<3*hf_sm_rad_low[subh]))/np.sum(stars['Masses'][dist2<3*hf_sm_rad_low[subh]]/h)


# In[42]:


np.min(dist2)


# In[43]:


3*hf_sm_rad_low[subh]


# In[57]:


ratio


# In[58]:


ratio<1


# In[51]:


bools = (group_num_low < -1)


# In[53]:


bools[subh] = (ratio<1)


# In[54]:


np.sum(bools)

