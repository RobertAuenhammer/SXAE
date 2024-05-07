# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 08:54:57 2023
"""
# # -*- coding: utf-8 -*-
# Created on Mon Jan 18 19:58:03 2021
# Robert Auenhammer, Chalmers University of Technology
import os
import numpy as np
import scipy.io
import time
import matplotlib.pyplot as plt

from functions import computeMT, computeMT_local, computeStiff_m, computeStiff_f, computeEshelby, E__, E__update, Rotate_z_E__
from C_plots import plot_compare_E11_histogram, plot_compare_E22_histogram, plot_compare_E33_histogram
##----------------------------------------------------------------------------------------------------------------------
#%%
start        = time.time()
script_name  = 'CodeOcean'
data_path    = '../data'
results_path = '../results'
##----------------------------------------------------------------------------------------------------------------------
#%% Define Constituent properties
Vf_mean  = 0.25
Em       = 4.48e+9          #[Pa]
num      = 0.35

E1f      = 294e9            #[Pa]
E2f      = 19e9             #[Pa]
E3f      = 19e9             #[Pa]
G12f     = 27e9             #[Pa]
G13f     = 27e9             #[Pa]
G23f     = 7e9              #[Pa]
nu12f    = 0.2
nu13f    = 0.2
nu23f    = 0.7
#%% Define fibre dimensions
a1       = 50e-6             #[m]
a2       = 3.5e-6            #[m]
a3       = 3.5e-6            #[m]
##----------------------------------------------------------------------------------------------------------------------
#%% Load tomography data
eig_vec_cf_peek      = scipy.io.loadmat(os.path.join(data_path, 'vec.mat'))['vec']
eig_val_cf_peek      = scipy.io.loadmat(os.path.join(data_path, 'val.mat'))['val']
eig_val_peek         = scipy.io.loadmat(os.path.join(data_path, 'val_peek.mat'))['val']
print('Data loaded')
##----------------------------------------------------------------------------------------------------------------------
#%% 
mean_scattering_cf_peek_array   = np.mean(eig_val_cf_peek, axis=-1)
mean_scattering_cf_peek_value   = np.mean(eig_val_cf_peek, where=eig_val_cf_peek>0.4)
#%% 
mean_scattering_pure_peek_array = np.mean(eig_val_peek, axis=-1)
mean_scattering_pure_peek_value = np.mean(eig_val_peek, where=eig_val_peek>0.25)
#%% Linear Correlation between scattering and fibre volume fraction
mean_Vf          = 0.199
slope            = (mean_scattering_cf_peek_value-mean_scattering_pure_peek_value)/mean_Vf
Vf_tomo          = (mean_scattering_cf_peek_array-mean_scattering_pure_peek_value)/slope
##----------------------------------------------------------------------------------------------------------------------
#%% 
# Adjust the size of the arrays, very small here for fast computation
ori      = eig_vec_cf_peek[200:203,200:203,200:203,:,2]
# 1st Eigenvalue
lambda1  = eig_val_cf_peek[200:203,200:203,200:203,0]
# 2nd Eigenvalue
lambda2  = eig_val_cf_peek[200:203,200:203,200:203,1]
# 3rd Eigenvalue
lambda3  = eig_val_cf_peek[200:203,200:203,200:203,2]
# Scattering array
mean_scattering_cf_peek_array_loop = mean_scattering_cf_peek_array[200:203,200:203,200:203]
print('Initialisation done')
##----------------------------------------------------------------------------------------------------------------------
#%% Write Abaqus input file
binning     = 1
voxelsize   = 100
es          = voxelsize*binning
noNo        = (int(ori.shape[0]/binning+1))*int((ori.shape[1]/binning+1))*int((ori.shape[2]/binning)+1)
noEl        = int(ori.shape[0]/binning)*int(ori.shape[1]/binning)*int(ori.shape[2]/binning)
IP_coords   = np.empty((noEl, 8, 3))
elements    = np.empty((noEl, 9))
elements    = elements.astype(int)

x_shape     = int(ori.shape[0]/binning)
y_shape     = int(ori.shape[1]/binning)
z_shape     = int(ori.shape[2]/binning)
print(x_shape,y_shape,z_shape)

startX=0
startY=0
startZ=0
kk=0
ee=1

nodes = np.empty((noNo, 4))
with open(os.path.join(results_path, 'mesh.inp'), 'w') as output:
    output.write('*Node' '\n')
    
    for l in range(0,z_shape):
        for k in range(0,y_shape):
            for h in range(0,x_shape):
                kk=kk+1
                if mean_scattering_cf_peek_array_loop[h*binning,k*binning,l*binning] > 0.6*mean_scattering_cf_peek_value:
                    # Node definition
                    n8 = kk
                    n7 = kk+1
                    n5 = kk+x_shape+1
                    n6 = kk+x_shape+1+1
                    n4 = kk+(x_shape+1)*(y_shape+1)
                    n3 = kk+(x_shape+1)*(y_shape+1)+1
                    n1 = kk+(x_shape+1)*(y_shape+1)+x_shape+1
                    n2 = kk+(x_shape+1)*(y_shape+1)+1+x_shape+1

                    if l == z_shape or k == y_shape or h == x_shape:
                        ee = ee-1
                    else:
                        elements[ee-1,0] = ee
                        elements[ee-1,8] = n8
                        elements[ee-1,7] = n7
                        elements[ee-1,5] = n5
                        elements[ee-1,6] = n6
                        elements[ee-1,4] = n4
                        elements[ee-1,3] = n3
                        elements[ee-1,1] = n1
                        elements[ee-1,2] = n2

    #                       Node 8:
                        nodes[n8-1,0] = n8
                        nodes[n8-1,1] = startX+h*es
                        nodes[n8-1,2] = startY+k*es
                        nodes[n8-1,3] = startZ+l*es


    #                       Node 7:
                        nodes[n7-1,0] = n7
                        nodes[n7-1,1] = startX+h*es + es
                        nodes[n7-1,2] = startY+k*es
                        nodes[n7-1,3] = startZ+l*es


    #                       Node 6:                
                        nodes[n6-1,0] = n6
                        nodes[n6-1,1] = startX+h*es + es
                        nodes[n6-1,2] = startY+k*es + es
                        nodes[n6-1,3] = startZ+l*es


    #                       Node 5:
                        nodes[n5-1,0] = n5
                        nodes[n5-1,1] = startX+h*es
                        nodes[n5-1,2] = startY+k*es + es
                        nodes[n5-1,3] = startZ+l*es


    #                       Node 4:
                        nodes[n4-1,0] = n4
                        nodes[n4-1,1] = startX+h*es
                        nodes[n4-1,2] = startY+k*es
                        nodes[n4-1,3] = startZ+l*es + es


    #                       Node 3:
                        nodes[n3-1,0] = n3
                        nodes[n3-1,1] = startX+h*es + es
                        nodes[n3-1,2] = startY+k*es
                        nodes[n3-1,3] = startZ+l*es + es


    #                       Node 2:                        
                        nodes[n2-1,0] = n2
                        nodes[n2-1,1] = startX+h*es + es
                        nodes[n2-1,2] = startY+k*es + es
                        nodes[n2-1,3] = startZ+l*es + es


    #                       Node 1:
                        nodes[n1-1,0] = n1
                        nodes[n1-1,1] = startX+h*es
                        nodes[n1-1,2] = startY+k*es + es
                        nodes[n1-1,3] = startZ+l*es + es
                        
                        X0              = startX+h*es
                        Y0              = startY+k*es
                        Z0              = startZ+l*es
                        
                        IP_coords[ee-1][0][:] = [X0+es*0.215, Y0+es*0.785, Z0+es*0.785]
                        IP_coords[ee-1][1][:] = [X0+es*0.785, Y0+es*0.785, Z0+es*0.785]
                        IP_coords[ee-1][2][:] = [X0+es*0.215, Y0+es*0.215, Z0+es*0.785]
                        IP_coords[ee-1][3][:] = [X0+es*0.785, Y0+es*0.215, Z0+es*0.785]
                        IP_coords[ee-1][4][:] = [X0+es*0.215, Y0+es*0.785, Z0+es*0.215]
                        IP_coords[ee-1][5][:] = [X0+es*0.785, Y0+es*0.785, Z0+es*0.215]
                        IP_coords[ee-1][6][:] = [X0+es*0.215, Y0+es*0.215, Z0+es*0.215]
                        IP_coords[ee-1][7][:] = [X0+es*0.785, Y0+es*0.215, Z0+es*0.215]     

                    ee=ee+1           
    noEl      = ee-1
    print(noEl)
    IP_coords = IP_coords[0:noEl]
    print('Integration point array created')
    print('Arrays created')
    nodes     = nodes[nodes[:,0]  >=1]
    nodes     = nodes[nodes[:,0]  <noNo]
    nodes     = nodes[nodes[:,1]  >1]
    nodes     = nodes[nodes[:,2]  >1]
    nodes     = nodes[nodes[:,3]  >1]
    nodes     = nodes[nodes[:,3]  <ori.shape[2]*voxelsize]

    #Center nodes and IP coords
    x_offset         = -20.001e3
    y_offset         = -11.590e3
    z_offset         = -14.601e3
    
    IP_coords_off        = np.empty((noEl,8,3))

    IP_coords_off[:,:,0] = IP_coords[:,:,0] + x_offset
    IP_coords_off[:,:,1] = IP_coords[:,:,1] + y_offset
    IP_coords_off[:,:,2] = IP_coords[:,:,2] + z_offset
    nodes[:,1]       = nodes[:,1] + x_offset
    nodes[:,2]       = nodes[:,2] + y_offset
    nodes[:,3]       = nodes[:,3] + z_offset  


    alpha=-(90+4.7)
    alpha=np.deg2rad(alpha)
    R=np.array([[np.cos(alpha), -np.sin(alpha), 0],
                [np.sin(alpha),   np.cos(alpha), 0],
                [0,               0,             1]])
    # Rotate IP_coords
    IP_coords_rot = np.empty((noEl,8,3))
    beta          = np.empty((noEl,8))
    for j in range(noEl):
        for i in range(8):
            IP_coords_rot[j,i,:] = np.dot(R,IP_coords_off[j,i,:])

            if IP_coords_rot[j,i,2]       == 0:
                beta[j,i] == np.pi/2
            elif IP_coords_rot[j,i,2]     > 0:
                beta[j,i] = np.arctan(IP_coords_rot[j,i,1]/IP_coords_rot[j,i,2])
            else:
                beta[j,i] = np.arctan(IP_coords_rot[j,i,1]/IP_coords_rot[j,i,2]) + np.pi
    
    #Write nodes in Abaqus syntax
    for i in range(int(nodes.shape[0])):
        # Rotate nodes along pultrusion direction
        nodes[i,1:4]=np.dot(R,nodes[i,1:4])

        output.write(f' {int(nodes[i,0])}, {nodes[i,1]/1000}, {nodes[i,2]/1000}, {nodes[i,3]/1000}' '\n')
                                                                                              
    print('Nodes written')
                
               
    #Write elements in Abaqus syntax            
    output.write('** \n') 
    output.write('*ELEMENT, TYPE=C3D8, ELSET=FoV' '\n')

    for i in range(ee-1):
        output.write(f' {elements[i,0]}, {elements[i,1]}, {elements[i,2]}, {elements[i,3]}, {elements[i,4]}, {elements[i,5]}, {elements[i,6]}, {elements[i,7]}, {elements[i,8]}' '\n')

    output.write('**' '\n') 
    #Solid Section
    output.write('*Solid SECTION, ELSET=FoV, MATERIAL=PEEK_CF_30' '\n')
 
    #Material definition
    output.write('** MATERIALS' '\n')
    output.write('*MATERIAL, NAME=PEEK_CF' '\n')
    output.write('*DENSITY' '\n')
    output.write('                  1.6E-9,' '\n')
    output.write('*DEPVAR' '\n')
    output.write('              9,' '\n')
    output.write('*USER MATERIAL, constants=1' '\n')
    output.write('0,' '\n')
    output.write('**--------------------------------------------------------------------------------------------------------------------------------------------------------------------' '\n')
    output.write('**' '\n')
    print('Elements written')
    output.write('**--------------------------------------------------------------------------------------------------------------------------------------------------------------------' '\n')
print('Modell created') 
##----------------------------------------------------------------------------------------------------------------------
#%% Mori-Tanaka tensor
# Compute stiffness matrices of fibre and matrix
C_f = computeStiff_f(E1f, E2f, E3f, G12f, G13f, G23f, nu12f, nu13f, nu23f)
C_m = computeStiff_m(num, Em)

# Comput Eshelby Tensor 
Eshelby   = computeEshelby(num, a1,a2,a3)

# Compute Mori-Tanaka stiffness tensor
C_MT    = computeMT(C_f, C_m, Eshelby, Vf_mean, results_path)
C_MT    = np.asarray(C_MT)
C_11_0 = C_MT[0,0]
C_22_0 = C_MT[1,1]
C_66_0 = C_MT[4,4]
C_12_0 = C_MT[0,1]
C_23_0 = C_MT[1,2]
print('C_MT_11', f'{C_11_0/1e9:.2f}', 'GPa')
print('C_MT_22', f'{C_22_0/1e9:.2f}', 'GPa')
print('C_MT_66', f'{C_66_0/1e9:.2f}', 'GPa')
print('C_MT_12', f'{C_12_0/1e9:.2f}', 'GPa')
print('C_MT_23', f'{C_23_0/1e9:.2f}', 'GPa')
##----------------------------------------------------------------------------------------------------------------------
#%% Calculate new stiffness components based on the fibre orientation scattering
nE11 = np.empty((noEl,8))
nE12 = np.empty((noEl,8))
nE13 = np.empty((noEl,8))
nE14 = np.empty((noEl,8))
nE15 = np.empty((noEl,8))
nE16 = np.empty((noEl,8))
nE22 = np.empty((noEl,8))
nE23 = np.empty((noEl,8))
nE24 = np.empty((noEl,8))
nE25 = np.empty((noEl,8))
nE26 = np.empty((noEl,8))
nE33 = np.empty((noEl,8))
nE34 = np.empty((noEl,8))
nE35 = np.empty((noEl,8))
nE36 = np.empty((noEl,8))
nE41 = np.empty((noEl,8))
nE42 = np.empty((noEl,8))
nE43 = np.empty((noEl,8))
nE44 = np.empty((noEl,8))
nE45 = np.empty((noEl,8))
nE46 = np.empty((noEl,8))
nE51 = np.empty((noEl,8))
nE52 = np.empty((noEl,8))
nE53 = np.empty((noEl,8))
nE54 = np.empty((noEl,8))
nE55 = np.empty((noEl,8))
nE56 = np.empty((noEl,8))
nE61 = np.empty((noEl,8))
nE62 = np.empty((noEl,8))
nE63 = np.empty((noEl,8))
nE66 = np.empty((noEl,8))
oE11 = np.empty((noEl,8))
oE12 = np.empty((noEl,8))
oE13 = np.empty((noEl,8))
oE14 = np.empty((noEl,8))
oE15 = np.empty((noEl,8))
oE16 = np.empty((noEl,8))
oE22 = np.empty((noEl,8))
oE23 = np.empty((noEl,8))
oE24 = np.empty((noEl,8))
oE25 = np.empty((noEl,8))
oE26 = np.empty((noEl,8))
oE33 = np.empty((noEl,8))
oE34 = np.empty((noEl,8))
oE35 = np.empty((noEl,8))
oE36 = np.empty((noEl,8))
oE41 = np.empty((noEl,8))
oE42 = np.empty((noEl,8))
oE43 = np.empty((noEl,8))
oE44 = np.empty((noEl,8))
oE45 = np.empty((noEl,8))
oE46 = np.empty((noEl,8))
oE51 = np.empty((noEl,8))
oE52 = np.empty((noEl,8))
oE53 = np.empty((noEl,8))
oE54 = np.empty((noEl,8))
oE55 = np.empty((noEl,8))
oE56 = np.empty((noEl,8))
oE61 = np.empty((noEl,8))
oE62 = np.empty((noEl,8))
oE63 = np.empty((noEl,8))
oE66 = np.empty((noEl,8))


C_m9    = np.load(os.path.join(results_path, '001_C_m9.npy'))
C_f9    = np.load(os.path.join(results_path, '001_C_f9.npy'))
A_MT9   = np.load(os.path.join(results_path, '001_A_MT9.npy'))

a=0
for j in range(noEl):
    for i in range(8):

        g1=int(IP_coords[j][i][0]/voxelsize)
        g2=int(IP_coords[j][i][1]/voxelsize)
        g3=int(IP_coords[j][i][2]/voxelsize)
        
       
        ori_s = np.sign(ori[g1,g2,g3,0])*ori[g1,g2,g3,:]
        #Calculate theta
        theta_center = np.arccos((ori_s[2]))
        
        #Calculate phi
        phi_center = np.arctan(ori_s[1]/ori_s[0])
        if np.isnan(phi_center) == True:
            phi_center   = np.pi/2
            theta_center = np.pi/2
        

        ##Assumed linear relation between the Directional Anisotropy values (0...1) and the angles (0...90)
        DA_xy = lambda3[g1,g2,g3]/lambda1[g1,g2,g3]
        if DA_xy > 0:
            phi_intervall   = DA_xy*np.pi/2
            phi_scat=np.arange(phi_center-phi_intervall/2, phi_center+phi_intervall/2, phi_intervall/10)
        else:
            phi_scat=np.array((phi_center, phi_center))
            
        DA_xz = lambda3[g1,g2,g3]/lambda2[g1,g2,g3]    
        if DA_xz > 0:
            theta_intervall = DA_xz*np.pi/2
            theta_scat = np.arange(theta_center-theta_intervall/2, theta_center+theta_intervall/2, theta_intervall/10)
        else:
            theta_scat = np.array((theta_center, theta_center))
        #=========================================================================
        ## Local fibre volume fraction
        #=========================================================================
        if Vf_tomo[g1,g2,g3]   < 0.1:
            Vf         = 0.1
        elif Vf_tomo[g1,g2,g3] > 0.4:   
            Vf         = 0.4
        else:    
            Vf         = Vf_tomo[g1,g2,g3]
 
        C_MT           = computeMT_local(C_m9, C_f9, A_MT9, Vf)
        C_MT           = np.asarray(C_MT)        
        C_11           = C_MT[0,0]
        C_22           = C_MT[1,1]
        C_66           = C_MT[5,5]
        C_12           = C_MT[0,1]
        C_23           = C_MT[1,2]
        #=========================================================================

        C              = E__update(phi_scat, theta_scat, C_11, C_22, C_66, C_12, C_23)
        C              = np.asarray(C)
        C              = C/1e6
        size_loop      = len(phi_scat)*len(theta_scat)
        
        nE11[j,i]      = np.sum(C[0])/size_loop
        nE12[j,i]      = np.sum(C[1])/size_loop
        nE13[j,i]      = np.sum(C[2])/size_loop
        nE14[j,i]      = np.sum(C[3])/size_loop
        nE15[j,i]      = np.sum(C[4])/size_loop
        nE16[j,i]      = np.sum(C[5])/size_loop
        
        nE22[j,i]      = np.sum(C[6])/size_loop
        nE23[j,i]      = np.sum(C[7])/size_loop
        nE24[j,i]      = np.sum(C[8])/size_loop
        nE25[j,i]      = np.sum(C[9])/size_loop
        nE26[j,i]      = np.sum(C[10])/size_loop

        nE33[j,i]      = np.sum(C[11])/size_loop
        nE34[j,i]      = np.sum(C[12])/size_loop
        nE35[j,i]      = np.sum(C[13])/size_loop
        nE36[j,i]      = np.sum(C[14])/size_loop

        nE41[j,i]      = np.sum(C[15])/size_loop
        nE42[j,i]      = np.sum(C[16])/size_loop
        nE43[j,i]      = np.sum(C[17])/size_loop
        nE44[j,i]      = np.sum(C[18])/size_loop
        nE45[j,i]      = np.sum(C[19])/size_loop
        nE46[j,i]      = np.sum(C[20])/size_loop

        nE51[j,i]      = np.sum(C[21])/size_loop
        nE52[j,i]      = np.sum(C[22])/size_loop
        nE53[j,i]      = np.sum(C[23])/size_loop
        nE55[j,i]      = np.sum(C[24])/size_loop
        nE56[j,i]      = np.sum(C[25])/size_loop

        nE61[j,i]      = np.sum(C[26])/size_loop
        nE62[j,i]      = np.sum(C[27])/size_loop
        nE63[j,i]      = np.sum(C[28])/size_loop
        nE66[j,i]      = np.sum(C[29])/size_loop
        
        
        a              = a + size_loop
        
        
        C              = E__(phi_center, theta_center, C_11_0, C_22_0, C_66_0, C_12_0, C_23_0)
        C              = np.asarray(C)
        C              = C/1e6
        oE11[j,i]      = C[0]
        oE12[j,i]      = C[1]
        oE13[j,i]      = C[2]
        oE14[j,i]      = C[3]
        oE15[j,i]      = C[4]
        oE16[j,i]      = C[5]
        
        oE22[j,i]      = C[6]
        oE23[j,i]      = C[7]
        oE24[j,i]      = C[8]
        oE25[j,i]      = C[9]
        oE26[j,i]      = C[10]
        
        oE33[j,i]      = C[11]
        oE34[j,i]      = C[12]
        oE35[j,i]      = C[13]
        oE36[j,i]      = C[14]
        
        oE41[j,i]      = C[15]
        oE42[j,i]      = C[16]
        oE43[j,i]      = C[17]
        oE44[j,i]      = C[18]
        oE45[j,i]      = C[19]
        oE46[j,i]      = C[20]
        
        oE51[j,i]      = C[21]
        oE52[j,i]      = C[22]
        oE53[j,i]      = C[23]
        oE55[j,i]      = C[24]
        oE56[j,i]      = C[25]
        
        oE61[j,i]      = C[26]
        oE62[j,i]      = C[27]
        oE63[j,i]      = C[28]
        oE66[j,i]      = C[29]
print('Amount of evaluations', a)
print('Orientation recalculation done')
#%% Rotate Stiffness matrices
# Updated Matrix
C_rot              = Rotate_z_E__(alpha, nE11, nE12, nE13, nE14, nE15, nE16, nE22, nE23, nE24, nE25, nE26, nE33, nE34, nE35, nE36, nE41, nE42, nE43, nE44, nE45, nE46, nE51, nE52, nE53, nE55, nE56, nE61, nE62, nE63, nE66)
C_rot              = np.asarray(C_rot)
nE11_rot           = C_rot[0]
nE12_rot           = C_rot[1]
nE13_rot           = C_rot[2]
nE14_rot           = C_rot[3]
nE15_rot           = C_rot[4]
nE16_rot           = C_rot[5]

nE22_rot           = C_rot[6]
nE23_rot           = C_rot[7]
nE24_rot           = C_rot[8]
nE25_rot           = C_rot[9]
nE26_rot           = C_rot[10]

nE33_rot           = C_rot[11]
nE34_rot           = C_rot[12]
nE35_rot           = C_rot[13]
nE36_rot           = C_rot[14]

nE41_rot           = C_rot[15]
nE42_rot           = C_rot[16]
nE43_rot           = C_rot[17]
nE44_rot           = C_rot[18]
nE45_rot           = C_rot[19]
nE46_rot           = C_rot[20]

nE51_rot           = C_rot[21]
nE52_rot           = C_rot[22]
nE53_rot           = C_rot[23]
nE55_rot           = C_rot[24]
nE56_rot           = C_rot[25]

nE61_rot           = C_rot[26]
nE62_rot           = C_rot[27]
nE63_rot           = C_rot[28]
nE66_rot           = C_rot[29]

# Old Matrix
C_rot              = Rotate_z_E__(alpha, oE11, oE12, oE13, oE14, oE15, oE16, oE22, oE23, oE24, oE25, oE26, oE33, oE34, oE35, oE36, oE41, oE42, oE43, oE44, oE45, oE46, oE51, oE52, oE53, oE55, oE56, oE61, oE62, oE63, oE66)
C_rot              = np.asarray(C_rot)
oE11_rot           = C_rot[0]
oE12_rot           = C_rot[1]
oE13_rot           = C_rot[2]
oE14_rot           = C_rot[3]
oE15_rot           = C_rot[4]
oE16_rot           = C_rot[5]

oE22_rot           = C_rot[6]
oE23_rot           = C_rot[7]
oE24_rot           = C_rot[8]
oE25_rot           = C_rot[9]
oE26_rot           = C_rot[10]

oE33_rot           = C_rot[11]
oE34_rot           = C_rot[12]
oE35_rot           = C_rot[13]
oE36_rot           = C_rot[14]

oE41_rot           = C_rot[15]
oE42_rot           = C_rot[16]
oE43_rot           = C_rot[17]
oE44_rot           = C_rot[18]
oE45_rot           = C_rot[19]
oE46_rot           = C_rot[20]

oE51_rot           = C_rot[21]
oE52_rot           = C_rot[22]
oE53_rot           = C_rot[23]
oE55_rot           = C_rot[24]
oE56_rot           = C_rot[25]

oE61_rot           = C_rot[26]
oE62_rot           = C_rot[27]
oE63_rot           = C_rot[28]
oE66_rot           = C_rot[29]
##----------------------------------------------------------------------------------------------------------------------
with open(os.path.join(results_path, 'Stiffness_Matrix_Components.f'), 'w') as output:
    output.write(f'      real*8 E11({8},{noEl})' '\n')
    output.write(f'      real*8 E12({8},{noEl})' '\n')
    output.write(f'      real*8 E13({8},{noEl})' '\n')
    output.write(f'      real*8 E14({8},{noEl})' '\n')
    output.write(f'      real*8 E15({8},{noEl})' '\n')
    output.write(f'      real*8 E16({8},{noEl})' '\n')
    output.write(f'      real*8 E22({8},{noEl})' '\n')
    output.write(f'      real*8 E23({8},{noEl})' '\n')
    output.write(f'      real*8 E24({8},{noEl})' '\n')
    output.write(f'      real*8 E25({8},{noEl})' '\n')
    output.write(f'      real*8 E26({8},{noEl})' '\n')
    output.write(f'      real*8 E33({8},{noEl})' '\n')
    output.write(f'      real*8 E34({8},{noEl})' '\n')
    output.write(f'      real*8 E35({8},{noEl})' '\n')
    output.write(f'      real*8 E36({8},{noEl})' '\n')
    output.write(f'      real*8 E41({8},{noEl})' '\n')
    output.write(f'      real*8 E42({8},{noEl})' '\n')
    output.write(f'      real*8 E43({8},{noEl})' '\n')
    output.write(f'      real*8 E44({8},{noEl})' '\n')
    output.write(f'      real*8 E45({8},{noEl})' '\n')
    output.write(f'      real*8 E46({8},{noEl})' '\n')
    output.write(f'      real*8 E51({8},{noEl})' '\n')
    output.write(f'      real*8 E52({8},{noEl})' '\n')
    output.write(f'      real*8 E53({8},{noEl})' '\n')
    output.write(f'      real*8 E55({8},{noEl})' '\n')
    output.write(f'      real*8 E56({8},{noEl})' '\n')
    output.write(f'      real*8 E61({8},{noEl})' '\n')
    output.write(f'      real*8 E62({8},{noEl})' '\n')
    output.write(f'      real*8 E63({8},{noEl})' '\n')
    output.write(f'      real*8 E66({8},{noEl})' '\n')
    output.write(f'      integer  :: I' '\n')
    for j in range(noEl):
        
        output.write(f'      DATA (E11(I,{j+1}), I=1,8)/ {nE11_rot[j,0]:.1f}, {nE11_rot[j,1]:.1f}, {nE11_rot[j,2]:.1f}, {nE11_rot[j,3]:.1f}, &' '\n')
        output.write(f'            {nE11_rot[j,4]:.1f}, {nE11_rot[j,5]:.1f}, {nE11_rot[j,6]:.1f}, {nE11_rot[j,7]:.1f}/' '\n')
        output.write(f'      DATA (E12(I,{j+1}), I=1,8)/ {nE12_rot[j,0]:.1f}, {nE12_rot[j,1]:.1f}, {nE12_rot[j,2]:.1f}, {nE12_rot[j,3]:.1f}, &' '\n')
        output.write(f'            {nE12_rot[j,4]:.1f}, {nE12_rot[j,5]:.1f}, {nE12_rot[j,6]:.1f}, {nE12_rot[j,7]:.1f}/' '\n')
        output.write(f'      DATA (E13(I,{j+1}), I=1,8)/ {nE13_rot[j,0]:.1f}, {nE13_rot[j,1]:.1f}, {nE13_rot[j,2]:.1f}, {nE13_rot[j,3]:.1f}, &' '\n')
        output.write(f'            {nE13_rot[j,4]:.1f}, {nE13_rot[j,5]:.1f}, {nE13_rot[j,6]:.1f}, {nE13_rot[j,7]:.1f}/' '\n')
        output.write(f'      DATA (E14(I,{j+1}), I=1,8)/ {nE14_rot[j,0]:.1f}, {nE14_rot[j,1]:.1f}, {nE14_rot[j,2]:.1f}, {nE14_rot[j,3]:.1f}, &' '\n')
        output.write(f'            {nE14_rot[j,4]:.1f}, {nE14_rot[j,5]:.1f}, {nE14_rot[j,6]:.1f}, {nE14_rot[j,7]:.1f}/' '\n')
        output.write(f'      DATA (E15(I,{j+1}), I=1,8)/ {nE15_rot[j,0]:.1f}, {nE15_rot[j,1]:.1f}, {nE15_rot[j,2]:.1f}, {nE15_rot[j,3]:.1f}, &' '\n')
        output.write(f'            {nE15_rot[j,4]:.1f}, {nE15_rot[j,5]:.1f}, {nE15_rot[j,6]:.1f}, {nE15_rot[j,7]:.1f}/' '\n')
        output.write(f'      DATA (E16(I,{j+1}), I=1,8)/ {nE16_rot[j,0]:.1f}, {nE16_rot[j,1]:.1f}, {nE16_rot[j,2]:.1f}, {nE16_rot[j,3]:.1f}, &' '\n')
        output.write(f'            {nE16_rot[j,4]:.1f}, {nE16_rot[j,5]:.1f}, {nE16_rot[j,6]:.1f}, {nE16_rot[j,7]:.1f}/' '\n')
#----------------------------------------------------------------------
        output.write(f'      DATA (E22(I,{j+1}), I=1,8)/ {nE22_rot[j,0]:.1f}, {nE22_rot[j,1]:.1f}, {nE22_rot[j,2]:.1f}, {nE22_rot[j,3]:.1f}, &' '\n')
        output.write(f'            {nE22_rot[j,4]:.1f}, {nE22_rot[j,5]:.1f}, {nE22_rot[j,6]:.1f}, {nE22_rot[j,7]:.1f}/' '\n')
        output.write(f'      DATA (E23(I,{j+1}), I=1,8)/ {nE23_rot[j,0]:.1f}, {nE23_rot[j,1]:.1f}, {nE23_rot[j,2]:.1f}, {nE23_rot[j,3]:.1f}, &' '\n')
        output.write(f'            {nE23_rot[j,4]:.1f}, {nE23_rot[j,5]:.1f}, {nE23_rot[j,6]:.1f}, {nE23_rot[j,7]:.1f}/' '\n')
        output.write(f'      DATA (E24(I,{j+1}), I=1,8)/ {nE24_rot[j,0]:.1f}, {nE24_rot[j,1]:.1f}, {nE24_rot[j,2]:.1f}, {nE24_rot[j,3]:.1f}, &' '\n')
        output.write(f'            {nE24_rot[j,4]:.1f}, {nE24_rot[j,5]:.1f}, {nE24_rot[j,6]:.1f}, {nE24_rot[j,7]:.1f}/' '\n')
        output.write(f'      DATA (E25(I,{j+1}), I=1,8)/ {nE25_rot[j,0]:.1f}, {nE25_rot[j,1]:.1f}, {nE25_rot[j,2]:.1f}, {nE25_rot[j,3]:.1f}, &' '\n')
        output.write(f'            {nE25_rot[j,4]:.1f}, {nE25_rot[j,5]:.1f}, {nE25_rot[j,6]:.1f}, {nE25_rot[j,7]:.1f}/' '\n')
        output.write(f'      DATA (E26(I,{j+1}), I=1,8)/ {nE26_rot[j,0]:.1f}, {nE26_rot[j,1]:.1f}, {nE26_rot[j,2]:.1f}, {nE26_rot[j,3]:.1f}, &' '\n')
        output.write(f'            {nE26_rot[j,4]:.1f}, {nE26_rot[j,5]:.1f}, {nE26_rot[j,6]:.1f}, {nE26_rot[j,7]:.1f}/' '\n')
#----------------------------------------------------------------------
        output.write(f'      DATA (E33(I,{j+1}), I=1,8)/ {nE33_rot[j,0]:.1f}, {nE33_rot[j,1]:.1f}, {nE33_rot[j,2]:.1f}, {nE33_rot[j,3]:.1f}, &' '\n')
        output.write(f'            {nE33_rot[j,4]:.1f}, {nE33_rot[j,5]:.1f}, {nE33_rot[j,6]:.1f}, {nE33_rot[j,7]:.1f}/' '\n')
        output.write(f'      DATA (E34(I,{j+1}), I=1,8)/ {nE34_rot[j,0]:.1f}, {nE34_rot[j,1]:.1f}, {nE34_rot[j,2]:.1f}, {nE34_rot[j,3]:.1f}, &' '\n')
        output.write(f'            {nE34_rot[j,4]:.1f}, {nE34_rot[j,5]:.1f}, {nE34_rot[j,6]:.1f}, {nE34_rot[j,7]:.1f}/' '\n')
        output.write(f'      DATA (E35(I,{j+1}), I=1,8)/ {nE35_rot[j,0]:.1f}, {nE35_rot[j,1]:.1f}, {nE35_rot[j,2]:.1f}, {nE35_rot[j,3]:.1f}, &' '\n')
        output.write(f'            {nE35_rot[j,4]:.1f}, {nE35_rot[j,5]:.1f}, {nE35_rot[j,6]:.1f}, {nE35_rot[j,7]:.1f}/' '\n')
        output.write(f'      DATA (E36(I,{j+1}), I=1,8)/ {nE36_rot[j,0]:.1f}, {nE36_rot[j,1]:.1f}, {nE36_rot[j,2]:.1f}, {nE36_rot[j,3]:.1f}, &' '\n')
        output.write(f'            {nE36_rot[j,4]:.1f}, {nE36_rot[j,5]:.1f}, {nE36_rot[j,6]:.1f}, {nE36_rot[j,7]:.1f}/' '\n')
#----------------------------------------------------------------------
        output.write(f'      DATA (E41(I,{j+1}), I=1,8)/ {nE41_rot[j,0]:.1f}, {nE41_rot[j,1]:.1f}, {nE41_rot[j,2]:.1f}, {nE41_rot[j,3]:.1f}, &' '\n')
        output.write(f'            {nE41_rot[j,4]:.1f}, {nE41_rot[j,5]:.1f}, {nE41_rot[j,6]:.1f}, {nE41_rot[j,7]:.1f}/' '\n')
        output.write(f'      DATA (E42(I,{j+1}), I=1,8)/ {nE42_rot[j,0]:.1f}, {nE42_rot[j,1]:.1f}, {nE42_rot[j,2]:.1f}, {nE42_rot[j,3]:.1f}, &' '\n')
        output.write(f'            {nE42_rot[j,4]:.1f}, {nE42_rot[j,5]:.1f}, {nE42_rot[j,6]:.1f}, {nE42_rot[j,7]:.1f}/' '\n')
        output.write(f'      DATA (E43(I,{j+1}), I=1,8)/ {nE43_rot[j,0]:.1f}, {nE43_rot[j,1]:.1f}, {nE43_rot[j,2]:.1f}, {nE43_rot[j,3]:.1f}, &' '\n')
        output.write(f'            {nE43_rot[j,4]:.1f}, {nE43_rot[j,5]:.1f}, {nE43_rot[j,6]:.1f}, {nE43_rot[j,7]:.1f}/' '\n')
        output.write(f'      DATA (E44(I,{j+1}), I=1,8)/ {nE44_rot[j,0]:.1f}, {nE44_rot[j,1]:.1f}, {nE44_rot[j,2]:.1f}, {nE44_rot[j,3]:.1f}, &' '\n')
        output.write(f'            {nE44_rot[j,4]:.1f}, {nE44_rot[j,5]:.1f}, {nE44_rot[j,6]:.1f}, {nE44_rot[j,7]:.1f}/' '\n')
        output.write(f'      DATA (E45(I,{j+1}), I=1,8)/ {nE45_rot[j,0]:.1f}, {nE45_rot[j,1]:.1f}, {nE45_rot[j,2]:.1f}, {nE45_rot[j,3]:.1f}, &' '\n')
        output.write(f'            {nE45_rot[j,4]:.1f}, {nE45_rot[j,5]:.1f}, {nE45_rot[j,6]:.1f}, {nE45_rot[j,7]:.1f}/' '\n')
        output.write(f'      DATA (E46(I,{j+1}), I=1,8)/ {nE46_rot[j,0]:.1f}, {nE46_rot[j,1]:.1f}, {nE46_rot[j,2]:.1f}, {nE46_rot[j,3]:.1f}, &' '\n')
        output.write(f'            {nE46_rot[j,4]:.1f}, {nE46_rot[j,5]:.1f}, {nE46_rot[j,6]:.1f}, {nE46_rot[j,7]:.1f}/' '\n')
#----------------------------------------------------------------------
        output.write(f'      DATA (E51(I,{j+1}), I=1,8)/ {nE51_rot[j,0]:.1f}, {nE51_rot[j,1]:.1f}, {nE51_rot[j,2]:.1f}, {nE51_rot[j,3]:.1f}, &' '\n')
        output.write(f'            {nE51_rot[j,4]:.1f}, {nE51_rot[j,5]:.1f}, {nE51_rot[j,6]:.1f}, {nE51_rot[j,7]:.1f}/' '\n')
        output.write(f'      DATA (E52(I,{j+1}), I=1,8)/ {nE52_rot[j,0]:.1f}, {nE52_rot[j,1]:.1f}, {nE52_rot[j,2]:.1f}, {nE52_rot[j,3]:.1f}, &' '\n')
        output.write(f'            {nE52_rot[j,4]:.1f}, {nE52_rot[j,5]:.1f}, {nE52_rot[j,6]:.1f}, {nE52_rot[j,7]:.1f}/' '\n')
        output.write(f'      DATA (E53(I,{j+1}), I=1,8)/ {nE53_rot[j,0]:.1f}, {nE53_rot[j,1]:.1f}, {nE53_rot[j,2]:.1f}, {nE53_rot[j,3]:.1f}, &' '\n')
        output.write(f'            {nE53_rot[j,4]:.1f}, {nE53_rot[j,5]:.1f}, {nE53_rot[j,6]:.1f}, {nE53_rot[j,7]:.1f}/' '\n')
        output.write(f'      DATA (E55(I,{j+1}), I=1,8)/ {nE55_rot[j,0]:.1f}, {nE55_rot[j,1]:.1f}, {nE55_rot[j,2]:.1f}, {nE55_rot[j,3]:.1f}, &' '\n')
        output.write(f'            {nE55_rot[j,4]:.1f}, {nE55_rot[j,5]:.1f}, {nE55_rot[j,6]:.1f}, {nE55_rot[j,7]:.1f}/' '\n')
        output.write(f'      DATA (E56(I,{j+1}), I=1,8)/ {nE56_rot[j,0]:.1f}, {nE56_rot[j,1]:.1f}, {nE56_rot[j,2]:.1f}, {nE56_rot[j,3]:.1f}, &' '\n')
        output.write(f'            {nE56_rot[j,4]:.1f}, {nE56_rot[j,5]:.1f}, {nE56_rot[j,6]:.1f}, {nE56_rot[j,7]:.1f}/' '\n')
#----------------------------------------------------------------------
        output.write(f'      DATA (E61(I,{j+1}), I=1,8)/ {nE61_rot[j,0]:.1f}, {nE61_rot[j,1]:.1f}, {nE61_rot[j,2]:.1f}, {nE61_rot[j,3]:.1f}, &' '\n')
        output.write(f'            {nE61_rot[j,4]:.1f}, {nE61_rot[j,5]:.1f}, {nE61_rot[j,6]:.1f}, {nE61_rot[j,7]:.1f}/' '\n')
        output.write(f'      DATA (E62(I,{j+1}), I=1,8)/ {nE62_rot[j,0]:.1f}, {nE62_rot[j,1]:.1f}, {nE62_rot[j,2]:.1f}, {nE62_rot[j,3]:.1f}, &' '\n')
        output.write(f'            {nE62_rot[j,4]:.1f}, {nE62_rot[j,5]:.1f}, {nE62_rot[j,6]:.1f}, {nE62_rot[j,7]:.1f}/' '\n')
        output.write(f'      DATA (E63(I,{j+1}), I=1,8)/ {nE63_rot[j,0]:.1f}, {nE63_rot[j,1]:.1f}, {nE63_rot[j,2]:.1f}, {nE63_rot[j,3]:.1f}, &' '\n')
        output.write(f'            {nE63_rot[j,4]:.1f}, {nE63_rot[j,5]:.1f}, {nE63_rot[j,6]:.1f}, {nE63_rot[j,7]:.1f}/' '\n')
        output.write(f'      DATA (E66(I,{j+1}), I=1,8)/ {nE66_rot[j,0]:.1f}, {nE66_rot[j,1]:.1f}, {nE66_rot[j,2]:.1f}, {nE66_rot[j,3]:.1f}, &' '\n')
        output.write(f'            {nE66_rot[j,4]:.1f}, {nE66_rot[j,5]:.1f}, {nE66_rot[j,6]:.1f}, {nE66_rot[j,7]:.1f}/' '\n')
print('Output done')
##----------------------------------------------------------------------------------------------------------------------
#%% Print plots
fig_theme='default'
save_tif='True'
save_svg='True'
#%% Comparison E11 
plot_compare_E11_histogram(nE11_rot, oE11_rot, script_name,results_path,fig_theme,save_tif,save_svg)
#%% Comparison E22 
plot_compare_E22_histogram(nE22_rot, oE22_rot, script_name,results_path,fig_theme,save_tif,save_svg)
#%% Comparison E33 
plot_compare_E33_histogram(nE33_rot, oE33_rot, script_name,results_path,fig_theme,save_tif,save_svg) 
#%% 
end = time.time()
print("The computation time was:", end-start)
