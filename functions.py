# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 12:12:47 2022

@author: robaue
"""
from numba import njit
import numpy as np
from scipy.integrate import nquad
#===========================================================================================================
def Mori_Tanaka(Vf_mean, Em, num, Gm, C11_f, C22_f, C12_f, C13_f, C66_f, a1, a2, a3, notebook_name):


				
    # Stiffness matrix fibre
    C_f= np.zeros((6, 6))
    C_f[0,0] =  C11_f
    C_f[1,1] =  C22_f
    C_f[2,2] =  C22_f
    C_f[0,1] =  C12_f
    C_f[0,2] =  C12_f
    C_f[1,0] =  C12_f
    C_f[1,2] =  C13_f
    C_f[2,0] =  C12_f
    C_f[2,1] =  C13_f
    C_f[3,3] =  (C22_f-C13_f)/2
    C_f[4,4] =  C66_f
    C_f[5,5] =  C66_f

    np.save(f'001_C_f_{notebook_name}.npy', C_f)
    #=========================================================================
    # Transformation Stiffness matrix to tensor
    #=========================================================================
    #%%
    Cf=np.zeros((3,3,3,3))

    # First row
    Cf[0,0,0,0] = C_f[0,0]
    Cf[0,0,1,1] = C_f[0,1]
    Cf[0,0,2,2] = C_f[0,2]
    Cf[0,0,1,2] = C_f[0,3]
    Cf[0,0,0,2] = C_f[0,4]
    Cf[0,0,0,1] = C_f[0,5]

    # Second row
    Cf[1,1,0,0] = C_f[1,0]
    Cf[1,1,1,1] = C_f[1,1]
    Cf[1,1,2,2] = C_f[1,2]
    Cf[1,1,1,2] = C_f[1,3]
    Cf[1,1,0,2] = C_f[1,4]
    Cf[1,1,0,1] = C_f[1,5]

    # Third row
    Cf[2,2,0,0] = C_f[2,0]
    Cf[2,2,1,1] = C_f[2,1]
    Cf[2,2,2,2] = C_f[2,2]
    Cf[2,2,1,2] = C_f[2,3]
    Cf[2,2,0,2] = C_f[2,4]
    Cf[2,2,0,1] = C_f[2,5]

    # Forth row
    Cf[1,2,0,0] = C_f[3,0]
    Cf[1,2,1,1] = C_f[3,1]
    Cf[1,2,2,2] = C_f[3,2]
    Cf[1,2,1,2] = C_f[3,3]
    Cf[1,2,0,2] = C_f[3,4]
    Cf[1,2,0,1] = C_f[3,5]

    # Fifth row
    Cf[0,2,0,0] = C_f[4,0]
    Cf[0,2,1,1] = C_f[4,1]
    Cf[0,2,2,2] = C_f[4,2]
    Cf[0,2,1,2] = C_f[4,3]
    Cf[0,2,0,2] = C_f[4,4]
    Cf[0,2,0,1] = C_f[4,5]

    # Sixth row
    Cf[0,1,0,0] = C_f[5,0]
    Cf[0,1,1,1] = C_f[5,1]
    Cf[0,1,2,2] = C_f[5,2]
    Cf[0,1,1,2] = C_f[5,3]
    Cf[0,1,0,2] = C_f[5,4]
    Cf[0,1,0,1] = C_f[5,5]
    #=========================================================================
    # Implementation of Symmetry
    #=========================================================================
    C=np.zeros((3,3,3,3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    if Cf[i,j,k,l] > 0:
                        C[i,j,k,l] = Cf[i,j,k,l]
        
                        # Minor symmetry
                        C[i,j,l,k] = Cf[i,j,k,l]    
                        C[j,i,l,k] = Cf[i,j,k,l]    
                        C[j,i,k,l] = Cf[i,j,k,l]
        
                        # Major symmetry                
                        C[k,l,i,j] = Cf[i,j,k,l]
    #=========================================================================
    # Numerical integration
    #=========================================================================
    def Integration(ii,jj,kk,ll):
        ranges=((0, 2*np.pi), (0, np.pi))
        @njit(nogil=True)
        def Integral(phi, theta):
            z=np.array((np.cos(phi)*np.sin(theta),     np.sin(phi)*np.sin(theta),     np.cos(theta)))
            beta=np.sqrt((a1**2*np.cos(phi)**2+a2**2*np.sin(phi)**2)*np.sin(theta)**2+a3**2*np.cos(theta)**2)
            T=np.zeros((3,3))
            for k in range(3):
                for i in range(3):
                    for p in range(3):
                        for m in range(3):
                            T[k,i] = T[k,i] + C[p,k,i,m]*z[p]*z[m]
            invT=np.linalg.inv(T)
            invT=np.linalg.inv(T)
            D = invT[ii,jj]*z[kk]*z[ll] * np.sin(theta)/beta**3
            return D
        
        K = nquad(Integral,ranges)
        return K
    #=========================================================================
    # Calculation Eshelby tensor
    #=========================================================================
    D = np.zeros((3,3,3,3))
    for ii in range(3):
        for jj in range(3):
            for kk in range(3):
                for ll in range(3):
                    J = Integration(ii,jj,kk,ll)
                    D[ii,jj,kk,ll] = -a1*a2*a3/(4*np.pi)*J[0]
    #%%
    S = np.zeros((3,3,3,3))
    for i in range(3):
        for j in range(3):
            for m in range(3):
                for n in range(3):
                    for k in range(3):
                        for l in range(3):
                            #orthoE[i,j,m,n]
                            S[i,j,m,n] = -0.5  * C[l,k,m,n] * (D[i,k,l,j] + D[j,k,l,i]) + S[i,j,m,n] 

    #=========================================================================
    # Transformation Eshelby tensor to matrix
    #=========================================================================
    Eshelby = np.zeros((6,6))

    # First row
    Eshelby[0,0] = S[0,0,0,0]
    Eshelby[0,1] = S[0,0,1,1]
    Eshelby[0,2] = S[0,0,2,2]
    Eshelby[0,3] = S[0,0,1,2]
    Eshelby[0,4] = S[0,0,0,2]
    Eshelby[0,5] = S[0,0,0,1]

    # Second row
    Eshelby[1,0] = S[1,1,0,0]
    Eshelby[1,1] = S[1,1,1,1]
    Eshelby[1,2] = S[1,1,2,2]
    Eshelby[1,3] = S[1,1,1,2]
    Eshelby[1,4] = S[1,1,0,2]
    Eshelby[1,5] = S[1,1,0,1]

    # Third row
    Eshelby[2,0] = S[2,2,0,0]
    Eshelby[2,1] = S[2,2,1,1]
    Eshelby[2,2] = S[2,2,2,2]
    Eshelby[2,3] = S[2,2,1,2]
    Eshelby[2,4] = S[2,2,0,2]
    Eshelby[2,5] = S[2,2,0,1]

    # Forth row
    Eshelby[3,0] = S[1,2,0,0]
    Eshelby[3,1] = S[1,2,1,1]
    Eshelby[3,2] = S[1,2,2,2]
    Eshelby[3,3] = S[1,2,1,2]
    Eshelby[3,4] = S[1,2,0,2]
    Eshelby[3,5] = S[1,2,0,1]

    # Fifth row
    Eshelby[4,0] = S[0,2,0,0]
    Eshelby[4,1] = S[0,2,1,1]
    Eshelby[4,2] = S[0,2,2,2]
    Eshelby[4,3] = S[0,2,1,2]
    Eshelby[4,4] = S[0,2,0,2]
    Eshelby[4,5] = S[0,2,0,1]

    # Sixth row
    Eshelby[5,0] = S[0,1,0,0]
    Eshelby[5,1] = S[0,1,1,1]
    Eshelby[5,2] = S[0,1,2,2]
    Eshelby[5,3] = S[0,1,1,2]
    Eshelby[5,4] = S[0,1,0,2]
    Eshelby[5,5] = S[0,1,0,1]
    
    np.save(f'001_Eshelby_{notebook_name}.npy', Eshelby)
    
    #=========================================================================
    # Define Matrix isotropic stiffness TENSOR
    #=========================================================================
    S_m=0
    S_m= np.zeros((6, 6))
    S_m[0,0] =  1/Em
    S_m[1,1] =  1/Em
    S_m[2,2] =  1/Em
    S_m[1,0] =  -num/Em
    S_m[2,0] =  -num/Em
    S_m[2,1] =  -num/Em
    S_m[0,1] =  -num/Em
    S_m[0,2] =  -num/Em
    S_m[1,2] =  -num/Em
    S_m[3,3] =  1/Gm
    S_m[4,4] =  1/Gm
    S_m[5,5] =  1/Gm
    C_m=np.zeros([6, 6])
    C_m=np.linalg.inv(S_m)
    np.save(f'001_C_m_{notebook_name}.npy', C_m)
    #=========================================================================
    # Calculate Mori Tanaka TENSOR
    #=========================================================================
    # Equations follow Raju et al. 2021
    II   = np.identity(6)
    E_MT = np.zeros((6, 6))
    E_MT = np.matmul(S_m,C_m-C_f)
    B_MT = np.zeros((6, 6))
    B_MT = II-np.matmul(Eshelby,E_MT)
    
    np.save(f'001_B_MT_{notebook_name}.npy',B_MT)
    
    A_MT = np.zeros((6, 6))
    A_MT = np.linalg.inv(B_MT)
    
    np.save(f'001_A_MT_{notebook_name}.npy', A_MT)
    
    D_MT = np.zeros((6, 6))
    D_MT = np.linalg.inv((1-Vf_mean)*II+Vf_mean*A_MT)
    B_MT = np.matmul(A_MT, D_MT)
    C_MT = C_m + np.matmul(Vf_mean*(C_f-C_m),B_MT)
    #print('C_MT', C_MT)
    C_MT_11=C_MT[0,0]
    C_MT_22=C_MT[1,1]
    C_MT_66=C_MT[5,5]
    C_MT_12=C_MT[0,1]
    C_MT_23=C_MT[1,2]
    # print('C_MT_11', C_MT_11)
    # print('C_MT_22', C_MT_22)
    # print('C_MT_66', C_MT_66)
    # print('C_MT_12', C_MT_12)
    # print('C_MT_23', C_MT_23)
    return C_MT_11, C_MT_22, C_MT_66, C_MT_12, C_MT_23
##-----------------------------------------------------------------------------------------
@njit(nogil=True, parallel=True)
def E__update(phi_scat, theta_scat, C_11, C_22, C_66, C_12, C_23):

    phi_sct_steps=len(phi_scat)
    theta_sct_steps=len(theta_scat)
    E11_arr=np.zeros((phi_sct_steps,theta_sct_steps))
    E12_arr=np.zeros((phi_sct_steps,theta_sct_steps))
    E13_arr=np.zeros((phi_sct_steps,theta_sct_steps))
    E14_arr=np.zeros((phi_sct_steps,theta_sct_steps))
    E15_arr=np.zeros((phi_sct_steps,theta_sct_steps))
    E16_arr=np.zeros((phi_sct_steps,theta_sct_steps))
    E22_arr=np.zeros((phi_sct_steps,theta_sct_steps))
    E23_arr=np.zeros((phi_sct_steps,theta_sct_steps))
    E24_arr=np.zeros((phi_sct_steps,theta_sct_steps))
    E25_arr=np.zeros((phi_sct_steps,theta_sct_steps))
    E26_arr=np.zeros((phi_sct_steps,theta_sct_steps))
    E33_arr=np.zeros((phi_sct_steps,theta_sct_steps))
    E34_arr=np.zeros((phi_sct_steps,theta_sct_steps))
    E35_arr=np.zeros((phi_sct_steps,theta_sct_steps))
    E36_arr=np.zeros((phi_sct_steps,theta_sct_steps))
    E41_arr=np.zeros((phi_sct_steps,theta_sct_steps))
    E42_arr=np.zeros((phi_sct_steps,theta_sct_steps))
    E43_arr=np.zeros((phi_sct_steps,theta_sct_steps))
    E44_arr=np.zeros((phi_sct_steps,theta_sct_steps))
    E45_arr=np.zeros((phi_sct_steps,theta_sct_steps))
    E46_arr=np.zeros((phi_sct_steps,theta_sct_steps))
    E51_arr=np.zeros((phi_sct_steps,theta_sct_steps))
    E52_arr=np.zeros((phi_sct_steps,theta_sct_steps))
    E53_arr=np.zeros((phi_sct_steps,theta_sct_steps))
    E55_arr=np.zeros((phi_sct_steps,theta_sct_steps))
    E56_arr=np.zeros((phi_sct_steps,theta_sct_steps))
    E61_arr=np.zeros((phi_sct_steps,theta_sct_steps))
    E62_arr=np.zeros((phi_sct_steps,theta_sct_steps))
    E63_arr=np.zeros((phi_sct_steps,theta_sct_steps))
    E66_arr=np.zeros((phi_sct_steps,theta_sct_steps)) 
    for i in range(phi_sct_steps):
        phi=phi_scat[i]
		
		# phi sin
        sp   = np.sin(phi)
        sp2  = np.sin(phi)**2
        sp4  = np.sin(phi)**4
        # phi cos
        cp   = np.cos(phi)
        cp2  = np.cos(phi)**2
        cp4  = np.cos(phi)**4
        for j in range(theta_sct_steps):
            theta=theta_scat[j]
        
            # theta sin
            st  = np.sin(theta)
            st2 = np.sin(theta)**2
            st4 = np.sin(theta)**4
            # theta cos
            ct  = np.cos(theta)
            ct2 = np.cos(theta)**2
            ct4 = np.cos(theta)**4
            c4t = np.cos(4*theta)

            E11_arr[i,j] = C_22 + 2*C_12*st2 + C_11*st4 - 2*C_12*st4 - 2*C_22*st2 + C_22*st4 + 2*C_66*st2 - 2*C_66*st4 - 2*C_12*sp2*st2 - 2*C_11*sp2*st4 + 4*C_12*sp2*st4 + C_11*sp4*st4 - 2*C_12*sp4*st4 + C_22*sp2*st2 + C_23*sp2*st2 - C_22*sp2*st4 - C_23*sp2*st4 + C_22*sp4*st4 - 2*C_66*sp2*st2 + 4*C_66*sp2*st4 - 2*C_66*sp4*st4
            E12_arr[i,j] = C_23 + C_12*sp2 - C_23*sp2 + C_12*st2 - C_23*st2 + C_11*sp2*st2 - 3*C_12*sp2*st2 - C_11*sp4*st2 + 2*C_12*sp4*st2 + C_22*sp2*st2 + C_23*sp2*st2 - C_22*sp4*st2 - 2*C_66*sp2*st2 + 2*C_66*sp4*st2
            E13_arr[i,j] = C_12 - C_12*sp2 + C_23*sp2 + C_11*st2 - 2*C_12*st2 - C_11*st4 + 2*C_12*st4 + C_22*st2 - C_22*st4 - 2*C_66*st2 + 2*C_66*st4 - 2*C_11*sp2*st2 + 4*C_12*sp2*st2 + 2*C_11*sp2*st4 + C_11*sp4*st2 - 4*C_12*sp2*st4 - 2*C_12*sp4*st2 - C_11*sp4*st4 + 2*C_12*sp4*st4 - C_22*sp2*st2 - C_23*sp2*st2 + C_22*sp2*st4 + C_22*sp4*st2 + C_23*sp2*st4 - C_22*sp4*st4 + 4*C_66*sp2*st2 - 4*C_66*sp2*st4 - 2*C_66*sp4*st2 + 2*C_66*sp4*st4
            E14_arr[i,j] = -cp*ct*sp*(C_22 - 2*C_12 + C_23 - 2*C_11*cp2 + 4*C_12*cp2 - 2*C_22*cp2 + 4*C_66*cp2 - C_22*ct2 + C_23*ct2 + 2*C_11*cp2*ct2 - 4*C_12*cp2*ct2 + 2*C_22*cp2*ct2 - 4*C_66*cp2*ct2)
            E15_arr[i,j] = -ct*st*(C_22 - C_23 + 2*C_12*cp2 + 2*C_11*cp4 - 4*C_12*cp4 - 3*C_22*cp2 + C_23*cp2 + 2*C_22*cp4 + 2*C_66*cp2 - 4*C_66*cp4 - 2*C_22*ct2 + 2*C_23*ct2 - 2*C_11*cp4*ct2 + 4*C_12*cp4*ct2 + 2*C_22*cp2*ct2 - 2*C_23*cp2*ct2 - 2*C_22*cp4*ct2 + 4*C_66*cp4*ct2)
            E16_arr[i,j] = -cp*sp*st*(2*C_12 - 2*C_22 + 2*C_66 + 2*C_11*cp2 - 4*C_12*cp2 + 2*C_22*cp2 - 4*C_66*cp2 + C_22*ct2 - C_23*ct2 - 2*C_11*cp2*ct2 + 4*C_12*cp2*ct2 - 2*C_22*cp2*ct2 + 4*C_66*cp2*ct2)

            # E2
            # E21_arr[i,j] = symmetric
            E22_arr[i,j] = C_22 + 2*C_12*sp2 + C_11*sp4 - 2*C_12*sp4 - 2*C_22*sp2 + C_22*sp4 + 2*C_66*sp2 - 2*C_66*sp4
            E23_arr[i,j] = C_12 + C_11*sp2 - 2*C_12*sp2 - C_11*sp4 + 2*C_12*sp4 + C_22*sp2 - C_22*sp4 - 2*C_66*sp2 + 2*C_66*sp4 - C_12*st2 + C_23*st2 - C_11*sp2*st2 + 3*C_12*sp2*st2 + C_11*sp4*st2 - 2*C_12*sp4*st2 - C_22*sp2*st2 - C_23*sp2*st2 + C_22*sp4*st2 + 2*C_66*sp2*st2 - 2*C_66*sp4*st2
            E24_arr[i,j] = -2*cp*ct*sp*(C_12 - C_11 + C_66 + C_11*cp2 - 2*C_12*cp2 + C_22*cp2 - 2*C_66*cp2)
            E25_arr[i,j] = 2*cp2*ct*st*(C_12 - C_11 - C_22 + C_23 + 2*C_66 + C_11*cp2 - 2*C_12*cp2 + C_22*cp2 - 2*C_66*cp2)
            E26_arr[i,j] = 2*cp*sp*st*(C_12 - C_11 + C_66 + C_11*cp2 - 2*C_12*cp2 + C_22*cp2 - 2*C_66*cp2)

            # E3
            # E31_arr[i,j] = symmetric
            # E32_arr[i,j] = symmetric
            E33_arr[i,j] = C_11 - 2*C_11*sp2 + 2*C_12*sp2 + C_11*sp4 - 2*C_12*sp4 + C_22*sp4 + 2*C_66*sp2 - 2*C_66*sp4 - 2*C_11*st2 + 2*C_12*st2 + C_11*st4 - 2*C_12*st4 + C_22*st4 + 2*C_66*st2 - 2*C_66*st4 + 4*C_11*sp2*st2 - 6*C_12*sp2*st2 - 2*C_11*sp2*st4 - 2*C_11*sp4*st2 + 4*C_12*sp2*st4 + 4*C_12*sp4*st2 + C_11*sp4*st4 - 2*C_12*sp4*st4 + C_22*sp2*st2 + C_23*sp2*st2 - C_22*sp2*st4 - 2*C_22*sp4*st2 - C_23*sp2*st4 + C_22*sp4*st4 - 6*C_66*sp2*st2 + 4*C_66*sp2*st4 + 4*C_66*sp4*st2 - 2*C_66*sp4*st4
            E34_arr[i,j] = cp*ct*sp*(2*C_12 - C_22 - C_23 + 2*C_66 - C_22*ct2 + C_23*ct2 + 2*C_11*cp2*ct2 - 4*C_12*cp2*ct2 + 2*C_22*cp2*ct2 - 4*C_66*cp2*ct2)
            E35_arr[i,j] = ct*st*(C_22 - C_23 - 2*C_12*cp2 + C_22*cp2 + C_23*cp2 - 2*C_66*cp2 - 2*C_22*ct2 + 2*C_23*ct2 - 2*C_11*cp4*ct2 + 4*C_12*cp4*ct2 + 2*C_22*cp2*ct2 - 2*C_23*cp2*ct2 - 2*C_22*cp4*ct2 + 4*C_66*cp4*ct2)
            E36_arr[i,j] = -cp*sp*st*(2*C_12 - 2*C_23 - C_22*ct2 + C_23*ct2 + 2*C_11*cp2*ct2 - 4*C_12*cp2*ct2 + 2*C_22*cp2*ct2 - 4*C_66*cp2*ct2)

            # E4
            E41_arr[i,j] = -(cp*ct*sp*(C_22 - 2*C_12 + C_23 - 2*C_11*cp2 + 4*C_12*cp2 - 2*C_22*cp2 + 4*C_66*cp2 - C_22*ct2 + C_23*ct2 + 2*C_11*cp2*ct2 - 4*C_12*cp2*ct2 + 2*C_22*cp2*ct2 - 4*C_66*cp2*ct2))/2
            E42_arr[i,j] = -cp*ct*sp*(C_12 - C_11 + C_66 + C_11*cp2 - 2*C_12*cp2 + C_22*cp2 - 2*C_66*cp2)
            E43_arr[i,j] = (cp*ct*sp*(2*C_12 - C_22 - C_23 + 2*C_66 - C_22*ct2 + C_23*ct2 + 2*C_11*cp2*ct2 - 4*C_12*cp2*ct2 + 2*C_22*cp2*ct2 - 4*C_66*cp2*ct2))/2
            E44_arr[i,j] = C_66*ct2 + C_66*st2 - 4*C_66*cp2*ct2 + 4*C_66*cp4*ct2 + (C_22*cp2*st2)/2 - (C_23*cp2*st2)/2 - C_66*cp2*st2 + 2*C_11*cp2*ct2*sp2 - 4*C_12*cp2*ct2*sp2 + 2*C_22*cp2*ct2*sp2
            E45_arr[i,j] = -(cp*sp*st*(C_23 - C_22 + 2*C_66 - 2*C_22*ct2 + 2*C_23*ct2 + 4*C_11*cp2*ct2 - 8*C_12*cp2*ct2 + 4*C_22*cp2*ct2 - 8*C_66*cp2*ct2))/2
            E46_arr[i,j] = -(cp2*ct*st*(C_23 - C_22 + 2*C_66 + 4*C_11*sp2 - 8*C_12*sp2 + 4*C_22*sp2 - 8*C_66*sp2))/2

            # E5
            E51_arr[i,j] = -(ct*st*(C_22 - C_23 + 2*C_12*cp2 + 2*C_11*cp4 - 4*C_12*cp4 - 3*C_22*cp2 + C_23*cp2 + 2*C_22*cp4 + 2*C_66*cp2 - 4*C_66*cp4 - 2*C_22*ct2 + 2*C_23*ct2 - 2*C_11*cp4*ct2 + 4*C_12*cp4*ct2 + 2*C_22*cp2*ct2 - 2*C_23*cp2*ct2 - 2*C_22*cp4*ct2 + 4*C_66*cp4*ct2))/2
            E52_arr[i,j] = cp2*ct*st*(C_12 - C_11 - C_22 + C_23 + 2*C_66 + C_11*cp2 - 2*C_12*cp2 + C_22*cp2 - 2*C_66*cp2)
            E53_arr[i,j] = (ct*st*(C_22 - C_23 - 2*C_12*cp2 + C_22*cp2 + C_23*cp2 - 2*C_66*cp2 - 2*C_22*ct2 + 2*C_23*ct2 - 2*C_11*cp4*ct2 + 4*C_12*cp4*ct2 + 2*C_22*cp2*ct2 - 2*C_23*cp2*ct2 - 2*C_22*cp4*ct2 + 4*C_66*cp4*ct2))/2
            # E54_arr[i,j] = symmetric
            E55_arr[i,j] = C_22/2 - C_23/2 - (C_22*c4t)/2 + (C_23*c4t)/2 + 3*C_66*cp2*ct2 - 2*C_66*cp2*ct4 - 4*C_66*cp4*ct2 + 4*C_66*cp4*ct4 - (C_22*ct2*sp2)/2 + (C_23*ct2*sp2)/2 + C_22*ct4*sp2 - C_23*ct4*sp2 + C_66*cp2*st2 - (C_22*sp2*st2)/2 + (C_23*sp2*st2)/2 + C_22*sp2*st4 - C_23*sp2*st4 + 2*C_11*cp4*ct2*st2 - 4*C_12*cp4*ct2*st2 - 4*C_22*cp2*ct2*st2 + 4*C_23*cp2*ct2*st2 + 2*C_22*cp4*ct2*st2 - 2*C_66*cp2*ct2*st2
            E56_arr[i,j] = (cp*ct*sp*(3*C_23 - 3*C_22 + 2*C_66 + 4*C_11*cp2 - 8*C_12*cp2 + 4*C_22*cp2 - 8*C_66*cp2 + 2*C_22*ct2 - 2*C_23*ct2 - 4*C_11*cp2*ct2 + 8*C_12*cp2*ct2 - 4*C_22*cp2*ct2 + 8*C_66*cp2*ct2))/2
            # E6
            E61_arr[i,j] = -(cp*sp*st*(2*C_12 - 2*C_22 + 2*C_66 + 2*C_11*cp2 - 4*C_12*cp2 + 2*C_22*cp2 - 4*C_66*cp2 + C_22*ct2 - C_23*ct2 - 2*C_11*cp2*ct2 + 4*C_12*cp2*ct2 - 2*C_22*cp2*ct2 + 4*C_66*cp2*ct2))/2
            E62_arr[i,j] = cp*sp*st*(C_12 - C_11 + C_66 + C_11*cp2 - 2*C_12*cp2 + C_22*cp2 - 2*C_66*cp2)
            E63_arr[i,j] = -(cp*sp*st*(2*C_12 - 2*C_23 - C_22*ct2 + C_23*ct2 + 2*C_11*cp2*ct2 - 4*C_12*cp2*ct2 + 2*C_22*cp2*ct2 - 4*C_66*cp2*ct2))/2
            # E64_arr[i,j] = symmetric
            # E65_arr[i,j] = symmetric
            E66_arr[i,j] = C_66*ct2 + C_66*st2 + (C_22*cp2*ct2)/2 - (C_23*cp2*ct2)/2 - C_66*cp2*ct2 - 4*C_66*cp2*st2 + 4*C_66*cp4*st2 + 2*C_11*cp2*sp2*st2 - 4*C_12*cp2*sp2*st2 + 2*C_22*cp2*sp2*st2

    return E11_arr, E12_arr, E13_arr, E14_arr, E15_arr, E16_arr, E22_arr, E23_arr, E24_arr, E25_arr, E26_arr, E33_arr, E34_arr, E35_arr, E36_arr, E41_arr, E42_arr, E43_arr, E44_arr, E45_arr, E46_arr, E51_arr, E52_arr, E53_arr, E55_arr, E56_arr, E61_arr, E62_arr, E63_arr, E66_arr
##------------------------------------------------------------------------------------------
@njit(nogil=True)
def E__(phi, theta, C_11, C_22, C_66, C_12, C_23):
    
    # phi sin
    sp   = np.sin(phi)
    sp2  = np.sin(phi)**2
    sp4  = np.sin(phi)**4
    # phi cos
    cp   = np.cos(phi)
    cp2  = np.cos(phi)**2
    cp4  = np.cos(phi)**4
        
    # theta sin
    st  = np.sin(theta)
    st2 = np.sin(theta)**2
    st4 = np.sin(theta)**4
    # theta cos
    ct  = np.cos(theta)
    ct2 = np.cos(theta)**2
    ct4 = np.cos(theta)**4
    c4t = np.cos(4*theta)

    E11 = C_22 + 2*C_12*st2 + C_11*st4 - 2*C_12*st4 - 2*C_22*st2 + C_22*st4 + 2*C_66*st2 - 2*C_66*st4 - 2*C_12*sp2*st2 - 2*C_11*sp2*st4 + 4*C_12*sp2*st4 + C_11*sp4*st4 - 2*C_12*sp4*st4 + C_22*sp2*st2 + C_23*sp2*st2 - C_22*sp2*st4 - C_23*sp2*st4 + C_22*sp4*st4 - 2*C_66*sp2*st2 + 4*C_66*sp2*st4 - 2*C_66*sp4*st4
    E12 = C_23 + C_12*sp2 - C_23*sp2 + C_12*st2 - C_23*st2 + C_11*sp2*st2 - 3*C_12*sp2*st2 - C_11*sp4*st2 + 2*C_12*sp4*st2 + C_22*sp2*st2 + C_23*sp2*st2 - C_22*sp4*st2 - 2*C_66*sp2*st2 + 2*C_66*sp4*st2
    E13 = C_12 - C_12*sp2 + C_23*sp2 + C_11*st2 - 2*C_12*st2 - C_11*st4 + 2*C_12*st4 + C_22*st2 - C_22*st4 - 2*C_66*st2 + 2*C_66*st4 - 2*C_11*sp2*st2 + 4*C_12*sp2*st2 + 2*C_11*sp2*st4 + C_11*sp4*st2 - 4*C_12*sp2*st4 - 2*C_12*sp4*st2 - C_11*sp4*st4 + 2*C_12*sp4*st4 - C_22*sp2*st2 - C_23*sp2*st2 + C_22*sp2*st4 + C_22*sp4*st2 + C_23*sp2*st4 - C_22*sp4*st4 + 4*C_66*sp2*st2 - 4*C_66*sp2*st4 - 2*C_66*sp4*st2 + 2*C_66*sp4*st4
    E14 = -cp*ct*sp*(C_22 - 2*C_12 + C_23 - 2*C_11*cp2 + 4*C_12*cp2 - 2*C_22*cp2 + 4*C_66*cp2 - C_22*ct2 + C_23*ct2 + 2*C_11*cp2*ct2 - 4*C_12*cp2*ct2 + 2*C_22*cp2*ct2 - 4*C_66*cp2*ct2)
    E15 = -ct*st*(C_22 - C_23 + 2*C_12*cp2 + 2*C_11*cp4 - 4*C_12*cp4 - 3*C_22*cp2 + C_23*cp2 + 2*C_22*cp4 + 2*C_66*cp2 - 4*C_66*cp4 - 2*C_22*ct2 + 2*C_23*ct2 - 2*C_11*cp4*ct2 + 4*C_12*cp4*ct2 + 2*C_22*cp2*ct2 - 2*C_23*cp2*ct2 - 2*C_22*cp4*ct2 + 4*C_66*cp4*ct2)
    E16 = -cp*sp*st*(2*C_12 - 2*C_22 + 2*C_66 + 2*C_11*cp2 - 4*C_12*cp2 + 2*C_22*cp2 - 4*C_66*cp2 + C_22*ct2 - C_23*ct2 - 2*C_11*cp2*ct2 + 4*C_12*cp2*ct2 - 2*C_22*cp2*ct2 + 4*C_66*cp2*ct2)

    # E2
    # E21 = symmetric
    E22 = C_22 + 2*C_12*sp2 + C_11*sp4 - 2*C_12*sp4 - 2*C_22*sp2 + C_22*sp4 + 2*C_66*sp2 - 2*C_66*sp4
    E23 = C_12 + C_11*sp2 - 2*C_12*sp2 - C_11*sp4 + 2*C_12*sp4 + C_22*sp2 - C_22*sp4 - 2*C_66*sp2 + 2*C_66*sp4 - C_12*st2 + C_23*st2 - C_11*sp2*st2 + 3*C_12*sp2*st2 + C_11*sp4*st2 - 2*C_12*sp4*st2 - C_22*sp2*st2 - C_23*sp2*st2 + C_22*sp4*st2 + 2*C_66*sp2*st2 - 2*C_66*sp4*st2
    E24 = -2*cp*ct*sp*(C_12 - C_11 + C_66 + C_11*cp2 - 2*C_12*cp2 + C_22*cp2 - 2*C_66*cp2)
    E25 = 2*cp2*ct*st*(C_12 - C_11 - C_22 + C_23 + 2*C_66 + C_11*cp2 - 2*C_12*cp2 + C_22*cp2 - 2*C_66*cp2)
    E26 = 2*cp*sp*st*(C_12 - C_11 + C_66 + C_11*cp2 - 2*C_12*cp2 + C_22*cp2 - 2*C_66*cp2)

    # E3
    # E31 = symmetric
    # E32 = symmetric
    E33 = C_11 - 2*C_11*sp2 + 2*C_12*sp2 + C_11*sp4 - 2*C_12*sp4 + C_22*sp4 + 2*C_66*sp2 - 2*C_66*sp4 - 2*C_11*st2 + 2*C_12*st2 + C_11*st4 - 2*C_12*st4 + C_22*st4 + 2*C_66*st2 - 2*C_66*st4 + 4*C_11*sp2*st2 - 6*C_12*sp2*st2 - 2*C_11*sp2*st4 - 2*C_11*sp4*st2 + 4*C_12*sp2*st4 + 4*C_12*sp4*st2 + C_11*sp4*st4 - 2*C_12*sp4*st4 + C_22*sp2*st2 + C_23*sp2*st2 - C_22*sp2*st4 - 2*C_22*sp4*st2 - C_23*sp2*st4 + C_22*sp4*st4 - 6*C_66*sp2*st2 + 4*C_66*sp2*st4 + 4*C_66*sp4*st2 - 2*C_66*sp4*st4
    E34 = cp*ct*sp*(2*C_12 - C_22 - C_23 + 2*C_66 - C_22*ct2 + C_23*ct2 + 2*C_11*cp2*ct2 - 4*C_12*cp2*ct2 + 2*C_22*cp2*ct2 - 4*C_66*cp2*ct2)
    E35 = ct*st*(C_22 - C_23 - 2*C_12*cp2 + C_22*cp2 + C_23*cp2 - 2*C_66*cp2 - 2*C_22*ct2 + 2*C_23*ct2 - 2*C_11*cp4*ct2 + 4*C_12*cp4*ct2 + 2*C_22*cp2*ct2 - 2*C_23*cp2*ct2 - 2*C_22*cp4*ct2 + 4*C_66*cp4*ct2)
    E36 = -cp*sp*st*(2*C_12 - 2*C_23 - C_22*ct2 + C_23*ct2 + 2*C_11*cp2*ct2 - 4*C_12*cp2*ct2 + 2*C_22*cp2*ct2 - 4*C_66*cp2*ct2)

    # E4
    E41 = -(cp*ct*sp*(C_22 - 2*C_12 + C_23 - 2*C_11*cp2 + 4*C_12*cp2 - 2*C_22*cp2 + 4*C_66*cp2 - C_22*ct2 + C_23*ct2 + 2*C_11*cp2*ct2 - 4*C_12*cp2*ct2 + 2*C_22*cp2*ct2 - 4*C_66*cp2*ct2))/2
    E42 = -cp*ct*sp*(C_12 - C_11 + C_66 + C_11*cp2 - 2*C_12*cp2 + C_22*cp2 - 2*C_66*cp2)
    E43 = (cp*ct*sp*(2*C_12 - C_22 - C_23 + 2*C_66 - C_22*ct2 + C_23*ct2 + 2*C_11*cp2*ct2 - 4*C_12*cp2*ct2 + 2*C_22*cp2*ct2 - 4*C_66*cp2*ct2))/2
    E44 = C_66*ct2 + C_66*st2 - 4*C_66*cp2*ct2 + 4*C_66*cp4*ct2 + (C_22*cp2*st2)/2 - (C_23*cp2*st2)/2 - C_66*cp2*st2 + 2*C_11*cp2*ct2*sp2 - 4*C_12*cp2*ct2*sp2 + 2*C_22*cp2*ct2*sp2
    E45 = -(cp*sp*st*(C_23 - C_22 + 2*C_66 - 2*C_22*ct2 + 2*C_23*ct2 + 4*C_11*cp2*ct2 - 8*C_12*cp2*ct2 + 4*C_22*cp2*ct2 - 8*C_66*cp2*ct2))/2
    E46 = -(cp2*ct*st*(C_23 - C_22 + 2*C_66 + 4*C_11*sp2 - 8*C_12*sp2 + 4*C_22*sp2 - 8*C_66*sp2))/2

    # E5
    E51 = -(ct*st*(C_22 - C_23 + 2*C_12*cp2 + 2*C_11*cp4 - 4*C_12*cp4 - 3*C_22*cp2 + C_23*cp2 + 2*C_22*cp4 + 2*C_66*cp2 - 4*C_66*cp4 - 2*C_22*ct2 + 2*C_23*ct2 - 2*C_11*cp4*ct2 + 4*C_12*cp4*ct2 + 2*C_22*cp2*ct2 - 2*C_23*cp2*ct2 - 2*C_22*cp4*ct2 + 4*C_66*cp4*ct2))/2
    E52 = cp2*ct*st*(C_12 - C_11 - C_22 + C_23 + 2*C_66 + C_11*cp2 - 2*C_12*cp2 + C_22*cp2 - 2*C_66*cp2)
    E53 = (ct*st*(C_22 - C_23 - 2*C_12*cp2 + C_22*cp2 + C_23*cp2 - 2*C_66*cp2 - 2*C_22*ct2 + 2*C_23*ct2 - 2*C_11*cp4*ct2 + 4*C_12*cp4*ct2 + 2*C_22*cp2*ct2 - 2*C_23*cp2*ct2 - 2*C_22*cp4*ct2 + 4*C_66*cp4*ct2))/2
    # E54 = symmetric
    E55 = C_22/2 - C_23/2 - (C_22*c4t)/2 + (C_23*c4t)/2 + 3*C_66*cp2*ct2 - 2*C_66*cp2*ct4 - 4*C_66*cp4*ct2 + 4*C_66*cp4*ct4 - (C_22*ct2*sp2)/2 + (C_23*ct2*sp2)/2 + C_22*ct4*sp2 - C_23*ct4*sp2 + C_66*cp2*st2 - (C_22*sp2*st2)/2 + (C_23*sp2*st2)/2 + C_22*sp2*st4 - C_23*sp2*st4 + 2*C_11*cp4*ct2*st2 - 4*C_12*cp4*ct2*st2 - 4*C_22*cp2*ct2*st2 + 4*C_23*cp2*ct2*st2 + 2*C_22*cp4*ct2*st2 - 2*C_66*cp2*ct2*st2
    E56 = (cp*ct*sp*(3*C_23 - 3*C_22 + 2*C_66 + 4*C_11*cp2 - 8*C_12*cp2 + 4*C_22*cp2 - 8*C_66*cp2 + 2*C_22*ct2 - 2*C_23*ct2 - 4*C_11*cp2*ct2 + 8*C_12*cp2*ct2 - 4*C_22*cp2*ct2 + 8*C_66*cp2*ct2))/2
    # E6
    E61 = -(cp*sp*st*(2*C_12 - 2*C_22 + 2*C_66 + 2*C_11*cp2 - 4*C_12*cp2 + 2*C_22*cp2 - 4*C_66*cp2 + C_22*ct2 - C_23*ct2 - 2*C_11*cp2*ct2 + 4*C_12*cp2*ct2 - 2*C_22*cp2*ct2 + 4*C_66*cp2*ct2))/2
    E62 = cp*sp*st*(C_12 - C_11 + C_66 + C_11*cp2 - 2*C_12*cp2 + C_22*cp2 - 2*C_66*cp2)
    E63 = -(cp*sp*st*(2*C_12 - 2*C_23 - C_22*ct2 + C_23*ct2 + 2*C_11*cp2*ct2 - 4*C_12*cp2*ct2 + 2*C_22*cp2*ct2 - 4*C_66*cp2*ct2))/2
    # E64 = symmetric
    # E65 = symmetric
    E66 = C_66*ct2 + C_66*st2 + (C_22*cp2*ct2)/2 - (C_23*cp2*ct2)/2 - C_66*cp2*ct2 - 4*C_66*cp2*st2 + 4*C_66*cp4*st2 + 2*C_11*cp2*sp2*st2 - 4*C_12*cp2*sp2*st2 + 2*C_22*cp2*sp2*st2
        
    return E11, E12, E13, E14, E15, E16, E22, E23, E24, E25, E26, E33, E34, E35, E36, E41, E42, E43, E44, E45, E46, E51, E52, E53, E55, E56, E61, E62, E63, E66
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@njit(nogil=True)
def E__TT(phi, theta, C_11, C_22, C_66, C_12, C_23):  
    
    # phi sin

    sp2  = np.sin(phi)**2
    sp4  = np.sin(phi)**4
    # phi cos

        
    # theta sin

    st2 = np.sin(theta)**2
    st4 = np.sin(theta)**4
    # theta cos

    
    E11 = C_22 + 2*C_12*st2 + C_11*st4 - 2*C_12*st4 - 2*C_22*st2 + C_22*st4 + 2*C_66*st2 - 2*C_66*st4 - 2*C_12*sp2*st2 - 2*C_11*sp2*st4 + 4*C_12*sp2*st4 + C_11*sp4*st4 - 2*C_12*sp4*st4 + C_22*sp2*st2 + C_23*sp2*st2 - C_22*sp2*st4 - C_23*sp2*st4 + C_22*sp4*st4 - 2*C_66*sp2*st2 + 4*C_66*sp2*st4 - 2*C_66*sp4*st4
    E22 = C_22 + 2*C_12*sp2 + C_11*sp4 - 2*C_12*sp4 - 2*C_22*sp2 + C_22*sp4 + 2*C_66*sp2 - 2*C_66*sp4
    E33 = C_11 - 2*C_11*sp2 + 2*C_12*sp2 + C_11*sp4 - 2*C_12*sp4 + C_22*sp4 + 2*C_66*sp2 - 2*C_66*sp4 - 2*C_11*st2 + 2*C_12*st2 + C_11*st4 - 2*C_12*st4 + C_22*st4 + 2*C_66*st2 - 2*C_66*st4 + 4*C_11*sp2*st2 - 6*C_12*sp2*st2 - 2*C_11*sp2*st4 - 2*C_11*sp4*st2 + 4*C_12*sp2*st4 + 4*C_12*sp4*st2 + C_11*sp4*st4 - 2*C_12*sp4*st4 + C_22*sp2*st2 + C_23*sp2*st2 - C_22*sp2*st4 - 2*C_22*sp4*st2 - C_23*sp2*st4 + C_22*sp4*st4 - 6*C_66*sp2*st2 + 4*C_66*sp2*st4 + 4*C_66*sp4*st2 - 2*C_66*sp4*st4

    return E11, E22, E33
##-----------------------------------------------------------------------------------------------------
@njit(nogil=True)
def Mori_Tanaka_local_Vf(Vf, C_f, C_m, B_MT, A_MT):
    II      = np.identity(6)
    D_MT    = np.zeros((6, 6))
    D_MT    = np.linalg.inv((1-Vf)*II+Vf*A_MT)
    B_MT    = np.dot(A_MT, D_MT)
    C_MT    = C_m + np.dot(Vf*(C_f-C_m),B_MT)
    C_MT_11 = C_MT[0,0]
    C_MT_22 = C_MT[1,1]
    C_MT_66 = C_MT[5,5]
    C_MT_12 = C_MT[0,1]
    C_MT_23 = C_MT[1,2]
    return C_MT_11, C_MT_22, C_MT_66, C_MT_12, C_MT_23
##-----------------------------------------------------------------------------------------------------    
def Rotate_z_E__(alpha, E11, E12, E13, E14, E15, E16, E22, E23, E24, E25, E26, E33, E34, E35, E36, E41, E42, E43, E44, E45, E46, E51, E52, E53, E55, E56, E61, E62, E63, E66):
    ca = np.cos(alpha)
    c2a= np.cos(2*alpha)
    c4a= np.cos(4*alpha)
    ca2 = ca**2
    ca3 = ca**3
    ca4 = ca**4

    sa = np.sin(alpha)
    s2a= np.sin(2*alpha)
    s4a= np.sin(4*alpha)
    sa2 = sa**2
    sa3 = sa**3

    E21 = E12
    E31 = E13
    E32 = E23
    E54 = E45
    E64 = E46
    E65 = E56

    E11_rot      = E22 + (E26*s2a)/2 + E62*s2a - (E66*(c4a - 1))/4 + E12*ca2 + E11*ca4 - E12*ca4 + E21*ca2 - 2*E22*ca2 - E21*ca4 + E22*ca4 + E16*ca3*sa - E26*ca3*sa + 2*E61*ca3*sa - 2*E62*ca3*sa
    E12_rot      = E21 - (E26*s2a)/2 + E61*s2a + (E66*(c4a - 1))/4 + E11*ca2 - E11*ca4 + E12*ca4 - 2*E21*ca2 + E22*ca2 + E21*ca4 - E22*ca4 - E16*ca3*sa + E26*ca3*sa - 2*E61*ca3*sa + 2*E62*ca3*sa
    E13_rot      = E23 + E63*s2a + E13*ca2 - E23*ca2
    E14_rot      = E15*sa3 - E25*sa3 - 2*E64*sa3 + E24*ca - 2*E65*ca - E15*sa + 2*E64*sa + E14*ca3 - E24*ca3 + 2*E65*ca3
    E15_rot      = E24*sa3 - E14*sa3 - 2*E65*sa3 + E25*ca + 2*E64*ca + E14*sa + 2*E65*sa + E15*ca3 - E25*ca3 - 2*E64*ca3
    E16_rot      = E16/4 - E26/4 - E61/2 + E62/2 - (E11*s2a)/2 + (E12*s2a)/2 - (E11*s4a)/4 + (E12*s4a)/4 - (E21*s2a)/2 + (E22*s2a)/2 + (E21*s4a)/4 - (E22*s4a)/4 + (E66*s4a)/2 + (E16*c2a)/2 + (E16*c4a)/4 + (E26*c2a)/2 - (E26*c4a)/4 + (E61*c4a)/2 - (E62*c4a)/2

    E22_rot      = E11 - (E16*s2a)/2 - E61*s2a - (E66*(c4a - 1))/4 - 2*E11*ca2 + E12*ca2 + E11*ca4 - E12*ca4 + E21*ca2 - E21*ca4 + E22*ca4 + E16*ca3*sa - E26*ca3*sa + 2*E61*ca3*sa - 2*E62*ca3*sa
    E23_rot      = E13 - E63*s2a - E13*ca2 + E23*ca2
    E24_rot      = E25*sa3 - E15*sa3 + 2*E64*sa3 + E14*ca + 2*E65*ca - E25*sa - 2*E64*sa - E14*ca3 + E24*ca3 - 2*E65*ca3
    E25_rot      = E14*sa3 - E24*sa3 + 2*E65*sa3 + E15*ca - 2*E64*ca + E24*sa - 2*E65*sa - E15*ca3 + E25*ca3 + 2*E64*ca3
    E26_rot      = E26/4 - E16/4 + E61/2 - E62/2 - (E11*s2a)/2 + (E12*s2a)/2 + (E11*s4a)/4 - (E12*s4a)/4 - (E21*s2a)/2 + (E22*s2a)/2 - (E21*s4a)/4 + (E22*s4a)/4 - (E66*s4a)/2 + (E16*c2a)/2 - (E16*c4a)/4 + (E26*c2a)/2 + (E26*c4a)/4 - (E61*c4a)/2 + (E62*c4a)/2

    E33_rot      = E33
    E34_rot      = E34*ca - E35*sa
    E35_rot      = E35*ca + E34*sa
    E36_rot      = E32*s2a - E31*s2a + E36*c2a

    E41_rot      = E51*sa3 - E46*sa3 - E52*sa3 + E42*ca - E56*ca + E46*sa - E51*sa + E41*ca3 - E42*ca3 + E56*ca3
    E42_rot      = E46*sa3 - E51*sa3 + E52*sa3 + E41*ca + E56*ca - E46*sa - E52*sa - E41*ca3 + E42*ca3 - E56*ca3
    E43_rot      = E43*ca - E53*sa
    E44_rot      = E55*sa2 - (E54*s2a)/2 - (E45*s2a)/2 + E44*ca2
    E45_rot      = (E44*s2a)/2 - (E55*s2a)/2 - E54*sa2 + E45*ca2
    E46_rot      = E56*sa - E46*ca + 2*E46*ca3 - 2*E41*ca2*sa + 2*E42*ca2*sa + 2*E51*ca*sa2 - 2*E52*ca*sa2 - 2*E56*ca2*sa

    E51_rot      = E42*sa3 - E41*sa3 - E56*sa3 + E46*ca + E52*ca + E41*sa + E56*sa - E46*ca3 + E51*ca3 - E52*ca3
    E52_rot      = E41*sa3 - E42*sa3 + E56*sa3 - E46*ca + E51*ca + E42*sa - E56*sa + E46*ca3 - E51*ca3 + E52*ca3
    E53_rot      = E53*ca + E43*sa
    E55_rot      = (E45*s2a)/2 + (E54*s2a)/2 + E44*sa2 + E55*ca2
    E56_rot      = 2*E56*ca3 - E46*sa - E56*ca - 2*E41*ca*sa2 + 2*E42*ca*sa2 + 2*E46*ca2*sa - 2*E51*ca2*sa + 2*E52*ca2*sa

    E61_rot      = E26/8 - E16/8 + E61/4 - E62/4 - (E11*s2a)/4 - (E12*s2a)/4 - (E11*s4a)/8 + (E12*s4a)/8 + (E21*s2a)/4 + (E22*s2a)/4 + (E21*s4a)/8 - (E22*s4a)/8 + (E66*s4a)/4 + (E16*c4a)/8 - (E26*c4a)/8 + (E61*c2a)/2 + (E62*c2a)/2 + (E61*c4a)/4 - (E62*c4a)/4
    E62_rot      = E16/8 - E26/8 - E61/4 + E62/4 - (E11*s2a)/4 - (E12*s2a)/4 + (E11*s4a)/8 - (E12*s4a)/8 + (E21*s2a)/4 + (E22*s2a)/4 - (E21*s4a)/8 + (E22*s4a)/8 - (E66*s4a)/4 - (E16*c4a)/8 + (E26*c4a)/8 + (E61*c2a)/2 + (E62*c2a)/2 - (E61*c4a)/4 + (E62*c4a)/4
    E63_rot      = (E23*s2a)/2 - (E13*s2a)/2 + E63*c2a
    E66_rot      = E11/4 - E12/4 - E21/4 + E22/4 + E66/2 - (E16*s4a)/4 + (E26*s4a)/4 - (E61*s4a)/2 + (E62*s4a)/2 - (E11*c4a)/4 + (E12*c4a)/4 + (E21*c4a)/4 - (E22*c4a)/4 + (E66*c4a)/2
    return E11_rot, E12_rot, E13_rot, E14_rot, E15_rot, E16_rot, E22_rot, E23_rot, E24_rot, E25_rot, E26_rot, E33_rot, E34_rot, E35_rot, E36_rot, E41_rot, E42_rot, E43_rot, E44_rot, E45_rot, E46_rot, E51_rot, E52_rot, E53_rot, E55_rot, E56_rot, E61_rot, E62_rot, E63_rot, E66_rot,

##-----------------------------------------------------------------------------------------------------    
def Transform_cyl_E__(beta, E11, E22, E24, E23, E33, E34, E42, E43, E44):
  
    cb = np.cos(beta)
    c4b= np.cos(4*beta)
    cb2 = cb**2
    cb3 = cb**3
    cb4 = cb**4

    sb = np.sin(beta)
    s2b= np.sin(2*beta)
    
    E32          = E23

    Exx          = E11
    Err          = E22 - (E24*s2b)/2 - E42*s2b - (E44*(c4b - 1))/4 - 2*E22*cb2 + E23*cb2 + E22*cb4 - E23*cb4 + E32*cb2 - E32*cb4 + E33*cb4 + E24*cb3*sb - E34*cb3*sb + 2*E42*cb3*sb - 2*E43*cb3*sb    
    Ethth        = E33 + (E34*s2b)/2 + E43*s2b - (E44*(c4b - 1))/4 + E23*cb2 + E22*cb4 - E23*cb4 + E32*cb2 - 2*E33*cb2 - E32*cb4 + E33*cb4 + E24*cb3*sb - E34*cb3*sb + 2*E42*cb3*sb - 2*E43*cb3*sb

    return Exx, Err, Ethth
