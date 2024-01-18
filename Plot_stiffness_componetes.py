# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 17:40:14 2023
"""
import seaborn as sns
import matplotlib.pyplot as plt
import os
##------------------------------------------------------------------------------------------------------------------------------------------------------------------
#%% Compare new and old E11
def plot_compare_E11_histogram(nE11,oE11,script_name,fig_path,fig_theme='dark',save_tif=False,save_svg=False):
    plt.style.use("default")
    if fig_theme=='dark':
        plt.style.use("dark_background")
    else:
        plt.style.use("default")
    font = {'family': 'serif',
            'weight': 'normal',
            'size': 36,
            }
    f, ax = plt.subplots(figsize=(10, 8))
    n=0
    c=['mediumseagreen','steelblue']

    for a in [nE11,oE11]:
        hisplot=sns.histplot(a.flatten()/1000, edgecolor="1", bins=60, binrange=[5,35], color=c[n],
                linewidth=0.3,stat='probability',  alpha=.6)
        n=n+1

    plt.legend(labels=["updated $E_{11}$","original $E_{11}$"], fontsize=36, ncol=1,loc='upper right')
    plt.grid(color='grey', linestyle='-',  linewidth=0.3, alpha=0.9)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=24)
    hisplot.spines['top'].set_visible(False)
    hisplot.spines['right'].set_visible(False)
    hisplot.set(ylabel=None)
    hisplot.set_xticks([0,10,20,30,40])
#    hisplot.set_yticks([0.05,0.1])
    hisplot.set_xlim([0,40])
#    hisplot.set_ylim([0, 0.1])
    hisplot.set_xlabel('$E_{11}$ $[GPa]$',fontdict= font, labelpad=5)
    hisplot.set_ylabel('Norm. IP count',fontdict= font, labelpad=15)
    if save_svg=='True':
        plt.savefig(os.path.join(fig_path, f'{script_name}_E11_comp_Histogram_{fig_theme}.svg'), bbox_inches='tight', pad_inches=0, dpi=600)
    if save_tif=='True':
        plt.savefig(os.path.join(fig_path, f'{script_name}_E11_comp_Histogram_{fig_theme}.tif'), bbox_inches='tight', pad_inches=0, dpi=100) 
##--------------------------------------------------------------------------------------------------------------------------------------------
#%% Compare new and old E22
def plot_compare_E22_histogram(nE22,oE22,script_name,fig_path,fig_theme='dark',save_tif=False,save_svg=False):
    plt.style.use("default")
    if fig_theme=='dark':
        plt.style.use("dark_background")
    else:
        plt.style.use("default")
    font = {'family': 'serif',
            'weight': 'normal',
            'size': 36,
            }
    f, ax = plt.subplots(figsize=(10, 8))
    n=0
    c=['mediumseagreen','steelblue']

    for a in [nE22,oE22]:
        hisplot=sns.histplot(a.flatten()/1000, edgecolor="1", bins=60, binrange=[5,35], color=c[n],
                linewidth=0.3,stat='probability',  alpha=.6)
        n=n+1

    plt.legend(labels=["updated $E_{22}$","original $E_{22}$"], fontsize=36, ncol=1,loc='upper right')
    plt.grid(color='grey', linestyle='-',  linewidth=0.3, alpha=0.9)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=24)
    hisplot.spines['top'].set_visible(False)
    hisplot.spines['right'].set_visible(False)
    hisplot.set(ylabel=None)
    hisplot.set_xticks([0,10,20,30,40])
#    hisplot.set_yticks([0.05,0.1])
    hisplot.set_xlim([0,40])
#    hisplot.set_ylim([0, 0.1])
    hisplot.set_xlabel('$E_{22}$ $[GPa]$',fontdict= font, labelpad=5)
    hisplot.set_ylabel('Norm. IP count',fontdict= font, labelpad=15)
    if save_svg=='True':
        plt.savefig(os.path.join(fig_path, f'{script_name}_E22_comp_Histogram_{fig_theme}.svg'), bbox_inches='tight', pad_inches=0, dpi=600)
    if save_tif=='True':
        plt.savefig(os.path.join(fig_path, f'{script_name}_E22_comp_Histogram_{fig_theme}.tif'), bbox_inches='tight', pad_inches=0, dpi=100) 
##--------------------------------------------------------------------------------------------------------------------------------------------
#%% Compare new and old E33
def plot_compare_E33_histogram(nE33,oE33,script_name,fig_path,fig_theme='dark',save_tif=False,save_svg=False):
    plt.style.use("default")
    if fig_theme=='dark':
        plt.style.use("dark_background")
    else:
        plt.style.use("default")
    font = {'family': 'serif',
            'weight': 'normal',
            'size': 36,
            }
    f, ax = plt.subplots(figsize=(10, 8))
    n=0
    c=['mediumseagreen','steelblue']

    for a in [nE33,oE33]:
        hisplot=sns.histplot(a.flatten()/1000, edgecolor="1", bins=60, binrange=[5,35], color=c[n],
                linewidth=0.3,stat='probability',  alpha=.6)
        n=n+1

    plt.legend(labels=["updated $E_{33}$","original $E_{33}$"], fontsize=36, ncol=1,loc='upper right')
    plt.grid(color='grey', linestyle='-',  linewidth=0.3, alpha=0.9)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=24)
    hisplot.spines['top'].set_visible(False)
    hisplot.spines['right'].set_visible(False)
    hisplot.set(ylabel=None)
    hisplot.set_xticks([0,10,20,30,40])
#    hisplot.set_yticks([0.05,0.1])
    hisplot.set_xlim([0,40])
#    hisplot.set_ylim([0, 0.1])
    hisplot.set_xlabel('$E_{33}$ $[GPa]$',fontdict= font, labelpad=5)
    hisplot.set_ylabel('Norm. IP count',fontdict= font, labelpad=15)
    if save_svg=='True':
        plt.savefig(os.path.join(fig_path, f'{script_name}_E33_comp_Histogram_{fig_theme}.svg'), bbox_inches='tight', pad_inches=0, dpi=600)
    if save_tif=='True':
        plt.savefig(os.path.join(fig_path, f'{script_name}_E33_comp_Histogram_{fig_theme}.tif'), bbox_inches='tight', pad_inches=0, dpi=100) 
##--------------------------------------------------------------------------------------------------------------------------------------------

