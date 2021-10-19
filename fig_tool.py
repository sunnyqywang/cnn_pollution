import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.gridspec as gridspec
import pickle


from pprint import pprint
#import numba
#import numexpr
from collections import defaultdict


from sklearn.cluster import *
from sklearn import metrics
import pandas as pd
import numpy as np
import scipy
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
import scipy.stats as ss
import scipy.io as sio


# create colormaps
import seaborn as sns
[plblue,pblue,plgreen,pgreen,plred,pred,plorange,porange,plpurple,ppurple,plbrown,pbrown] = sns.color_palette("Paired",12)

import palettable
pal = palettable.colorbrewer.qualitative.Set1_9
colors = pal.mpl_colors
[cred,cblue,cgreen,cpurple,corange,cyellow,cbrown,cpink,cgray] = colors
#matplotlib.rc("axes",color_cycle=colors)

import matplotlib.colors as colors
colors.colorConverter.colors['cblue'] = cblue
colors.colorConverter.colors['cred'] = cred
colors.colorConverter.colors['cgreen'] = cgreen
colors.colorConverter.colors['corange'] = corange
colors.colorConverter.colors['cpink'] = cpink
colors.colorConverter.colors['cbrown'] = cbrown
colors.colorConverter.colors['cgray'] = cgray
colors.colorConverter.colors['cpurple'] = cpurple
colors.colorConverter.colors['cyellow'] = cyellow

colors.colorConverter.colors['plblue'] = plblue
colors.colorConverter.colors['pblue'] = pblue
colors.colorConverter.colors['plgreen'] = plgreen
colors.colorConverter.colors['pgreen'] = pgreen
colors.colorConverter.colors['plred'] = plred
colors.colorConverter.colors['pred'] = pred
colors.colorConverter.colors['plorange'] = plorange
colors.colorConverter.colors['porange'] = porange
colors.colorConverter.colors['plpurple'] = plpurple
colors.colorConverter.colors['ppurple'] = ppurple
colors.colorConverter.colors['plbrown'] = plbrown
colors.colorConverter.colors['pbrown'] = pbrown

import palettable
#import psycopg2


# import colormaps as cmaps
# cmap_viridis = cmaps.viridis
# cmap_magma = cmaps.magma
# cmap_inferno = cmaps.inferno
# cmap_plasma = cmaps.plasma
# cmap_spectral = palettable.colorbrewer.diverging.Spectral_11_r.mpl_colormap
# cmap_viridis_r = matplotlib.colors.ListedColormap(cmaps.viridis.colors[::-1])


font_arial = {'family' : 'Arial',
                'weight' : 'medium',
                'size'   : 12,
                'style'  : 'normal'}

def universal_fig(figsize=(3,3),fontsize=12,axislinewidth=1,markersize=5,text=None,limits=[-7,7],offset=[-44,12],projection=None, fontfamily=["Helvetica","Arial"]):
    '''
    Create universal figure settings with publication quality
    returen fig, ax (similar to plt.plot)
    fig, ax = universal_fig()
    '''
    # ----------------------------------------------------------------
    if projection is None: fig,ax = plt.subplots(frameon = False)
    else: fig,ax = plt.subplots(frameon = False, subplot_kw=dict(projection=projection))
    fig.set_size_inches(figsize)
    matplotlib.rc("font",**{"family":"sans-serif", "sans-serif": fontfamily, "size": fontsize})
    #matplotlib.rc('pdf', fonttype=42,use14corefonts=False,compression=6)
    #matplotlib.rc('ps',useafm=False,usedistiller='none')
    matplotlib.rc("axes",unicode_minus=False,linewidth=axislinewidth,labelsize='medium')
    matplotlib.rc("axes.formatter",limits=limits)
    matplotlib.rc('savefig',bbox='tight',format='pdf',frameon=False,pad_inches=0.05)
    matplotlib.rc('legend')
    matplotlib.rc('lines',marker=None,markersize=markersize)
    matplotlib.rc('text',usetex=False)
    matplotlib.rc('xtick',direction='in')
    matplotlib.rc('xtick.major',size=4)
    matplotlib.rc('xtick.minor',size=2)
    matplotlib.rc('ytick',direction='in')
    matplotlib.rc('ytick.major',size=4)
    matplotlib.rc('ytick.minor',size=2)
    matplotlib.rcParams['lines.solid_capstyle'] = 'butt'
    matplotlib.rcParams['lines.solid_joinstyle'] = 'bevel'
    #matplotlib.rc('mathtext',fontset='stixsans')
    matplotlib.rcParams['mathtext.fontset']='custom'
    matplotlib.rcParams['mathtext.rm']='Arial'
    matplotlib.rcParams['mathtext.it']='Arial'
    matplotlib.rc('font',**font_arial)
    
    matplotlib.rc('legend',fontsize='medium',frameon=True,
                  handleheight=0.5,handlelength=1,handletextpad=0.4,numpoints=1)
    if text is not None: 
        w = ax.annotate(text, xy=(0, 1), xycoords='axes fraction', fontsize='large',weight='bold',
                xytext=(offset[0]/12*fontsize, offset[1]/12*fontsize), textcoords='offset points', ha='left', va='top')
        print(w.get_fontname())
    # ----------------------------------------------------------------
    # end universal settings
    return fig, ax
