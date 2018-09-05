import numpy as np
import root_numpy as rnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import matplotlib.mlab as mlab
import ROOT
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import scipy.stats
import matplotlib.ticker as mtick
import h5py
import sys



nbins=12
nbinsVertex = 5



NPlotsLines=6
colors = cm.brg(np.linspace(0, 1, NPlotsLines))


NN_Mode='kart'


def loadData(fName):
    treeName = 't'
    arrayName = rnp.root2array(fName, treename='MAPAnalyzer/t' branches=['Boson_Pt', 'Boson_Phi', 'NVertex' ,
    'recoilslimmedMETsPuppi_Pt', 'recoilslimmedMETsPuppi_Phi', 'recoilslimmedMETsPuppi_sumEt',
    'recoilslimmedMETs_Pt', 'recoilslimmedMETs_Phi', 'recoilslimmedMETs_sumEt',
    'recoilpatpfNoPUMET_Pt','recoilpatpfNoPUMET_Phi', 'recoilpatpfNoPUMET_sumEt',
    'recoilpatpfPUCorrectedMET_Pt', 'recoilpatpfPUCorrectedMET_Phi', 'recoilpatpfPUCorrectedMET_sumEt',
    'recoilpatpfPUMET_Pt', 'recoilpatpfPUMET_Phi', 'recoilpatpfPUMET_sumEt',
    'recoilpatpfTrackMET_Pt', 'recoilpatpfTrackMET_Phi', 'recoilpatpfTrackMET_sumEt' ],)
    DFName = pd.DataFrame.from_records(arrayName.view(np.recarray))
    return(DFName)

def loadData_proj(fName):
    treeName = 't'
    arrayName = rnp.root2array(fName, treeName, branches=['Boson_Pt', 'Boson_Phi', 'NVertex' ,   'recoilslimmedMETsPuppi_PerpZ', 'recoilslimmedMETsPuppi_LongZ', 'recoilslimmedMETs_PerpZ', 'recoilslimmedMETs_LongZ', 'recoilpatpfNoPUMET_PerpZ','recoilpatpfNoPUMET_LongZ','recoilpatpfPUCorrectedMET_PerpZ', 'recoilpatpfPUCorrectedMET_LongZ', 'recoilpatpfPUMET_PerpZ', 'recoilpatpfPUMET_LongZ', 'recoilpatpfTrackMET_PerpZ', 'recoilpatpfTrackMET_LongZ' ],)
    DFName = pd.DataFrame.from_records(arrayName.view(np.recarray))
    return(DFName)

def pol2kar_x(norm, phi):
    x = []
    x = np.sin(phi[:])*norm[:]
    return(x)
def pol2kar_y(norm, phi):
    y = []
    y = np.cos(phi[:])*norm[:]
    return(y)

def kar2pol_n(x, y):
    x = []
    x = np.sin(phi[:])*norm[:]
    return(norm)
def kar2pol_phi(x, y):
    y = []
    y = np.cos(phi[:])*norm[:]
    return(phi)

import numpy as np
import root_numpy as rnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import matplotlib.mlab as mlab
import ROOT
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import scipy.stats
import matplotlib.ticker as mtick







def clopper_pearson(k,n,alpha=0.32):
    """
    http://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
    alpha confidence intervals for a binomial distribution of k expected successes on n trials
    Clopper Pearson intervals are a conservative estimate.
    """
    lo = scipy.stats.beta.ppf(alpha/2, k, n-k+1)
    hi = scipy.stats.beta.ppf(1 - alpha/2, k+1, n-k)
    return lo, hi

def Histogram_px(branchString, labelName, errbars_shift):
    binwidth = ((-(DFName[branchString])-DFName.Boson_Pt.values).max() - (-(DFName[branchString])-DFName.Boson_Pt.values).min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram((-(DFName[branchString])-DFName.Boson_Pt.values), bins=nbins)
    #sy, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=-(DFName[branchString]))
    #sy2, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(DFName[branchString])**2)
    #mean = sy / n
    #std = np.sqrt(sy2/n - mean*mean)
    #plt.errorbar((_[:-1]+(_[1:]-_[:-1])/5*errbars_shift), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])
    #plt.plot((_[1:] + _[:-1])/2, std, marker='.', label=labelName, linestyle="None", color=MVAcolors[errbars_shift])
    plt.hist((-(DFName[branchString])-DFName.Boson_Pt.values), bins=nbinsHist, range=[-200, 200], label=labelName, histtype='step', ec=colors[errbars_shift])


def Histogram_y(branchString, labelName, errbars_shift):
    binwidth = ((-(DFName[branchString])-DFName.Boson_Pt.values).max() - (-(DFName[branchString])-DFName.Boson_Pt.values).min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram((-(DFName[branchString])-DFName.Boson_Pt.values), bins=nbins)
    #sy, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=-(DFName[branchString]))
    #sy2, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(DFName[branchString])**2)
    #mean = sy / n
    #std = np.sqrt(sy2/n - mean*mean)
    #plt.errorbar((_[:-1]+(_[1:]-_[:-1])/5*errbars_shift), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])
    #plt.plot((_[1:] + _[:-1])/2, std, marker='.', label=labelName, linestyle="None", color=MVAcolors[errbars_shift])
    plt.hist((-(DFName[branchString])-DFName.Boson_Pt.values), bins=nbinsHist, range=[-200, 200], label=labelName, histtype='step', ec=colors[errbars_shift])


def plotAveragedDataOverpTZ(branchString, labelName, errbars_shift):
    binwidth = (DFName.Boson_Pt.values.max() - DFName.Boson_Pt.values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName.Boson_Pt, bins=nbins)
    sy, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=-(DFName[branchString]/DFName.Boson_Pt))
    sy2, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(DFName[branchString]/DFName.Boson_Pt)**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[:-1]+(_[1:]-_[:-1])/5*errbars_shift), mean, yerr=std, marker='', label=None, linestyle="None", capsize=0,  color=colors[errbars_shift])
    #plt.plot((_[1:] + _[:-1])/2, mean, marker='.', label=labelName, linestyle="None", color=colors[errbars_shift])
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotAveragedDataOverpTZ_woutError(branchString, labelName, errbars_shift):
    binwidth = (DFName.Boson_Pt.values.max() - DFName.Boson_Pt.values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName.Boson_Pt, bins=nbins)
    sy, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=-(DFName[branchString]/DFName.Boson_Pt))
    sy2, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(DFName[branchString]/DFName.Boson_Pt)**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    #plt.errorbar((_[:-1]+(_[1:]-_[:-1])/5*errbars_shift), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])
    #plt.plot((_[1:] + _[:-1])/2, mean, marker='.', label=labelName, linestyle="None", color=colors[errbars_shift])
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAResponseOverpTZ_woutError(branchString, labelName, errbars_shift):
    binwidth = (DFName.Boson_Pt.values.max() - DFName.Boson_Pt.values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName.Boson_Pt, bins=nbins)
    sy, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=-(DFName[branchString]/DFName.Boson_Pt))
    sy2, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(DFName[branchString]/DFName.Boson_Pt)**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    #plt.errorbar((_[:-1]+(_[1:]-_[:-1])/5*errbars_shift), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])
    #plt.plot((_[1:] + _[:-1])/2, mean, marker='.', label=labelName, linestyle="None", color=MVAcolors[errbars_shift])
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAResponseOverNVertex_woutError(branchString, labelName, errbars_shift):
    binwidth = (DFName_nVertex.NVertex.values.max() - DFName_nVertex.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=-(DFName_nVertex[branchString]/DFName_nVertex.Boson_Pt))
    sy2, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(DFName_nVertex[branchString]/DFName_nVertex.Boson_Pt)**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    #plt.errorbar((_[:-1]+(_[1:]-_[:-1])/5*errbars_shift), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])
    #plt.plot((_[1:] + _[:-1])/2, mean, marker='.', label=labelName, linestyle="None", color=MVAcolors[errbars_shift])
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAResponseOverpTZ_wError(branchString, labelName, errbars_shift):
    binwidth = (DFName.Boson_Pt.values.max() - DFName.Boson_Pt.values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName.Boson_Pt, bins=nbins)
    sy, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=-(DFName[branchString]/DFName.Boson_Pt))
    sy2, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(DFName[branchString]/DFName.Boson_Pt)**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[:-1]+(_[1:]-_[:-1])/3*errbars_shift), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])
    #plt.plot((_[1:] + _[:-1])/2, mean, marker='.', label=labelName, linestyle="None", color=MVAcolors[errbars_shift])
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAResponseOverNVertex_wError(branchString, labelName, errbars_shift):
    binwidth = (DFName_nVertex.NVertex.values.max() - DFName_nVertex.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=-(DFName_nVertex[branchString]/DFName_nVertex.Boson_Pt))
    sy2, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(DFName_nVertex[branchString]/DFName_nVertex.Boson_Pt)**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[:-1]+(_[1:]-_[:-1])/3*errbars_shift), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])
    #plt.plot((_[1:] + _[:-1])/2, mean, marker='.', label=labelName, linestyle="None", color=MVAcolors[errbars_shift])
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])


def plotMVAResolutionOverpTZ_woutError_para(branchString, labelName, errbars_shift):
    binwidth = (DFName.Boson_Pt.values.max() - DFName.Boson_Pt.values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName.Boson_Pt, bins=nbins)
    sy, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=-(DFName[branchString]+DFName.Boson_Pt))
    sy2, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(DFName[branchString]+DFName.Boson_Pt)**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    #plt.errorbar((_[:-1]+(_[1:]-_[:-1])/5*errbars_shift), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])
    #plt.plot((_[1:] + _[:-1])/2, std, marker='.', label=labelName, linestyle="None", color=MVAcolors[errbars_shift])
    plt.errorbar((_[1:] + _[:-1])/2, std, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAResolutionOverNVertex_woutError_para(branchString, labelName, errbars_shift):
    binwidth = (DFName_nVertex.NVertex.values.max() - DFName_nVertex.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=-(DFName_nVertex[branchString]+DFName_nVertex.Boson_Pt))
    sy2, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(DFName_nVertex[branchString]+DFName_nVertex.Boson_Pt)**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    #plt.errorbar((_[:-1]+(_[1:]-_[:-1])/5*errbars_shift), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])
    #plt.plot((_[1:] + _[:-1])/2, std, marker='.', label=labelName, linestyle="None", color=MVAcolors[errbars_shift])
    plt.errorbar((_[1:] + _[:-1])/2, std, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAResolutionOverpTZ_woutError_perp(branchString, labelName, errbars_shift):
    binwidth = (DFName.Boson_Pt.values.max() - DFName.Boson_Pt.values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName.Boson_Pt, bins=nbins)
    sy, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=-(DFName[branchString]))
    sy2, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(DFName[branchString])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    #plt.errorbar((_[:-1]+(_[1:]-_[:-1])/5*errbars_shift), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])
    #plt.plot((_[1:] + _[:-1])/2, std, marker='.', label=labelName, linestyle="None", color=MVAcolors[errbars_shift])
    plt.errorbar((_[1:] + _[:-1])/2, std, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAResolutionOverNVertex_woutError_perp(branchString, labelName, errbars_shift):
    binwidth = (DFName_nVertex.NVertex.values.max() - DFName_nVertex.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=-(DFName_nVertex[branchString]))
    sy2, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(DFName_nVertex[branchString])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    #plt.errorbar((_[:-1]+(_[1:]-_[:-1])/5*errbars_shift), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])
    #plt.plot((_[1:] + _[:-1])/2, std, marker='.', label=labelName, linestyle="None", color=MVAcolors[errbars_shift])
    plt.errorbar((_[1:] + _[:-1])/2, std, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])


def plotInversedResponseOverpTZ(branchString, labelName, errbars_shift):
    binwidth = (DFName.Boson_Pt.values.max() - DFName.Boson_Pt.values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName.Boson_Pt, bins=nbins)
    sy, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=-(DFName.Boson_Pt)/DFName[branchString])
    sy2, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(DFName.Boson_Pt/DFName[branchString])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    #plt.errorbar((_[:-1]+(_[1:]-_[:-1])/5*errbars_shift), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])
    plt.plot((_[1:] + _[:-1])/2, mean, marker='.', label=labelName, linestyle="None", color=colors[errbars_shift])
    #plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])



def plotAveragedDataOverNVertex(branchString, labelName, errbars_shift):
    binwidth = (DFName_nVertex.NVertex.values.max() - DFName_nVertex.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=-(DFName_nVertex[branchString]/DFName_nVertex.Boson_Pt))
    sy2, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(DFName_nVertex[branchString]/DFName_nVertex.Boson_Pt)**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[:-1]+(_[1:]-_[:-1])/5*errbars_shift), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])
    #plt.plot((_[1:] + _[:-1])/2, mean, marker='.', label=labelName, linestyle="None", color=colors[errbars_shift])
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])


def plotAveragedDataOverNVertex_woutErrors(branchString, labelName, errbars_shift):
    binwidth = (DFName_nVertex.NVertex.values.max() - DFName_nVertex.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=-(DFName_nVertex[branchString]/DFName_nVertex.Boson_Pt))
    sy2, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(DFName_nVertex[branchString]/DFName_nVertex.Boson_Pt)**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    #plt.errorbar((_[:-1]+(_[1:]-_[:-1])/5*errbars_shift), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])
    #plt.plot((_[1:] + _[:-1])/2, mean, marker='.', label=labelName, linestyle="None", color=colors[errbars_shift])
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotAveragedResolutionOverpTZ(branchString, labelName, errbars_shift):
    binwidth = (DFName.Boson_Pt.values.max() - DFName.Boson_Pt.values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName.Boson_Pt, bins=nbins)
    if branchString == "recoilpatpfPUMET_LongZ":
        sy, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(DFName[branchString]))
        sy2, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(DFName[branchString])**2)
    else:
        sy, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(DFName[branchString]+DFName.Boson_Pt))
        sy2, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(DFName[branchString]+DFName.Boson_Pt)**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    #plt.errorbar((_[:-1]+(_[1:]-_[:-1])/5*errbars_shift), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])
    #plt.plot((_[1:] + _[:-1])/2, std, marker='.', label=labelName, linestyle="None", color=colors[errbars_shift])
    plt.errorbar((_[1:] + _[:-1])/2, std, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])



def plotAveragedResolutionOverNVertex(branchString, labelName, errbars_shift):
    binwidth = (DFName_nVertex.NVertex.values.max() - DFName_nVertex.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex)
    if branchString == "recoilpatpfPUMET_LongZ":
        sy, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(DFName_nVertex[branchString]))
        sy2, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(DFName_nVertex[branchString])**2)
    else:
        sy, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(DFName_nVertex[branchString]+DFName_nVertex.Boson_Pt))
        sy2, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(DFName_nVertex[branchString]+DFName_nVertex.Boson_Pt)**2)

    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    #plt.errorbar((_[:-1]+(_[1:]-_[:-1])/5*errbars_shift), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])
    #plt.plot((_[1:] + _[:-1])/2, std, marker='.', label=labelName, linestyle="None", color=colors[errbars_shift])
    plt.errorbar((_[1:] + _[:-1])/2, std, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotAveragedResolutionOverpTZPerp(branchString, labelName, errbars_shift):
    binwidth = (DFName.Boson_Pt.values.max() - DFName.Boson_Pt.values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName.Boson_Pt, bins=nbins)
    sy, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=-(DFName[branchString]))
    sy2, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(DFName[branchString])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    #plt.errorbar((_[:-1]+(_[1:]-_[:-1])/5*errbars_shift), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])
    #plt.plot((_[1:] + _[:-1])/2, std, marker='.', label=labelName, linestyle="None", color=colors[errbars_shift])
    plt.errorbar((_[1:] + _[:-1])/2, std, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])



def plotAveragedResolutionOverNVertexPerp(branchString, labelName, errbars_shift):
    binwidth = (DFName_nVertex.NVertex.values.max() - DFName_nVertex.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=-(DFName_nVertex[branchString]))
    sy2, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(DFName_nVertex[branchString])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    #plt.errorbar((_[:-1]+(_[1:]-_[:-1])/5*errbars_shift), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])
    #plt.plot((_[1:] + _[:-1])/2, std, marker='.', label=labelName, linestyle="None", color=colors[errbars_shift])
    plt.errorbar((_[1:] + _[:-1])/2, std, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])


def plotAveragedDataSSEntries(branchString, labelName, errbars_shift):
    df = DFName.Boson_Pt
    xSort = DFName.sort_values('Boson_Pt')
    nEntriesPerSet = len(DFName.Boson_Pt)/nbins
    EntryStart = [x*nEntriesPerSet for x in range(0,nbins)]
    EntryEnd = [(x+1)*nEntriesPerSet-1 for x in range(0,nbins)]
    SSbins = [ (xSort.Boson_Pt.values[EntryEnd[x]]+xSort.Boson_Pt.values[EntryStart[x]])/2 for x in range(0,nbins)]
    n, _ = np.histogram(DFName.Boson_Pt, bins=SSbins)
    sy, _ = np.histogram(DFName.Boson_Pt, bins=SSbins, weights=-(DFName[branchString]/DFName.Boson_Pt))
    sy2, _ = np.histogram(DFName.Boson_Pt, bins=SSbins, weights=(DFName[branchString]/DFName.Boson_Pt)**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[:-1]+(_[1:]-_[:-1])/6*errbars_shift), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])










def getPlotsInput(plotsD, inputD):
#Data settings
    DFName = loadData(inputD)
    DFName=DFName[DFName['Boson_Pt']<=200]
    DFName=DFName[DFName['Boson_Pt']>0]


    DFName_nVertex = DFName[DFName['Boson_Pt']<=200]
    DFName_nVertex = DFName[DFName['Boson_Pt']>60]
    DFName_nVertex = DFName[DFName['nVertex']<=50]

    #Plot settings
    NPlotsLines = 6
    MVA_NPlotsLines = 3
    pTRangeString = '$0\ \mathrm{GeV} < p_{T}^Z \leq 200\ \mathrm{GeV}$'
    pTRangeStringNVertex = '$60\ \mathrm{GeV} < p_{T}^Z \leq 200\ \mathrm{GeV}$'
    LegendTitle = '$\mathrm{Summer\ 17\ campaign}$' '\n'  '$\mathrm{Z \  \\rightarrow \ \mu \mu}$'
    MVAcolors =  colors
    ylimResMin, ylimResMax = 7.5 , 50
    ylimResMVAMin, ylimResMax = 12 , 50
    ResponseMin, ResponseMax = 0, 1.05





    nbins=12
    nbinsVertex = 10
    #nbins=[0.,10.,2ylimResMin, ylimResMax,40,50,60,70,80,90,100,125,150,175,200]

    #Plot with same data size bins
    ######### Histogram NVertex ###########

    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    mu, sigma = np.mean(DFName.NVertex), np.std(DFName.NVertex)
    #plt.hist(DFName.NVertex,label='$N_{\mathrm{Vertex}} \\ \mu=%f \\ \sigma=%f $ '%(mu,sigma), binwidth=10)
    n, bins, patches = plt.hist(DFName.NVertex,label='$N_{\mathrm{Vertex}}$ \n $\mu=%.2f $\n $\sigma=%.2f $ '%(mu,sigma),  bins=range(0,80,10), color='r', alpha=0.5)

    # add a 'best fit' line
    binss = range(0,80,1)
    #y = mlab.normpdf( binss, mu, sigma)
    #l = plt.plot(bins, y, 'r--', linewidth=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.xlabel('$ N_{\mathrm{Vertex}} $ ')
    plt.ylabel('$ N_{\mathrm{Events}} $')
    plt.title('Number of vertices')

    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(-0.005,2.)
    plt.savefig("%sHistogramVertices.png"%(plotsD))




    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    mu, sigma = np.mean(DFName.Boson_Pt), np.std(DFName.Boson_Pt)
    #plt.hist(DFName.NVertex,label='$N_{\mathrm{Vertex}} \\ \mu=%f \\ \sigma=%f $ '%(mu,sigma), binwidth=10)
    n, bins, patches = plt.hist(DFName.Boson_Pt,label='$p_T^Z$  ',  bins=range(0,200,10), color='r', alpha=0.5)

    # add a 'best fit' line
    binss = range(0,80,1)
    #y = mlab.normpdf( binss, mu, sigma)
    #l = plt.plot(bins, y, 'r--', linewidth=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.xlabel('$ {p_T}^Z $ in GeV')
    plt.ylabel('$ N_{\mathrm{Events}} $')
    plt.title('${p_T}^Z$ distribution')

    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(-0.005,2.)
    plt.savefig("%sHistogrampTZ.png"%(plotsD))

    '''
    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)


    plotAveragedDataSSEntries('recoilpatpfNoPUMET_LongZ', 'No PU MET')
    plotAveragedDataSSEntries('recoilpatpfPUCorrectedMET_LongZ', 'PU corrected MET')
    plotAveragedDataSSEntries( 'recoilpatpfPUMET_LongZ', 'PU MET')
    plotAveragedDataSSEntries('recoilpatpfTrackMET_LongZ', 'Track MET')
    plotAveragedDataSSEntries('LongZCorrectedRecoil_LongZ', 'PF MET')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.xlabel('$p_{T}^Z $ in GeV')
    plt.ylabel('$\\frac{u_{\parallel}}{p_{T}^Z} $ ')
    plt.title('Response for MET definitions')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    plt.ylim(-0.005,2.)
    plt.savefig("%sResponseBoson_PtWeighted.png"%(plotsD))
    plt.show()
    '''

    ########## Plot Response over pTZ ############
    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)


    plotAveragedDataOverpTZ('recoilpatpfNoPUMET_LongZ', 'No PU MET', 5)
    plotAveragedDataOverpTZ('recoilpatpfPUCorrectedMET_LongZ', 'PU corrected MET', 4)
    plotAveragedDataOverpTZ( 'recoilpatpfPUMET_LongZ', 'PU MET', 3)
    plotAveragedDataOverpTZ('recoilpatpfTrackMET_LongZ', 'Track MET', 2)
    plotAveragedDataOverpTZ('recoilslimmedMETs_LongZ', 'PF MET', 0)
    plotAveragedDataOverpTZ('recoilslimmedMETsPuppi_LongZ', 'Puppi', 1)
    plt.plot([0, 200], [1, 1], color='k', linestyle='--', linewidth=1)
    #plt.show()

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0, mpatches.Patch(color='none', label=pTRangeString))

    plt.xlabel('$p_{T}^Z $ in GeV')
    plt.ylabel('$\\langle \\frac{U_{\parallel}}{-p_{T}^Z} \\rangle $ ')
    plt.title('Response $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    plt.ylim(ResponseMin, ResponseMax)
    plt.savefig("%sResponseOverpTZ_wError.png"%(plotsD))
    #plt.show()

    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)


    plotAveragedDataOverpTZ_woutError('recoilpatpfNoPUMET_LongZ', 'No PU MET', 5)
    plotAveragedDataOverpTZ_woutError('recoilpatpfPUCorrectedMET_LongZ', 'PU corrected MET', 4)
    plotAveragedDataOverpTZ_woutError( 'recoilpatpfPUMET_LongZ', 'PU MET', 3)
    plotAveragedDataOverpTZ_woutError('recoilpatpfTrackMET_LongZ', 'Track MET', 2)
    plotAveragedDataOverpTZ_woutError('recoilslimmedMETs_LongZ', 'PF MET', 0)
    plotAveragedDataOverpTZ_woutError('recoilslimmedMETsPuppi_LongZ', 'Puppi', 1)
    plt.plot([0, 200], [1, 1], color='k', linestyle='--', linewidth=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0, mpatches.Patch(color='none', label=pTRangeString))

    plt.xlabel('$p_{T}^Z $ in GeV')
    plt.ylabel('$\\langle \\frac{U_{\parallel}}{-p_{T}^Z} \\rangle $ ')
    plt.title('Response $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    plt.ylim(ResponseMin, ResponseMax)
    plt.savefig("%sResponseOverpTZ_woutError.png"%(plotsD))
    #plt.show()

    ########## Plot Response over nPU ############
    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)


    plotAveragedDataOverNVertex('recoilpatpfNoPUMET_LongZ', 'No PU MET', 5)
    plotAveragedDataOverNVertex('recoilpatpfPUCorrectedMET_LongZ', 'PU corrected MET', 4)
    plotAveragedDataOverNVertex( 'recoilpatpfPUMET_LongZ', 'PU MET', 3)
    plotAveragedDataOverNVertex('recoilpatpfTrackMET_LongZ', 'Track MET', 2)
    plotAveragedDataOverNVertex('recoilslimmedMETs_LongZ', 'PF MET', 0)
    plotAveragedDataOverNVertex('recoilslimmedMETsPuppi_LongZ', 'Puppi', 1)
    plt.plot([0, 80], [1, 1], color='k', linestyle='--', linewidth=1)


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    plt.xlabel('#$ \mathrm{PV}$ ')
    plt.ylabel('$\\langle \\frac{U_{\parallel}}{-p_{T}^Z} \\rangle $ ')
    plt.title('Response $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    plt.ylim(ResponseMin, ResponseMax)
    plt.savefig("%sResponseOverNVertex_wError.png"%(plotsD))
    #plt.show()

    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)


    plotAveragedDataOverNVertex_woutErrors('recoilpatpfNoPUMET_LongZ', 'No PU MET', 5)
    plotAveragedDataOverNVertex_woutErrors('recoilpatpfPUCorrectedMET_LongZ', 'PU corrected MET', 4)
    plotAveragedDataOverNVertex_woutErrors( 'recoilpatpfPUMET_LongZ', 'PU MET', 3)
    plotAveragedDataOverNVertex_woutErrors('recoilpatpfTrackMET_LongZ', 'Track MET', 2)
    plotAveragedDataOverNVertex_woutErrors('recoilslimmedMETs_LongZ', 'PF MET', 0)
    plotAveragedDataOverNVertex_woutErrors('recoilslimmedMETsPuppi_LongZ', 'Puppi', 1)
    plt.plot([0, 80], [1, 1], color='k', linestyle='--', linewidth=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    plt.xlabel('#$ \mathrm{PV}$ ')
    plt.ylabel('$\\langle \\frac{U_{\parallel}}{-p_{T}^Z} \\rangle $ ')
    plt.title('Response $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    plt.ylim(ResponseMin, ResponseMax)
    plt.savefig("%sResponseOverNVertex_woutError.png"%(plotsD))
    #plt.show()



    ########## Plot u_para-pTZ over pTZ ############
    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)


    plotAveragedResolutionOverpTZ('recoilpatpfNoPUMET_LongZ', 'No PU MET', 5)
    plotAveragedResolutionOverpTZ('recoilpatpfPUCorrectedMET_LongZ', 'PU corrected MET', 4)
    plotAveragedResolutionOverpTZ( 'recoilpatpfPUMET_LongZ', 'PU MET', 3)
    plotAveragedResolutionOverpTZ('recoilpatpfTrackMET_LongZ', 'Track MET', 2)
    plotAveragedResolutionOverpTZ('recoilslimmedMETs_LongZ', 'PF MET', 0)
    plotAveragedResolutionOverpTZ('recoilslimmedMETsPuppi_LongZ', 'Puppi', 1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))
    print("Zeile 375")
    plt.xlabel('$p_{T}^Z $ in GeV')
    plt.ylabel('$\\sigma \\left( U_{\parallel}-p_{T}^Z \\right) $ in GeV')
    plt.title('Resolution $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    plt.ylim(ylimResMin, ylimResMax)
    plt.savefig("%sResolutionUParaOverPTZ.png"%(plotsD))
    #plt.show()



    ########## Plot u_para-pTZ over nPU ############
    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)


    plotAveragedResolutionOverNVertex('recoilpatpfNoPUMET_LongZ', 'No PU MET', 5)
    plotAveragedResolutionOverNVertex('recoilpatpfPUCorrectedMET_LongZ', 'PU corrected MET', 4)
    plotAveragedResolutionOverNVertex( 'recoilpatpfPUMET_LongZ', 'PU MET', 3)
    plotAveragedResolutionOverNVertex('recoilpatpfTrackMET_LongZ', 'Track MET', 2)
    plotAveragedResolutionOverNVertex('recoilslimmedMETs_LongZ', 'PF MET', 0)
    plotAveragedResolutionOverNVertex('recoilslimmedMETsPuppi_LongZ', 'Puppi', 1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    print("Zeile 406")
    plt.xlabel('#$ \mathrm{PV}$ ')
    plt.ylabel('$\\sigma \\left( U_{\parallel}-p_{T}^Z \\right) $ in GeV')
    plt.title('Resolution $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    plt.ylim(ylimResMin, ylimResMax)
    plt.savefig("%sResolutionU_paraOverNVertex.png"%(plotsD))
    #plt.show()



    ########## Plot u_perp over pTZ ############
    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)


    plotAveragedResolutionOverpTZPerp('recoilpatpfNoPUMET_PerpZ', 'No PU MET', 5)
    plotAveragedResolutionOverpTZPerp('recoilpatpfPUCorrectedMET_PerpZ', 'PU corrected MET', 4)
    plotAveragedResolutionOverpTZPerp( 'recoilpatpfPUMET_PerpZ', 'PU MET', 3)
    plotAveragedResolutionOverpTZPerp('recoilpatpfTrackMET_PerpZ', 'Track MET', 2)
    plotAveragedResolutionOverpTZPerp('recoilslimmedMETs_PerpZ', 'PF MET', 0)
    plotAveragedResolutionOverpTZPerp('recoilslimmedMETsPuppi_PerpZ', 'Puppi', 1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))


    plt.xlabel('$p_{T}^Z $ in GeV')
    plt.ylabel('$ \\sigma \\left( U_{\perp} \\right) $ in GeV')
    plt.title('Resolution $U_{\perp}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    plt.ylim(ylimResMin, ylimResMax)
    plt.savefig("%sResolutionUPerpOverPTZ.png"%(plotsD))
    #plt.show()



    ########## Plot u_perp over nPU ############
    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)
    box = ax.get_position()


    plotAveragedResolutionOverNVertexPerp('recoilpatpfNoPUMET_PerpZ', 'No PU MET', 5)
    plotAveragedResolutionOverNVertexPerp('recoilpatpfPUCorrectedMET_PerpZ', 'PU corrected MET', 4)
    plotAveragedResolutionOverNVertexPerp( 'recoilpatpfPUMET_PerpZ', 'PU MET', 3)
    plotAveragedResolutionOverNVertexPerp('recoilpatpfTrackMET_PerpZ', 'Track MET', 2)
    plotAveragedResolutionOverNVertexPerp('recoilslimmedMETs_PerpZ', 'PF MET', 0)
    plotAveragedResolutionOverNVertexPerp('recoilslimmedMETsPuppi_PerpZ', 'Puppi', 1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    plt.xlabel('#$ \mathrm{PV}$ ')
    plt.ylabel('$ \\sigma \\left( U_{\perp} \\right) $ in GeV')
    plt.title('Resolution $U_{\perp}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    plt.ylim(ylimResMin, ylimResMax)
    plt.savefig("%sResolutionUPerpOverNVertex.png"%(plotsD))
    #plt.show()




    ########## Inversed Response over pTZ ############
    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)




    plotInversedResponseOverpTZ('recoilpatpfNoPUMET_LongZ', 'No PU MET', 5)
    plotInversedResponseOverpTZ('recoilpatpfPUCorrectedMET_LongZ', 'PU corrected MET', 4)
    plotInversedResponseOverpTZ( 'recoilpatpfPUMET_LongZ', 'PU MET', 3)
    plotInversedResponseOverpTZ('recoilpatpfTrackMET_LongZ', 'Track MET', 2)
    plotInversedResponseOverpTZ('recoilslimmedMETs_LongZ', 'PF MET', 0)
    plotInversedResponseOverpTZ('recoilslimmedMETsPuppi_LongZ', 'PF MET', 1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.xlabel('$p_{\mathrm{T}}^{\mathrm{Z}} $ ')
    plt.ylabel('$\langle \\frac{-p_{T}^Z}{U_{\parallel}} \\rangle $ ')
    plt.title('Resolution $U_{\parallel}$')
    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)

    plt.grid()
    plt.ylim(ylimResMin, ylimResMax)
    plt.savefig("%sInversedResponseOverPTZ.png"%(plotsD))
    #plt.show()

    '''
    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    Histogram_px('LongZCorrectedRecoil_LongZ', 'GBRT', 0)
    Histogram_px('NN_LongZ', 'NN', 2)
    Histogram_px('recoilslimmedMETs_LongZ', 'PF', 1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.ylabel('Counts')
    plt.xlabel('$ U_{\parallel} - p_T^Z  $ in GeV')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_{T}^Z} \\right) $ in GeV')
    plt.title('Response Histogram')
    #plt.text('$p_T$ range restriction')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(ylimResMVAMin, ylimResMax)
    plt.savefig("%sOutput_Response_Hist.png"%(plotsD))

    '''











#scatter plot
#plt.scatter(DFName.Boson_Pt, -(DFName.recoilpatpfNoPUMET_LongZ/DFName.Boson_Pt))
#plt.xlabel('$p_{T}^Z $ in GeV')
#plt.ylabel('$u_{\parallel} $ ')
# plt.ylim(-0.001,0.001)
# plt.xlim(-0.001,0.001)
#plt.show()



if __name__ == "__main__":
    inputDir = sys.argv[1]
    outputDir = sys.argv[2]
    print(outputDir)
    getPlotsInput(outputDir, inputDir)
