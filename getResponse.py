
import numpy as np
import root_numpy as rnp
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import matplotlib.mlab as mlab
import ROOT
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import scipy.stats
import matplotlib.ticker as mtick
from matplotlib.colors import LogNorm
import h5py
import sys


fName ="/storage/b/tkopf/mvamet/skim/out.root"
nbins = 10
nbinsVertex = 5
nbinsHist = 250
nbinsHistBin =300
nbins_relR = 10
colors = cm.brg(np.linspace(0, 1, 8))

colors_InOut = cm.brg(np.linspace(0, 1, 8))
colors2 = cm.brg(np.linspace(0, 1, nbins))
HistLimMin, HistLimMax = -50, 50
ResponseMin, ResponseMax = -1,3
ResolutionParaMin, ResolutionParaMax = -40, 40
ResolutionPerpMin, ResolutionPerpMax = -40, 40

pTTresh = 10

#Data settings

def div0( a, b ):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
    return c

def kar2pol(x, y):
    rho = np.sqrt(np.multiply(x,x) + np.multiply(y,y))
    phi = np.arctan2(y, x)
    return(rho, phi)

def angularrange(Winkel):
    if isinstance(Winkel, (list, tuple, np.ndarray)):
        for i in range(0, len(Winkel) ):
            Winkel[i]=((Winkel[i]+np.pi)%(2*np.pi)-(np.pi))
    else:
        Winkel=((Winkel+np.pi)%(2*np.pi)-(np.pi))
    return(Winkel)

def loadData(inputD):

    tree='t'
    arrayName = rnp.root2array(inputD,  branches=['Boson_Pt', 'Boson_Phi', 'NVertex' ,
    'recoilslimmedMETsPuppi_Pt', 'recoilslimmedMETsPuppi_Phi',
    'recoilslimmedMETs_Pt', 'recoilslimmedMETs_Phi',
    'recoilpatpfNoPUMET_Pt','recoilpatpfNoPUMET_Phi',
    'recoilpatpfPUCorrectedMET_Pt', 'recoilpatpfPUCorrectedMET_Phi',
    'recoilpatpfPUMET_Pt', 'recoilpatpfPUMET_Phi',
    'recoilpatpfTrackMET_Pt', 'recoilpatpfTrackMET_Phi',
    'LongZCorrectedRecoil_Pt', 'LongZCorrectedRecoil_Phi',
    'LongZCorrectedRecoil_LongZ', 'LongZCorrectedRecoil_PerpZ',
    'recoilslimmedMETs_LongZ', 'recoilslimmedMETs_PerpZ',
    'recoilpatpfNoPUMET_LongZ','recoilpatpfNoPUMET_PerpZ',
    'recoilpatpfPUCorrectedMET_LongZ', 'recoilpatpfPUCorrectedMET_PerpZ',
    'recoilpatpfPUMET_LongZ', 'recoilpatpfPUMET_PerpZ',
    'recoilpatpfTrackMET_LongZ', 'recoilpatpfTrackMET_PerpZ',
    'recoilslimmedMETsPuppi_LongZ', 'recoilslimmedMETsPuppi_PerpZ',
    'NN_LongZ', 'NN_PerpZ'
     ],)
    DFName = pd.DataFrame.from_records(arrayName.view(np.recarray))
    return(DFName)

def plotMVAResponseOverpTZ_woutError(branchString, labelName, errbars_shift):
    binwidth = (DFName.Boson_Pt.values.max() - DFName.Boson_Pt.values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName.Boson_Pt, bins=nbins)
    sy, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(-(DFName[branchString])/DFName.Boson_Pt))
    sy2, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(DFName[branchString]/DFName.Boson_Pt)**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    meanc = np.mean((-(DFName[branchString])/DFName.Boson_Pt))
    stdc = np.std((-(DFName[branchString])/DFName.Boson_Pt))
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName+'%8.2f $\pm$ %8.2f'%(meanc, stdc), linestyle="None", capsize=0,  color=colors[errbars_shift])


def plotMVANormOverpTZ(branchString, labelName, errbars_shift):
    binwidth = (DFName.Boson_Pt.values.max() - DFName.Boson_Pt.values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName.Boson_Pt, bins=nbins)
    if branchString=='NN_LongZ':
        NN_Pt = np.sqrt(np.multiply(DFName['NN_LongZ'],DFName['NN_LongZ'])+np.multiply(DFName['NN_PerpZ'],DFName['NN_PerpZ']))
        sy, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(NN_Pt/DFName.Boson_Pt))
        sy2, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(NN_Pt/DFName.Boson_Pt)**2)
        meanc = np.mean(NN_Pt/DFName.Boson_Pt)
        stdc = np.std(NN_Pt/DFName.Boson_Pt)
    else:
        sy, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=((DFName[branchString])/DFName.Boson_Pt))
        sy2, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(DFName[branchString]/DFName.Boson_Pt)**2)
        meanc = np.mean(DFName[branchString]/DFName.Boson_Pt)
        stdc = np.std(DFName[branchString]/DFName.Boson_Pt)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName+'%8.2f $\pm$ %8.2f'%(meanc, stdc), linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAAngularOverpTZ(branchStringLong, branchStringPerp, labelName, errbars_shift):
    binwidth = (DFName.Boson_Pt.values.max() - DFName.Boson_Pt.values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName.Boson_Pt, bins=nbins)
    if branchStringLong=='NN_LongZ':
        r_, phi_ = kar2pol(DFName['NN_LongZ'], DFName['NN_PerpZ'])
        phi_ = angularrange(phi_+np.pi)
        sy, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(phi_))
        sy2, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(phi_)**2)
        meanc = np.mean(phi_)
        stdc = np.std(phi_)
    else:
        r_, phi_ = kar2pol(DFName[branchStringLong], DFName[branchStringPerp])
        phi_ = angularrange(phi_+np.pi)
        sy, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(phi_))
        sy2, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(phi_)**2)
        meanc = np.mean(phi_)
        stdc = np.std(phi_)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName+'%8.2f $\pm$ %8.2f'%(meanc, stdc), linestyle="None", capsize=0,  color=colors[errbars_shift])



def plotMVANormOverpTZ_wErr(branchString, labelName, errbars_shift):
    binwidth = (DFName.Boson_Pt.values.max() - DFName.Boson_Pt.values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName.Boson_Pt, bins=nbins)
    if branchString=='NN_LongZ':
        NN_Pt = np.sqrt(np.multiply(DFName['NN_LongZ'],DFName['NN_LongZ'])+np.multiply(DFName['NN_PerpZ'],DFName['NN_PerpZ']))
        sy, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(NN_Pt/DFName.Boson_Pt))
        sy2, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(NN_Pt/DFName.Boson_Pt)**2)
        meanc = np.mean(NN_Pt/DFName.Boson_Pt)
        stdc = np.std(NN_Pt/DFName.Boson_Pt)
    else:
        sy, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=((DFName[branchString])/DFName.Boson_Pt))
        sy2, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(DFName[branchString]/DFName.Boson_Pt)**2)
        meanc = np.mean(DFName[branchString]/DFName.Boson_Pt)
        stdc = np.std(DFName[branchString]/DFName.Boson_Pt)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName+'%8.2f $\pm$ %8.2f'%(meanc, stdc), linestyle="None", capsize=0,  color=colors[errbars_shift])
    if errbars_shift==5: errbars_shift2 = 0
    elif errbars_shift==6: errbars_shift2 = 1
    elif errbars_shift==1: errbars_shift2 = 0
    plt.errorbar((_[:-1]+(_[1:]-_[:-1])/3*errbars_shift2), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAAngularOverpTZ_wErr(branchStringLong, branchStringPerp, labelName, errbars_shift):
    binwidth = (DFName.Boson_Pt.values.max() - DFName.Boson_Pt.values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName.Boson_Pt, bins=nbins)
    if branchStringLong=='NN_LongZ':
        r_, phi_ = kar2pol(DFName['NN_LongZ'], DFName['NN_PerpZ'])
        phi_ = angularrange(phi_+np.pi)
        sy, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(phi_))
        sy2, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(phi_)**2)
        meanc = np.mean(phi_)
        stdc = np.std(phi_)
    else:
        r_, phi_ = kar2pol(DFName[branchStringLong], DFName[branchStringPerp])
        phi_ = angularrange(phi_+np.pi)
        sy, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(phi_))
        sy2, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(phi_)**2)
        meanc = np.mean(phi_)
        stdc = np.std(phi_)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName+'%8.2f $\pm$ %8.2f'%(meanc, stdc), linestyle="None", capsize=0,  color=colors[errbars_shift])
    if errbars_shift==5: errbars_shift2 = 0
    elif errbars_shift==6: errbars_shift2 = 1
    elif errbars_shift==1: errbars_shift2 = 0
    plt.errorbar((_[:-1]+(_[1:]-_[:-1])/3*errbars_shift2), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])



def plotMVAResponseOverpTZ_woutError_Tresh(branchString, labelName, errbars_shift):
    DFName_Tresh = DFName[DFName['Boson_Pt']>pTTresh]
    binwidth = (DFName_Tresh.Boson_Pt.values.max() - DFName_Tresh.Boson_Pt.values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName_Tresh.Boson_Pt, bins=nbins)
    sy, _ = np.histogram(DFName_Tresh.Boson_Pt, bins=nbins, weights=(-(DFName_Tresh[branchString])/DFName_Tresh.Boson_Pt))
    sy2, _ = np.histogram(DFName_Tresh.Boson_Pt, bins=nbins, weights=(DFName_Tresh[branchString]/DFName_Tresh.Boson_Pt)**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])


def plotMVAResponseOverNVertex_woutError(branchString, labelName, errbars_shift):
    binwidth = (DFName_nVertex.NVertex.values.max() - DFName_nVertex.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=-(DFName_nVertex[branchString])/DFName_nVertex.Boson_Pt)
    sy2, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(DFName_nVertex[branchString]/DFName_nVertex.Boson_Pt)**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    meanc=np.mean(-(DFName_nVertex[branchString])/DFName_nVertex.Boson_Pt)
    stdc=np.std(-(DFName_nVertex[branchString])/DFName_nVertex.Boson_Pt)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName+'%8.2f $\pm$ %8.2f'%(meanc, stdc), linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAResponseOverNVertex_woutError_Tresh(branchString, labelName, errbars_shift):
    DFName_nVertex_Tresh = DFName_nVertex[DFName_nVertex['Boson_Pt']>pTTresh]
    binwidth = (DFName_nVertex_Tresh.NVertex.values.max() - DFName_nVertex_Tresh.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName_nVertex_Tresh.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName_nVertex_Tresh.NVertex, bins=nbinsVertex, weights=-(DFName_nVertex_Tresh[branchString])/DFName_nVertex_Tresh.Boson_Pt)
    sy2, _ = np.histogram(DFName_nVertex_Tresh.NVertex, bins=nbinsVertex, weights=(DFName_nVertex_Tresh[branchString]/DFName_nVertex_Tresh.Boson_Pt)**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])


def plotMVAResponseOverpTZ_wError(branchString, labelName, errbars_shift):
    binwidth = (DFName.Boson_Pt.values.max() - DFName.Boson_Pt.values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName.Boson_Pt, bins=nbins)
    sy, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=-(DFName[branchString]/DFName.Boson_Pt))
    sy2, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(DFName[branchString]/DFName.Boson_Pt)**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    if errbars_shift==5: errbars_shift2 = 0
    elif errbars_shift==6: errbars_shift2 = 1
    elif errbars_shift==1: errbars_shift2 = 0

    plt.errorbar((_[:-1]+(_[1:]-_[:-1])/3*errbars_shift2), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAResponseOverNVertex_wError(branchString, labelName, errbars_shift):
    binwidth = (DFName_nVertex.NVertex.values.max() - DFName_nVertex.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=-(DFName_nVertex[branchString])/DFName_nVertex.Boson_Pt)
    sy2, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(DFName_nVertex[branchString]/DFName_nVertex.Boson_Pt)**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    if errbars_shift==5: errbars_shift2 = 0
    elif errbars_shift==6: errbars_shift2 = 1
    elif errbars_shift==1: errbars_shift2 = 0

    plt.errorbar((_[:-1]+(_[1:]-_[:-1])/3*errbars_shift2), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])


def plotMVAResolutionOverpTZ_woutError_para(branchString, labelName, errbars_shift):
    binwidth = (DFName.Boson_Pt.values.max() - DFName.Boson_Pt.values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName.Boson_Pt, bins=nbins)
    sy, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=-(DFName[branchString]+DFName.Boson_Pt))
    sy2, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(DFName[branchString]+DFName.Boson_Pt)**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, std, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])



def mean_Response(branchString_Long, branchString_Perp, labelName, errbars_shift):
    binwidth = (-np.pi - np.pi)/(200)
    n, _ = np.histogram(DFName.Boson_Pt, bins=nbins)
    NN_r, NN_phi = kar2pol(-DFName[branchString_Long], DFName[branchString_Perp])

    n, _ = np.histogram(NN_phi, bins=200)
    sy, _ = np.histogram(NN_phi, bins=200, weights=np.divide(-DFName[branchString_Long], DFName['Boson_Pt'] ))
    sy2, _ = np.histogram(NN_phi, bins=200, weights=np.divide(-DFName[branchString_Long], DFName['Boson_Pt'] )**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])
    plt.errorbar((_[:-1]+(_[1:]-_[:-1])/2*errbars_shift), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift], linewidth=1.0)



def mean_Response_CR(branchString_Long, branchString_Perp, labelName, errbars_shift):
    binwidth = (-np.pi/180*10- np.pi/180*10)/(200)
    n, _ = np.histogram(DFName.Boson_Pt, bins=nbins)
    NN_r, NN_phi = kar2pol(-DFName[branchString_Long], DFName[branchString_Perp])

    n, _ = np.histogram(NN_phi, bins=200, range=[-np.pi/180*10, np.pi/180*10])
    sy, _ = np.histogram(NN_phi, bins=200, weights=np.divide(-DFName[branchString_Long], DFName['Boson_Pt'] ), range=[-np.pi/180*10, np.pi/180*10])
    sy2, _ = np.histogram(NN_phi, bins=200, weights=np.divide(-DFName[branchString_Long], DFName['Boson_Pt'] )**2, range=[-np.pi/180*10, np.pi/180*10])
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])
    plt.errorbar((_[:-1]+(_[1:]-_[:-1])/2*errbars_shift), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift], linewidth=1.0)


def plotMVAResolutionOverNVertex_woutError_para(branchString, labelName, errbars_shift):
    binwidth = (DFName_nVertex.NVertex.values.max() - DFName_nVertex.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=-(DFName_nVertex[branchString]+DFName_nVertex.Boson_Pt))
    sy2, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(DFName_nVertex[branchString]+DFName_nVertex.Boson_Pt)**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, std, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def MeanDeviation_Pt(branchString, labelName, errbars_shift):
    binwidth = (DFName.Boson_Pt.values.max() - DFName.Boson_Pt.values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName.Boson_Pt, bins=nbins)
    sy, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=-(DFName[branchString]+DFName.Boson_Pt))
    sy2, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(DFName[branchString]+DFName.Boson_Pt)**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])


def MeanDeviation_PV(branchString, labelName, errbars_shift):
    binwidth = (DFName_nVertex.NVertex.values.max() - DFName_nVertex.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=-(DFName_nVertex[branchString]+DFName_nVertex.Boson_Pt))
    sy2, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(DFName_nVertex[branchString]+DFName_nVertex.Boson_Pt)**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAResolutionOverpTZ_woutError_perp(branchString, labelName, errbars_shift):
    binwidth = (DFName.Boson_Pt.values.max() - DFName.Boson_Pt.values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName.Boson_Pt, bins=nbins)
    sy, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=-(DFName[branchString]))
    sy2, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(DFName[branchString])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, std, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAResolutionOverpTZ_woutError_perp_RC(branchString, labelName, errbars_shift):
    binwidth = (DFName.Boson_Pt.values.max() - DFName.Boson_Pt.values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName.Boson_Pt, bins=nbins)
    sy, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(DFName[branchString]))
    sy2, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(DFName[branchString])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    sy_resp, _resp = np.histogram(DFName.Boson_Pt, bins=nbins, weights=-(DFName[branchString]+DFName.Boson_Pt))
    mean_resp = sy_resp / n
    plt.errorbar((_[1:] + _[:-1])/2, div0(std,mean_resp), marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])



def Histogram_Deviation_para_pT(branchString, labelName, errbars_shift):
    Mean = np.mean(-(DFName[branchString])-DFName.Boson_Pt.values)
    Std = np.std(-(DFName[branchString])-DFName.Boson_Pt.values)
    if branchString in ['NN_LongZ', 'recoilslimmedMETs_LongZ']:
        plt.hist((-(DFName[branchString])-DFName.Boson_Pt.values), bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift], linewidth=1.5)
    else:
        plt.hist((-(DFName[branchString])-DFName.Boson_Pt.values), bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift])


def Histogram_Deviation_perp_pT(branchString, labelName, errbars_shift):
    Mean = np.mean((DFName[branchString]))
    Std = np.std((DFName[branchString]))
    if branchString in ['NN_PerpZ', 'recoilslimmedMETs_PerpZ']:
        plt.hist(((DFName[branchString])), bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift], linewidth=1.5)
    else:
        plt.hist(((DFName[branchString])), bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift])

def Hist_Response(branchString, labelName, errbars_shift):
    Response = np.divide(-(DFName[branchString]), DFName['Boson_Pt'])
    Response = Response[~np.isnan(Response)]
    Mean = np.mean(Response)
    Std = np.std(Response)

    plt.hist(Response, bins=nbinsHist, range=[ResponseMin, ResponseMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift])

def Hist_InvMET(branchString, labelName, errbars_shift):
    Response2 = np.divide(-(DFName[branchString]), DFName['Boson_Pt'])
    Response = np.divide(1, DFName['Boson_Pt'])
    Response = Response[~np.isnan(Response2)]
    Mean = np.mean(Response)
    Std = np.std(Response)

    plt.hist(Response, bins=nbinsHist, range=[ResponseMin, ResponseMax], label='$\\frac{1}{\mathrm{MET}}$, %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift])


def Hist_Resolution_para(branchString, labelName, errbars_shift):
    Mean = np.mean((-(DFName[branchString])-DFName['Boson_Pt']))
    Std = np.std((-(DFName[branchString])-DFName['Boson_Pt']))
    plt.hist((-(DFName[branchString])-DFName['Boson_Pt']), bins=nbinsHist, range=[ResolutionParaMin, ResolutionParaMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift])


def Hist_Resolution_perp(branchString, labelName, errbars_shift):
    Mean = np.mean(DFName[branchString])
    Std = np.std(DFName[branchString])
    plt.hist(DFName[branchString]-Mean, bins=nbinsHist, range=[ResolutionPerpMin, ResolutionPerpMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift])


def Histogram_Deviation_para_Bin(branchString, labelName, bin):
    Mean = np.mean(-(DFName[branchString])-DFName.Boson_Pt.values)
    Std = np.std(-(DFName[branchString])-DFName.Boson_Pt.values)
    #n, _ = np.histogram(-(DFName[branchString])-DFName.Boson_Pt.values, bins=nbinsHistBin)
    plt.hist((-(DFName[branchString])-DFName.Boson_Pt.values), bins=nbinsHistBin, range=(_[bin], _[bin+1]), label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors2[bin])


def Histogram_Response(branchString, labelName, bin):
    Mean = np.mean(np.divide(-(DFName[branchString]),DFName.Boson_Pt.values))
    Std = np.std(np.divide(-(DFName[branchString]),DFName.Boson_Pt.values))
    Reso = np.divide(-(DFName[branchString]),DFName.Boson_Pt.values)
    #n, _ = np.histogram(-(DFName[branchString])-DFName.Boson_Pt.values, bins=nbinsHistBin)
    plt.hist(Reso[~np.isnan(Reso)], bins=nbinsHistBin,  label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors2[bin])



def Histogram_Norm_Comparison(branchStringLong, branchStringPerp, labelName, errbars_shift):
    Norm_ = np.sqrt(np.square(DFName[branchStringLong])+np.square(DFName[branchStringPerp]))
    Mean = np.mean(Norm_-DFName.Boson_Pt.values)
    Std = np.std(Norm_-DFName.Boson_Pt.values)
    if branchStringLong in ['NN_LongZ', 'recoilslimmedMETs_LongZ']:
        plt.hist(Norm_-DFName.Boson_Pt.values, bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift], linewidth=1.5)
    else:
        plt.hist(Norm_-DFName.Boson_Pt.values, bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift])

def Histogram_Angle_Dev(branchStringLong, branchStringPerp, labelName, errbars_shift):
    if branchStringLong == 'NN_LongZ':
        r_, phi_ = kar2pol(DFName[branchStringLong], DFName[branchStringPerp])
        phi_ = angularrange(phi_+np.pi)
        Mean = np.mean(phi_)
        Std = np.std(phi_)
    else:
        r_, phi_ = kar2pol(DFName[branchStringLong], DFName[branchStringPerp])
        phi_ = angularrange(phi_ +np.pi)
        Mean = np.mean(phi_)
        Std = np.std(phi_)
    if branchStringLong in ['NN_LongZ', 'recoilslimmedMETs_Pt']:
        plt.hist(phi_[~np.isnan(phi_)], bins=nbinsHist, range=[-np.pi, np.pi], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift], linewidth=1.5)
    else:
        plt.hist(phi_[~np.isnan(phi_)], bins=nbinsHist, range=[-np.pi, np.pi], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift])

def Histogram_Angle(branchStringLong, branchStringPerp, labelName, errbars_shift):
    r_, phi_ = kar2pol(DFName[branchStringLong], DFName[branchStringPerp])
    phi_ = angularrange(phi_ +np.pi)
    Mean = np.mean(phi_)
    Std = np.std(phi_)
    if branchStringLong in ['NN_LongZ', 'recoilslimmedMETs_LongZ']:
        plt.hist(phi_[~np.isnan(phi_)], bins=nbinsHist, range=[-np.pi, np.pi], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift], linewidth=1.5)
    else:
        plt.hist(phi_[~np.isnan(phi_)], bins=nbinsHist, range=[-np.pi, np.pi], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift])


def Histogram_Norm(branchStringLong, branchStringPerp, labelName, errbars_shift):
    Norm_ = np.sqrt(np.square(DFName[branchStringLong])+np.square(DFName[branchStringPerp]))
    Mean = np.mean(Norm_)
    Std = np.std(Norm_)
    if branchStringLong in ['NN_LongZ', 'recoilslimmedMETs_LongZ']:
        plt.hist(Norm_, bins=nbinsHist, range=[0, 75], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift], linewidth=1.5)
    else:
        plt.hist(Norm_, bins=nbinsHist, range=[0, 75], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift])

def Histogram_Norm_Pt(branchStringLong, labelName, errbars_shift):
    Norm_ = DFName.Boson_Pt.values
    Mean = np.mean(DFName.Boson_Pt.values)
    Std = np.std(Norm_)
    plt.hist(Norm_, bins=nbinsHist, range=[0, 75], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift], linewidth=1.5)


def Histogram_Deviation_para_PV(branchString, labelName, errbars_shift):
    Mean = np.mean(-(DFName_nVertex[branchString])-DFName_nVertex.Boson_Pt.values)
    Std = np.std(-(DFName_nVertex[branchString])-DFName_nVertex.Boson_Pt.values)
    if branchString in ['NN_LongZ', 'recoilslimmedMETs_LongZ']:
        plt.hist((-(DFName_nVertex[branchString])-DFName_nVertex.Boson_Pt.values), bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift], linewidth=1.5)
    else:
        plt.hist((-(DFName_nVertex[branchString])-DFName_nVertex.Boson_Pt.values), bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift])


def Histogram_Deviation_perp_PV(branchString, labelName, errbars_shift):
    Mean = np.mean((DFName_nVertex[branchString]))
    Std = np.std((DFName_nVertex[branchString]))
    if branchString in ['NN_PerpZ', 'recoilslimmedMETs_PerpZ']:
        plt.hist(((DFName_nVertex[branchString])), bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift], linewidth=1.5)
    else:
        plt.hist(((DFName_nVertex[branchString])), bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift])

def Hist_LongZ(branchString, labelName, errbars_shift):
    if branchString == 'Boson_Pt':
        Mean = np.mean((DFName_nVertex[branchString]))
        Std = np.std((DFName_nVertex[branchString]))
        plt.hist(((DFName_nVertex[branchString])), bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift], linewidth=1.5)

    else:
        if branchString in ['NN_LongZ', 'recoilslimmedMETs_LongZ']:
            Mean = np.mean(-(DFName_nVertex[branchString]))
            Std = np.std(-(DFName_nVertex[branchString]))
            plt.hist((-(DFName_nVertex[branchString])), bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift], linewidth=1.5)
        else:
            Mean = np.mean(-(DFName_nVertex[branchString]))
            Std = np.std(-(DFName_nVertex[branchString]))
            plt.hist((-(DFName_nVertex[branchString])), bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift])


def Hist_PerpZ(branchString, labelName, errbars_shift):
    Mean = np.mean((DFName_nVertex[branchString]))
    Std = np.std((DFName_nVertex[branchString]))
    if branchString in ['NN_PerpZ', 'recoilslimmedMETs_PerpZ']:
        plt.hist(((DFName_nVertex[branchString])), bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift], linewidth=1.5)
    else:
        plt.hist(((DFName_nVertex[branchString])), bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift])


def Histogram_relResponse(branchString, labelName, errbars_shift):
    binwidth = ((-(DFName[branchString])-DFName.Boson_Pt.values).max() - (-(DFName[branchString])-DFName.Boson_Pt.values).min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram((-(DFName[branchString])-DFName.Boson_Pt.values), bins=nbins_relR)
    plt.hist(div0((-(DFName[branchString])-DFName.Boson_Pt.values),DFName.Boson_Pt.values), bins=nbinsHist, range=[-20, 20], label=labelName, histtype='step', ec=colors[errbars_shift])

def Histogram_relResponse_PV(branchString, labelName, errbars_shift):
    binwidth = ((-(DFName_nVertex[branchString])-DFName_nVertex.Boson_Pt.values).max() - (-(DFName_nVertex[branchString])-DFName_nVertex.Boson_Pt.values).min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram((-(DFName_nVertex[branchString])-DFName_nVertex.Boson_Pt.values), bins=nbins)
    plt.hist(div0((-(DFName_nVertex[branchString])-DFName_nVertex.Boson_Pt.values), DFName_nVertex.Boson_Pt.values), bins=nbinsHist, range=[-20, 20], label=labelName, histtype='step', ec=colors[errbars_shift])



def NN_Response_pT( labelName, errbars_shift):
    binwidth = (DFName.Boson_Pt.values.max() - DFName.Boson_Pt.values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName.Boson_Pt, bins=nbins)
    sy, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=-(NN_LongZ/DFName.Boson_Pt))
    sy2, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(NN_LongZ/DFName.Boson_Pt)**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def NN_Response_PV(branchString, labelName, errbars_shift):
    binwidth = (DFName_nVertex.NVertex.values.max() - DFName_nVertex.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=-(DFName_nVertex[branchString]/DFName_nVertex.Boson_Pt))
    sy2, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(DFName_nVertex[branchString]/DFName_nVertex.Boson_Pt)**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAResponseOverpTZ_wError(branchString, labelName, errbars_shift):
    binwidth = (DFName.Boson_Pt.values.max() - DFName.Boson_Pt.values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName.Boson_Pt, bins=nbins)
    sy, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(-(DFName[branchString])/DFName.Boson_Pt))
    sy2, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(DFName[branchString]/DFName.Boson_Pt)**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    if errbars_shift==5: errbars_shift2 = 0
    elif errbars_shift==6: errbars_shift2 = 1
    elif errbars_shift==1: errbars_shift2 = 0
    plt.errorbar((_[:-1]+(_[1:]-_[:-1])/3*errbars_shift2), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAResponseOverNVertex_wError(branchString, labelName, errbars_shift):
    binwidth = (DFName_nVertex.NVertex.values.max() - DFName_nVertex.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(-(DFName_nVertex[branchString])/np.abs(DFName_nVertex.Boson_Pt)))
    sy2, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(DFName_nVertex[branchString]/DFName_nVertex.Boson_Pt)**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    if errbars_shift==5: errbars_shift2 = 0
    elif errbars_shift==6: errbars_shift2 = 1
    elif errbars_shift==1: errbars_shift2 = 0

    plt.errorbar((_[:-1]+(_[1:]-_[:-1])/3*errbars_shift2), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])
    #plt.plot((_[1:] + _[:-1])/2, mean, marker='.', label=labelName, linestyle="None", color=MVAcolors[errbars_shift])
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])


def plotMVAResolutionOverpTZ_woutError_para(branchString, labelName, errbars_shift):
    binwidth = (DFName.Boson_Pt.values.max() - DFName.Boson_Pt.values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName.Boson_Pt, bins=nbins)
    sy, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=-(DFName[branchString]+DFName.Boson_Pt))
    sy2, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(DFName[branchString]+DFName.Boson_Pt)**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, std, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAResolutionOverNVertex_woutError_para(branchString, labelName, errbars_shift):
    binwidth = (DFName_nVertex.NVertex.values.max() - DFName_nVertex.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=-(DFName_nVertex[branchString]+DFName_nVertex.Boson_Pt))
    sy2, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(DFName_nVertex[branchString]+DFName_nVertex.Boson_Pt)**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, std, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAResolutionOverpTZ_woutError_para_RC(branchString, labelName, errbars_shift):
    binwidth = (DFName.Boson_Pt.values.max() - DFName.Boson_Pt.values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName.Boson_Pt, bins=nbins)
    sy, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=-(DFName[branchString]+DFName.Boson_Pt))
    sy2, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(DFName[branchString]+DFName.Boson_Pt)**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    sy_resp, _resp = np.histogram(DFName.Boson_Pt, bins=nbins, weights=-(DFName[branchString]+DFName.Boson_Pt))
    mean_resp = sy_resp / n
    plt.errorbar((_[1:] + _[:-1])/2, div0(std,mean_resp), marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAResolutionOverNVertex_woutError_para_RC(branchString, labelName, errbars_shift):
    binwidth = (DFName_nVertex.NVertex.values.max() - DFName_nVertex.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=-(DFName_nVertex[branchString]+DFName_nVertex.Boson_Pt))
    sy2, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(DFName_nVertex[branchString]+DFName_nVertex.Boson_Pt)**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    sy_resp, _resp = np.histogram(DFName_nVertex.Boson_Pt, bins=nbinsVertex, weights=-(DFName_nVertex[branchString]+DFName_nVertex.Boson_Pt))
    mean_resp = sy_resp / n
    plt.errorbar((_[1:] + _[:-1])/2, div0(std,mean_resp), marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])


def plotMVAResolutionOverpTZ_woutError_perp(branchString, labelName, errbars_shift):
    binwidth = (DFName.Boson_Pt.values.max() - DFName.Boson_Pt.values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName.Boson_Pt, bins=nbins)
    sy, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=-(DFName[branchString]))
    sy2, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(DFName[branchString])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, std, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAResolutionOverNVertex_woutError_perp(branchString, labelName, errbars_shift):
    binwidth = (DFName_nVertex.NVertex.values.max() - DFName_nVertex.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=-(DFName_nVertex[branchString]))
    sy2, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(DFName_nVertex[branchString])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, std, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAResolutionOverpTZ_woutError_perp_RC(branchString, labelName, errbars_shift):
    binwidth = (DFName.Boson_Pt.values.max() - DFName.Boson_Pt.values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName.Boson_Pt, bins=nbins)
    sy, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=-(DFName[branchString]))
    sy2, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(DFName[branchString])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    sy_resp, _resp = np.histogram(DFName.Boson_Pt, bins=nbins, weights=-(DFName[branchString]+DFName.Boson_Pt))
    mean_resp = sy_resp / n
    plt.errorbar((_[1:] + _[:-1])/2, div0(std,mean_resp), marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAResolutionOverNVertex_woutError_perp_RC(branchString, labelName, errbars_shift):
    binwidth = (DFName_nVertex.NVertex.values.max() - DFName_nVertex.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(DFName_nVertex[branchString]))
    sy2, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(DFName_nVertex[branchString])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    sy_resp, _resp = np.histogram(DFName_nVertex.Boson_Pt, bins=nbinsVertex, weights=-(DFName_nVertex[branchString]+DFName_nVertex.Boson_Pt))
    mean_resp = sy_resp / n
    plt.errorbar((_[1:] + _[:-1])/2, div0(std,mean_resp), marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def Histogram_Deviation_perp(branchString, labelName, errbars_shift):
    Mean = np.mean(DFName[branchString])
    Std = np.std(DFName[branchString])
    plt.hist(DFName[branchString], bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors[errbars_shift])

def Mean_Std_Deviation_pTZ_para(branchString, labelName, errbars_shift):
    binwidth = (DFName.Boson_Pt.values.max() - DFName.Boson_Pt.values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName.Boson_Pt, bins=nbins)
    sy, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(-(DFName[branchString])-DFName.Boson_Pt.values))
    sy2, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(-(DFName[branchString])-DFName.Boson_Pt.values)**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])
    if errbars_shift==5: errbars_shift2 = 0
    elif errbars_shift==6: errbars_shift2 = 1
    elif errbars_shift==1: errbars_shift2 = 0
    plt.errorbar((_[:-1]+(_[1:]-_[:-1])/3*errbars_shift2), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])

def Mean_Std_Deviation_pTZ_perp(branchString, labelName, errbars_shift):
    binwidth = (DFName.Boson_Pt.values.max() - DFName.Boson_Pt.values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName.Boson_Pt, bins=nbins)
    sy, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(DFName[branchString]))
    sy2, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(DFName[branchString])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])
    if errbars_shift==5: errbars_shift2 = 0
    elif errbars_shift==6: errbars_shift2 = 1
    elif errbars_shift==1: errbars_shift2 = 0
    plt.errorbar((_[:-1]+(_[1:]-_[:-1])/3*errbars_shift2), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])

def Mean_Std_Deviation_PV_para(branchString, labelName, errbars_shift):
    binwidth = (DFName_nVertex.NVertex.values.max() - DFName_nVertex.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(-(DFName_nVertex[branchString])-DFName_nVertex.NVertex.values))
    sy2, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(-(DFName_nVertex[branchString])-DFName_nVertex.NVertex.values)**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])
    if errbars_shift==5: errbars_shift2 = 0
    elif errbars_shift==6: errbars_shift2 = 1
    elif errbars_shift==1: errbars_shift2 = 0
    plt.errorbar((_[:-1]+(_[1:]-_[:-1])/3*errbars_shift2), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])

def Mean_Std_Deviation_PV_perp(branchString, labelName, errbars_shift):
    binwidth = (DFName_nVertex.NVertex.values.max() - DFName_nVertex.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(DFName_nVertex[branchString]))
    sy2, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(DFName_nVertex[branchString])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])
    if errbars_shift==5: errbars_shift2 = 0
    elif errbars_shift==6: errbars_shift2 = 1
    elif errbars_shift==1: errbars_shift2 = 0
    plt.errorbar((_[:-1]+(_[1:]-_[:-1])/3*errbars_shift2), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])





def getPlotsOutput(inputD, filesD, plotsD,DFName, DFName_nVertex):



    #Plot settings
    NPlotsLines = 6
    MVA_NPlotsLines = 3
    pTRangeString = '$0\ \mathrm{GeV} < p_{T}^Z \leq 200\ \mathrm{GeV}$ \n $\mathrm{\# Vertex} \leq 50$'
    pTRangeString_Tresh = '$1\ \mathrm{GeV} < p_{T}^Z \leq 200\ \mathrm{GeV}$ \n $\mathrm{\# Vertex} \leq 50$'
    pTRangeStringNVertex = '$0\ \mathrm{GeV} < p_{T}^Z \leq 200\ \mathrm{GeV}$ \n $\mathrm{\# Vertex} \leq 50$'
    LegendTitle = '$\mathrm{Summer\ 17\ campaign}$' '\n'  '$\mathrm{Z \  \\rightarrow \ \mu \mu}$'
    NPlotsLines=7
    colors = cm.brg(np.linspace(0, 1, NPlotsLines))
    colors_InOut = cm.brg(np.linspace(0, 1, 8))
    colors = colors_InOut
    MVAcolors =  colors
    ylimResMin, ylimResMax = 7.5 , 50
    ylimResMVAMin, ylimResMax = 5 , 35
    ylimResMVAMin_RC, ylimResMax_RC = 0 , 50
    ResponseMin, ResponseMax = -1, 3
    ResponseMinErr, ResponseMaxErr = -0.5, 2.5








    #NN_mode='kart'

    ################################MVA Output ################################
    nbinsVertex = 10
    #########Response u_para/pTZ ###########

    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    plotMVAResponseOverpTZ_woutError('LongZCorrectedRecoil_LongZ', 'GBRT', 5)
    plotMVAResponseOverpTZ_woutError('NN_LongZ', 'NN', 6)
    plotMVAResponseOverpTZ_woutError('recoilslimmedMETs_LongZ', 'PF', 1)
    plt.plot([0, 200], [1, 1], color='k', linestyle='--', linewidth=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.xlabel('$p_{T}^Z $ in GeV')
    plt.ylabel('$\\langle \\frac{U_{\parallel}}{p_{T}^Z} \\rangle$ ')
    #plt.title('Response $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(ResponseMin, ResponseMax)
    plt.savefig("%sResponse_pT.png"%(plotsD), bbox_inches="tight")
    plt.close()

    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    plotMVANormOverpTZ('LongZCorrectedRecoil_Pt', 'GBRT', 5)
    plotMVANormOverpTZ('NN_LongZ', 'NN', 6)
    plotMVANormOverpTZ('recoilslimmedMETs_Pt', 'PF', 1)
    plt.plot([0, 200], [1, 1], color='k', linestyle='--', linewidth=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.xlabel('$p_{T}^Z $ in GeV')
    plt.ylabel('$\\langle \\frac{U_{\parallel}}{p_{T}^Z} \\rangle$ ')
    #plt.title('Response $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(ResponseMin, ResponseMax)
    plt.savefig("%sNorm_pT.png"%(plotsD), bbox_inches="tight")
    plt.close()


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    plotMVANormOverpTZ_wErr('LongZCorrectedRecoil_Pt', 'GBRT', 5)
    plotMVANormOverpTZ_wErr('NN_LongZ', 'NN', 6)
    plotMVANormOverpTZ_wErr('recoilslimmedMETs_Pt', 'PF', 1)
    plt.plot([0, 200], [1, 1], color='k', linestyle='--', linewidth=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.xlabel('$p_{T}^Z $ in GeV')
    plt.ylabel('$\\langle \\frac{U_{\parallel}}{p_{T}^Z} \\rangle$ ')
    #plt.title('Response $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(ResponseMin, ResponseMax)
    plt.savefig("%sNorm_pT_werr.png"%(plotsD), bbox_inches="tight")
    plt.close()


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    plotMVAAngularOverpTZ('LongZCorrectedRecoil_LongZ','LongZCorrectedRecoil_PerpZ', 'GBRT', 5)
    plotMVAAngularOverpTZ('NN_LongZ','NN_PerpZ', 'NN', 6)
    plotMVAAngularOverpTZ('recoilslimmedMETs_LongZ','recoilslimmedMETs_PerpZ', 'PF', 1)
    plt.plot([0, 200], [0, 0], color='k', linestyle='--', linewidth=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.xlabel('$p_{T}^Z $ in GeV')
    plt.ylabel('$\\langle \\frac{U_{\parallel}}{p_{T}^Z} \\rangle$ ')
    #plt.title('Response $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(ResponseMin, ResponseMax)
    plt.savefig("%sDelta_Alpha_pT.png"%(plotsD), bbox_inches="tight")
    plt.close()


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    plotMVAAngularOverpTZ_wErr('LongZCorrectedRecoil_LongZ','LongZCorrectedRecoil_PerpZ', 'GBRT', 5)
    plotMVAAngularOverpTZ_wErr('NN_LongZ','NN_PerpZ', 'NN', 6)
    plotMVAAngularOverpTZ_wErr('recoilslimmedMETs_LongZ','recoilslimmedMETs_PerpZ', 'PF', 1)
    plt.plot([0, 200], [0, 0], color='k', linestyle='--', linewidth=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.xlabel('$p_{T}^Z $ in GeV')
    plt.ylabel('$\\langle \\frac{U_{\parallel}}{p_{T}^Z} \\rangle$ ')
    #plt.title('Response $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(ResponseMin, ResponseMax)
    plt.savefig("%sDelta_Alpha_pT_werr.png"%(plotsD), bbox_inches="tight")
    plt.close()


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    Hist_Response('LongZCorrectedRecoil_LongZ', 'GBRT', 5)
    Hist_Response('NN_LongZ', 'NN', 6)
    Hist_Response('recoilslimmedMETs_LongZ', 'PF', 1)
    plt.plot([1, 1], [0, 90000], color='k', linestyle='--', linewidth=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.xlabel('$\\frac{U_{\parallel}}{\mathrm{MET}}$')
    plt.ylabel('Counts')
    plt.xlim(ResponseMin, ResponseMax)
    plt.ylim(0, 90000)
    #plt.title('Response $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    plt.xlim(ResponseMin, ResponseMax)
    plt.savefig("%sResponse.png"%(plotsD), bbox_inches="tight")
    plt.close()

    '''
    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    plotMVAResponseOverpTZ_woutError_Tresh('LongZCorrectedRecoil_LongZ', 'GBRT', 5)
    plotMVAResponseOverpTZ_woutError_Tresh('NN_LongZ', 'NN', 6)
    plotMVAResponseOverpTZ_woutError_Tresh('recoilslimmedMETs_LongZ', 'PF', 1)
    plt.plot([0, 200], [1, 1], color='k', linestyle='--', linewidth=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString_Tresh))

    plt.xlabel('$p_{T}^Z $ in GeV')
    plt.ylabel('$\\langle \\frac{U_{\parallel}}{p_{T}^Z} \\rangle$ ')
    #plt.title('Response $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(ResponseMin, ResponseMax)
    plt.savefig("%sResponse_pT_>1.png"%(plotsD), bbox_inches="tight")
    plt.close()
    '''




    '''
    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    plotMVAResponseOverpTZ_wError('LongZCorrectedRecoil_LongZ', 'GBRT', 5)
    plotMVAResponseOverpTZ_wError('NN_LongZ', 'NN', 6)
    plotMVAResponseOverpTZ_wError('recoilslimmedMETs_LongZ', 'PF', 1)
    plt.plot([0, 200], [1, 1], color='k', linestyle='--', linewidth=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.xlabel('$p_{T}^Z $ in GeV')
    plt.ylabel('$\\langle \\frac{U_{\parallel}}{p_{T}^Z} \\rangle$ ')
    #plt.title('Response $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(ResponseMinErr, ResponseMaxErr)
    plt.savefig("%sResponse_pT_wErr.png"%(plotsD), bbox_inches="tight")
    plt.close()


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    plotMVAResponseOverNVertex_woutError('LongZCorrectedRecoil_LongZ', 'GBRT', 5)
    plotMVAResponseOverNVertex_woutError('NN_LongZ', 'NN', 6)
    plotMVAResponseOverNVertex_woutError('recoilslimmedMETs_LongZ', 'PF', 1)
    plt.plot([0, 50], [1, 1], color='k', linestyle='--', linewidth=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    plt.xlabel('#$ \mathrm{PV}$ ')
    plt.ylabel('$\\langle \\frac{U_{\parallel}}{p_{T}^Z} \\rangle$ ')
    #plt.title('Response $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(0, ResponseMax)
    plt.savefig("%sResponse_PV.png"%(plotsD), bbox_inches="tight")
    plt.close()


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    plotMVAResponseOverNVertex_woutError_Tresh('LongZCorrectedRecoil_LongZ', 'GBRT', 5)
    plotMVAResponseOverNVertex_woutError_Tresh('NN_LongZ', 'NN', 6)
    plotMVAResponseOverNVertex_woutError_Tresh('recoilslimmedMETs_LongZ', 'PF', 1)
    plt.plot([0, 50], [1, 1], color='k', linestyle='--', linewidth=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString_Tresh))

    plt.xlabel('#$ \mathrm{PV}$ ')
    plt.ylabel('$\\langle \\frac{U_{\parallel}}{p_{T}^Z} \\rangle$ ')
    #plt.title('Response $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(0, ResponseMax)
    plt.savefig("%sResponse_PV_Tresh.png"%(plotsD), bbox_inches="tight")
    plt.close()



    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    plotMVAResponseOverNVertex_wError('LongZCorrectedRecoil_LongZ', 'GBRT', 5)
    plotMVAResponseOverNVertex_wError('NN_LongZ', 'NN', 6)
    plotMVAResponseOverNVertex_wError('recoilslimmedMETs_LongZ', 'PF', 1)
    plt.plot([0, 50], [1, 1], color='k', linestyle='--', linewidth=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    plt.xlabel('#$ \mathrm{PV}$ ')
    plt.ylabel('$\\langle \\frac{U_{\parallel}}{p_{T}^Z} \\rangle$ ')
    #plt.title('Response $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(ResponseMinErr, ResponseMaxErr)
    plt.savefig("%sResponse_PV_wErr.png"%(plotsD), bbox_inches="tight")
    plt.close()

    ##########Resolutions #########
    ######## u para ########
    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    plotMVAResolutionOverpTZ_woutError_para('LongZCorrectedRecoil_LongZ', 'GBRT', 5)
    plotMVAResolutionOverpTZ_woutError_para('NN_LongZ', 'NN', 6)
    plotMVAResolutionOverpTZ_woutError_para('recoilslimmedMETs_LongZ', 'PF', 1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.xlabel('$p_{T}^Z $ in GeV')
    plt.ylabel('$\sigma \\left( U_{\parallel}- p_{T}^Z \\right) $ in GeV')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\parallel}}{p_{T}^Z} \\right) $ in GeV')
    #plt.title('Resolution $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(ylimResMVAMin, ylimResMax)
    plt.savefig("%sResolution_para_pT.png"%(plotsD), bbox_inches="tight")
    plt.close()
    '''



    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    Hist_Resolution_para('LongZCorrectedRecoil_LongZ', 'GBRT', 5)
    Hist_Resolution_para('NN_LongZ', 'NN', 6)
    Hist_Resolution_para('recoilslimmedMETs_LongZ', 'PF', 1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.xlabel('$U_{\parallel}-\mathrm{MET}$')
    plt.ylabel('Counts')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\parallel}}{p_{T}^Z} \\right) $ in GeV')
    #plt.title('Resolution $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    plt.xlim(ResolutionParaMin, ResolutionParaMax )
    plt.savefig("%sHist_Resolution_para.png"%(plotsD), bbox_inches="tight")
    plt.close()


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    Hist_Resolution_perp('LongZCorrectedRecoil_PerpZ', 'GBRT', 5)
    Hist_Resolution_perp('NN_PerpZ', 'NN', 6)
    Hist_Resolution_perp('recoilslimmedMETs_PerpZ', 'PF', 1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.xlabel('$U_{\perp}$ in GeV')
    plt.ylabel('Counts')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\parallel}}{p_{T}^Z} \\right) $ in GeV')
    #plt.title('Resolution $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    plt.xlim(ResolutionPerpMin, ResolutionPerpMax )
    plt.savefig("%sHist_Resolution_perp.png"%(plotsD), bbox_inches="tight")
    plt.close()

    '''
    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    MeanDeviation_Pt('LongZCorrectedRecoil_LongZ', 'GBRT', 5)
    MeanDeviation_Pt('NN_LongZ', 'NN', 6)
    MeanDeviation_Pt('recoilslimmedMETs_LongZ', 'PF', 1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.xlabel('$p_{T}^Z $ in GeV')
    plt.ylabel('$\\langle U_{\parallel}- p_{T}^Z \\rangle$ in GeV')
    #plt.title('Mean Deviation $U_{\parallel}-p_T^Z$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(-20, 10)
    plt.savefig("%sDelta_para_pT.png"%(plotsD), bbox_inches="tight")
    plt.close()


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    plotMVAResolutionOverNVertex_woutError_para('LongZCorrectedRecoil_LongZ', 'GBRT', 5)
    plotMVAResolutionOverNVertex_woutError_para('NN_LongZ', 'NN', 6)
    plotMVAResolutionOverNVertex_woutError_para('recoilslimmedMETs_LongZ', 'PF', 1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    plt.xlabel('#$ \mathrm{PV}$ ')
    plt.ylabel('$\sigma \\left( U_{\parallel}- p_{T}^Z \\right) $ in GeV')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\parallel}}{p_{T}^Z} \\right) $ in GeV')
    #plt.title('Resolution $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(ylimResMVAMin, ylimResMax)
    plt.savefig("%sResolution_para_PV.png"%(plotsD), bbox_inches="tight")
    plt.close()





    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    MeanDeviation_PV('LongZCorrectedRecoil_LongZ', 'GBRT', 5)
    MeanDeviation_PV('NN_LongZ', 'NN', 6)
    MeanDeviation_PV('recoilslimmedMETs_LongZ', 'PF', 1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    plt.xlabel('#$ \mathrm{PV}$ ')
    plt.ylabel('$\\langle U_{\parallel} - \mathrm{MET} \\rangle$ in GeV')
    #plt.title('Mean Deviation $U_{\parallel}-p_T^Z$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(-20, 20)
    plt.savefig("%sDelta_para_PV.png"%(plotsD), bbox_inches="tight")
    plt.close()


    #######u perp ######
    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    plotMVAResolutionOverpTZ_woutError_perp('LongZCorrectedRecoil_PerpZ', 'GBRT', 5)
    plotMVAResolutionOverpTZ_woutError_perp('NN_PerpZ', 'NN', 6)
    plotMVAResolutionOverpTZ_woutError_perp('recoilslimmedMETs_PerpZ', 'PF', 1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.xlabel('$p_{T}^Z $ in GeV')
    plt.ylabel('$\sigma \\left( U_{\perp} \\right) $ in GeV')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_{T}^Z} \\right) $ in GeV')
    #plt.title('Resolution $U_{\perp}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(ylimResMVAMin, ylimResMax)
    plt.savefig("%sResolution_perp_pT.png"%(plotsD), bbox_inches="tight")
    plt.close()






    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    plotMVAResolutionOverNVertex_woutError_perp('LongZCorrectedRecoil_PerpZ', 'GBRT', 5)
    plotMVAResolutionOverNVertex_woutError_perp('NN_PerpZ', 'NN', 6)
    plotMVAResolutionOverNVertex_woutError_perp('recoilslimmedMETs_PerpZ', 'PF', 1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    plt.xlabel('#$ \mathrm{PV}$ ')
    plt.ylabel('$\sigma \\left( U_{\perp} \\right) $ in GeV')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_{T}^Z} \\right) $ in GeV')
    #plt.title('Resolution $U_{\perp}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(ylimResMVAMin, ylimResMax)
    plt.savefig("%sResolution_perp_PV.png"%(plotsD), bbox_inches="tight")
    plt.close()












    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)


    Histogram_Deviation_para_PV('recoilpatpfNoPUMET_LongZ', 'No PU MET',0)
    Histogram_Deviation_para_PV('recoilpatpfPUCorrectedMET_LongZ', 'PU corrected MET',1)
    Histogram_Deviation_para_PV( 'recoilpatpfPUMET_LongZ', 'PU MET',2)
    Histogram_Deviation_para_PV('recoilpatpfTrackMET_LongZ', 'Track MET',3)
    Histogram_Deviation_para_PV('recoilslimmedMETsPuppi_LongZ', 'Puppi MET',4)
    Histogram_Deviation_para_PV('LongZCorrectedRecoil_LongZ', 'GBRT MET', 5)
    Histogram_Deviation_para_PV('recoilslimmedMETs_LongZ', 'PF MET', 1)
    Histogram_Deviation_para_PV('NN_LongZ', 'NN MET', 6)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.ylabel('Counts')
    plt.xlabel('$ U_{\parallel} - \mathrm{MET}  $ in GeV')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_{T}^Z} \\right) $ in GeV')
    #plt.title('Deviation Histogram parallel')
    #plt.text('$p_T$ range restriction')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    ##plt.ylim(ylimResMVAMin, ylimResMax)
    plt.savefig("%sHist_Delta_para.png"%(plotsD), bbox_inches="tight")
    plt.close()
    '''

    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)


    Histogram_Norm_Comparison('recoilpatpfNoPUMET_LongZ','recoilpatpfNoPUMET_PerpZ', 'No PU MET',0)
    Histogram_Norm_Comparison('recoilpatpfPUCorrectedMET_LongZ', 'recoilpatpfPUCorrectedMET_PerpZ','PU corrected MET',1)
    Histogram_Norm_Comparison( 'recoilpatpfPUMET_LongZ', 'recoilpatpfPUMET_PerpZ', 'PU MET',2)
    Histogram_Norm_Comparison('recoilpatpfTrackMET_LongZ', 'recoilpatpfTrackMET_PerpZ', 'Track MET',3)
    Histogram_Norm_Comparison('recoilslimmedMETsPuppi_LongZ', 'recoilslimmedMETsPuppi_PerpZ', 'Puppi MET',4)
    Histogram_Norm_Comparison('LongZCorrectedRecoil_LongZ', 'LongZCorrectedRecoil_PerpZ', 'GBRT MET', 5)
    Histogram_Norm_Comparison('recoilslimmedMETs_LongZ', 'recoilslimmedMETs_PerpZ', 'PF MET', 1)
    Histogram_Norm_Comparison('NN_LongZ','NN_PerpZ', 'NN MET', 6)


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.ylabel('Counts')
    plt.xlabel('$ |\\vec{U}| - |\\vec{\mathrm{MET}}|  $ in GeV')
    plt.xlim(HistLimMin,HistLimMax)
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_{T}^Z} \\right) $ in GeV')
    #plt.title('Deviation Histogram norm')
    #plt.text('$p_T$ range restriction')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    ##plt.ylim(ylimResMVAMin, ylimResMax)
    plt.savefig("%sHist_Delta_norm.png"%(plotsD), bbox_inches="tight")
    plt.close()


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)


    Histogram_Norm('recoilpatpfNoPUMET_LongZ','recoilpatpfNoPUMET_PerpZ', 'No PU MET',0)
    Histogram_Norm('recoilpatpfPUCorrectedMET_LongZ', 'recoilpatpfPUCorrectedMET_PerpZ','PU corrected MET',1)
    Histogram_Norm( 'recoilpatpfPUMET_LongZ', 'recoilpatpfPUMET_PerpZ', 'PU MET',2)
    Histogram_Norm('recoilpatpfTrackMET_LongZ', 'recoilpatpfTrackMET_PerpZ', 'Track MET',3)
    Histogram_Norm('recoilslimmedMETsPuppi_LongZ', 'recoilslimmedMETsPuppi_PerpZ', 'Puppi MET',4)
    Histogram_Norm('LongZCorrectedRecoil_LongZ', 'LongZCorrectedRecoil_PerpZ', 'GBRT MET', 5)
    Histogram_Norm('recoilslimmedMETs_LongZ', 'recoilslimmedMETs_PerpZ', 'PF MET', 1)
    Histogram_Norm_Pt('Boson_Pt', 'Target MET',7)
    #Histogram_Norm('NN_LongZ','NN_PerpZ', 'NN MET', 6)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.ylabel('Counts')
    plt.xlabel('$ |U|   $ in GeV')
    plt.xlim(0,75)
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_{T}^Z} \\right) $ in GeV')
    #plt.title('Deviation Histogram norm')
    #plt.text('$p_T$ range restriction')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    ##plt.ylim(ylimResMVAMin, ylimResMax)
    plt.savefig("%sHist_norm.png"%(plotsD), bbox_inches="tight")
    plt.close()


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)




    Histogram_Deviation_perp_PV('recoilpatpfNoPUMET_PerpZ', 'No PU MET',0)
    Histogram_Deviation_perp_PV('recoilpatpfPUCorrectedMET_PerpZ', 'PU corrected MET',1)
    Histogram_Deviation_perp_PV( 'recoilpatpfPUMET_PerpZ', 'PU MET',2)
    Histogram_Deviation_perp_PV('recoilpatpfTrackMET_PerpZ', 'Track MET',3)
    Histogram_Deviation_perp_PV('recoilslimmedMETsPuppi_PerpZ', 'Puppi MET',4)
    Histogram_Deviation_perp_PV('LongZCorrectedRecoil_PerpZ', 'GBRT', 5)
    Histogram_Deviation_perp_PV('recoilslimmedMETs_PerpZ', 'PF', 1)
    Histogram_Deviation_perp_PV('NN_PerpZ', 'NN', 6)


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.ylabel('Counts')
    plt.xlabel('$ U_{\perp}  $ in GeV')
    plt.xlim(HistLimMin,HistLimMax)
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_{T}^Z} \\right) $ in GeV')
    #plt.title('Deviation Histogram perpendicular')
    #plt.text('$p_T$ range restriction')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    ##plt.ylim(ylimResMVAMin, ylimResMax)
    plt.savefig("%sHist_perp.png"%(plotsD), bbox_inches="tight")
    plt.close()

    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)


    Hist_LongZ('recoilpatpfNoPUMET_LongZ', 'No PU MET',0)
    Hist_LongZ('recoilpatpfPUCorrectedMET_LongZ', 'PU corrected MET',1)
    Hist_LongZ( 'recoilpatpfPUMET_LongZ', 'PU MET',2)
    Hist_LongZ('recoilpatpfTrackMET_LongZ', 'Track MET',3)
    Hist_LongZ('recoilslimmedMETsPuppi_LongZ', 'Puppi MET',3)
    Hist_LongZ('LongZCorrectedRecoil_LongZ', 'GBRT MET', 5)
    Hist_LongZ('recoilslimmedMETs_LongZ', 'PF MET', 1)
    Hist_LongZ('Boson_Pt', 'Target MET',4)
    Hist_LongZ('NN_LongZ', 'NN MET', 6)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.ylabel('Counts')
    plt.xlabel('$ U_{\parallel}   $ in GeV')
    plt.xlim(HistLimMin,HistLimMax)
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_{T}^Z} \\right) $ in GeV')
    #plt.title(' Histogram parallel component')
    #plt.text('$p_T$ range restriction')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    ##plt.ylim(ylimResMVAMin, ylimResMax)
    plt.savefig("%sHist_para.png"%(plotsD), bbox_inches="tight")
    plt.close()


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)


    Hist_PerpZ('recoilpatpfNoPUMET_PerpZ', 'No PU MET',0)
    Hist_PerpZ('recoilpatpfPUCorrectedMET_PerpZ', 'PU corrected MET',1)
    Hist_PerpZ( 'recoilpatpfPUMET_PerpZ', 'PU MET',2)
    Hist_PerpZ('recoilpatpfTrackMET_PerpZ', 'Track MET',3)
    Hist_PerpZ('recoilslimmedMETsPuppi_PerpZ', 'Puppi MET',4)
    Hist_PerpZ('LongZCorrectedRecoil_PerpZ', 'GBRT MET', 5)
    Hist_PerpZ('recoilslimmedMETs_PerpZ', 'PF MET', 1)
    Hist_PerpZ('NN_PerpZ', 'NN MET', 6)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.ylabel('Counts')
    plt.xlabel('$ U_{\perp}  $ in GeV')
    plt.xlim(HistLimMin,HistLimMax)
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_{T}^Z} \\right) $ in GeV')
    #plt.title(' Histogram perpendicular component')
    #plt.text('$p_T$ range restriction')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    ##plt.ylim(ylimResMVAMin, ylimResMax)
    plt.savefig("%sHist_perp.png"%(plotsD), bbox_inches="tight")
    plt.close()

    '''
    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)
    for Bin in range(0,nbins):
            Histogram_Deviation_para_Bin('NN_LongZ', 'NN', Bin)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.ylabel('Counts')
    plt.xlabel('$ U_{\parallel} - p_T^Z  $ in GeV')
    plt.xlim(HistLimMin,HistLimMax)
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_{T}^Z} \\right) $ in GeV')
    #plt.title('Response Histogram by Bin')
    #plt.text('$p_T$ range restriction')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    ##plt.ylim(ylimResMVAMin, ylimResMax)
    plt.savefig("%sOutput_Response_Hist_Bin.png"%(plotsD), bbox_inches="tight")
    plt.close()
    '''


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)





    Histogram_Deviation_para_PV('recoilpatpfNoPUMET_LongZ', 'No PU MET',0)
    Histogram_Deviation_para_PV('recoilpatpfPUCorrectedMET_LongZ', 'PU corrected MET',1)
    Histogram_Deviation_para_PV( 'recoilpatpfPUMET_LongZ', 'PU MET',2)
    Histogram_Deviation_para_PV('recoilpatpfTrackMET_LongZ', 'Track MET',3)
    Histogram_Deviation_para_PV('recoilslimmedMETsPuppi_LongZ', 'Puppi MET',4)
    Histogram_Deviation_para_PV('LongZCorrectedRecoil_LongZ', 'GBRT MET', 5)
    Histogram_Deviation_para_PV('recoilslimmedMETs_LongZ', 'PF MET', 7)
    Histogram_Deviation_para_PV('NN_LongZ', 'NN MET', 6)


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    plt.ylabel('Counts')
    plt.xlabel('$ U_{\parallel} - \mathrm{MET}  $ in GeV')
    plt.xlim(HistLimMin,HistLimMax)
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_{T}^Z} \\right) $ in GeV')
    #plt.title('Deviation Histogram parallel')
    #plt.text('$p_T$ and $\# PV$ range restriction')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    ##plt.ylim(ylimResMVAMin, ylimResMax)
    plt.savefig("%sHist_Delta_para.png"%(plotsD), bbox_inches="tight")
    plt.close()










    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)



    Histogram_Angle_Dev('recoilpatpfNoPUMET_LongZ','recoilpatpfNoPUMET_PerpZ', 'No PU MET',0)
    Histogram_Angle_Dev('recoilpatpfPUCorrectedMET_LongZ', 'recoilpatpfPUCorrectedMET_PerpZ','PU corrected MET',1)
    Histogram_Angle_Dev( 'recoilpatpfPUMET_LongZ', 'recoilpatpfPUMET_PerpZ', 'PU MET',2)
    Histogram_Angle_Dev('recoilpatpfTrackMET_LongZ', 'recoilpatpfTrackMET_PerpZ', 'Track MET',3)
    Histogram_Angle_Dev('recoilslimmedMETsPuppi_LongZ', 'recoilslimmedMETsPuppi_PerpZ', 'Puppi MET',4)
    Histogram_Angle_Dev('LongZCorrectedRecoil_LongZ', 'LongZCorrectedRecoil_PerpZ', 'GBRT MET', 5)
    Histogram_Angle_Dev('recoilslimmedMETs_LongZ', 'recoilslimmedMETs_PerpZ', 'PF MET', 1)
    Histogram_Angle_Dev('NN_LongZ','NN_PerpZ', 'NN MET', 6)


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    plt.ylabel('Counts')
    plt.xlabel('$ \\Delta \\alpha $ in rad')
    plt.xlim(-np.pi,np.pi)
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_{T}^Z} \\right) $ in GeV')
    #plt.title('Deviation Histogram perp')
    #plt.text('$p_T$ and $\# pT$ range restriction')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    ##plt.ylim(ylimResMVAMin, ylimResMax)
    plt.savefig("%sHist_Delta_phi.png"%(plotsD), bbox_inches="tight")
    plt.close()

    '''
    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)



    Histogram_Angle('recoilpatpfNoPUMET_Phi', 'No PU MET',0)
    Histogram_Angle('recoilpatpfPUCorrectedMET_Phi','PU corrected MET',1)
    Histogram_Angle( 'recoilpatpfPUMET_Phi', 'PU MET',2)
    Histogram_Angle('recoilpatpfTrackMET_Phi', 'Track MET',3)
    Histogram_Angle('recoilslimmedMETsPuppi_Phi', 'Puppi MET',4)
    Histogram_Angle('LongZCorrectedRecoil_Phi', 'GBRT MET', 5)
    Histogram_Angle('recoilslimmedMETs_Phi', 'PF MET', 1)
    Histogram_Angle('NN_LongZ','NN_PerpZ', 'NN MET', 6)


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    plt.ylabel('Counts')
    plt.xlabel('$  \\alpha $ in rad')
    plt.xlim(-np.pi,np.pi)
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_{T}^Z} \\right) $ in GeV')
    #plt.title('Deviation Histogram perp')
    #plt.text('$p_T$ and $\# pT$ range restriction')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    ##plt.ylim(ylimResMVAMin, ylimResMax)
    plt.savefig("%sHist_phi.png"%(plotsD), bbox_inches="tight")
    plt.close()
    '''
    '''
    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    Mean_Std_Deviation_pTZ_para('LongZCorrectedRecoil_LongZ', 'GBRT', 5)
    Mean_Std_Deviation_pTZ_para('NN_LongZ', 'NN', 6)
    Mean_Std_Deviation_pTZ_para('recoilslimmedMETs_LongZ', 'PF', 1)


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    plt.ylabel('$\\langle  U_{\parallel} - \mathrm{MET} \\rangle$')
    plt.xlabel('$\mathrm{MET} $ in GeV')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_{T}^Z} \\right) $ in GeV')
    #plt.title('parallel deviation over $p_T^Z$')
    #plt.text('$p_T$ and $\# PV$ range restriction')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    ##plt.ylim(HistLimMin, HistLimMax)
    plt.savefig("%sDelta_para_Std_para_pT.png"%(plotsD), bbox_inches="tight")
    '''



    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    Histogram_Response('LongZCorrectedRecoil_LongZ', 'GBRT', 5)
    Histogram_Response('NN_LongZ', 'NN', 6)
    Histogram_Response('recoilslimmedMETs_LongZ', 'PF', 1)


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    plt.xlabel('$  \\frac{U_{\parallel}}{p_T^Z}} $')
    plt.ylabel('Counts')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_{T}^Z} \\right) $ in GeV')
    #plt.title('parallel deviation over $p_T^Z$')
    #plt.text('$p_T$ and $\# PV$ range restriction')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    ##plt.ylim(HistLimMin, HistLimMax)
    plt.savefig("%sHist_Response.png"%(plotsD), bbox_inches="tight")




    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    Mean_Std_Deviation_pTZ_perp('LongZCorrectedRecoil_PerpZ', 'GBRT', 5)
    Mean_Std_Deviation_pTZ_perp('NN_PerpZ', 'NN', 6)
    Mean_Std_Deviation_pTZ_perp('recoilslimmedMETs_PerpZ', 'PF', 1)


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    plt.ylabel('$\\langle  U_{\perp} \\rangle$')
    plt.xlabel('$ \mathrm{MET} $ in GeV')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_{T}^Z} \\right) $ in GeV')
    #plt.title('perpendicular deviation over $p_T^Z$')
    #plt.text('$p_T$ and $\# PV$ range restriction')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    ##plt.ylim(HistLimMin, HistLimMax)
    plt.savefig("%sDelta_perp_Std_perp_pT.png"%(plotsD), bbox_inches="tight")


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    Mean_Std_Deviation_PV_para('LongZCorrectedRecoil_LongZ', 'GBRT', 5)
    Mean_Std_Deviation_PV_para('NN_LongZ', 'NN', 6)
    Mean_Std_Deviation_PV_para('recoilslimmedMETs_LongZ', 'PF', 1)


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    plt.ylabel('$\\langle  U_{\parallel} - \mathrm{MET} \\rangle$')
    plt.xlabel('\#PV')
    #plt.title('parallel deviation over \#PV')
    #plt.text('$p_T$ and $\# PV$ range restriction')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    ##plt.ylim(HistLimMin, HistLimMax)
    plt.savefig("%sDelta_para_Std_para_PV.png"%(plotsD), bbox_inches="tight")


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    Mean_Std_Deviation_PV_perp('LongZCorrectedRecoil_PerpZ', 'GBRT', 5)
    Mean_Std_Deviation_PV_perp('NN_PerpZ', 'NN', 6)
    Mean_Std_Deviation_PV_perp('recoilslimmedMETs_PerpZ', 'PF', 1)


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    plt.ylabel('$\\langle  U_{\perp} \\rangle$')
    plt.xlabel('\#PV')
    #plt.title('parallel deviation over \#PV')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    ##plt.ylim(HistLimMin, HistLimMax)
    plt.savefig("%sDelta_perp_Std_perp_PV.png"%(plotsD), bbox_inches="tight")

    plt.clf()
    plt.figure()
    #plt.suptitle('x-Korrektur Prediction-Target ')
    plt.xlabel("$ \Delta \\alpha $")
    plt.ylabel("Response")
    NN_r, NN_phi = kar2pol(-DFName['NN_LongZ'], DFName['NN_PerpZ'])
    Response = np.divide(-DFName['NN_LongZ'], DFName['Boson_Pt'] )
    heatmap, xedges, yedges = np.histogram2d(  NN_phi, Response,  bins=50,
                                             range=[[-np.pi/2,np.pi/2],
                                                    [0,2]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #plt.clf()
    HM =plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm(), aspect='auto')
    plt.colorbar(HM)
    plt.legend()
    plt.savefig("%sHM_Response_NN_Delta_Alpha.png"%(plotsD), bbox_inches="tight")


    plt.clf()
    plt.figure()
    #plt.suptitle('x-Korrektur Prediction-Target ')
    plt.xlabel("$ \Delta \\alpha $")
    plt.ylabel("Response")
    dPhi = np.linspace(-np.pi, np.pi, 200)
    Response_pTperfect = np.cos(dPhi)
    mean_Response('NN_LongZ', 'NN_PerpZ', 'NN', 0)
    mean_Response('recoilslimmedMETs_LongZ', 'recoilslimmedMETs_PerpZ', 'PF', 1)
    plt.plot(np.linspace(-np.pi, np.pi, 2000), np.cos(np.linspace(-np.pi, np.pi, 2000)), linewidth=2, markersize=12, label='perfect guess $p_T$')
    plt.legend()
    plt.ylim(-20,20)
    plt.rcParams['agg.path.chunksize'] = 10000
    plt.savefig("%sNN_Delta_Alpha_perfect_Guess.png"%(plotsD), bbox_inches="tight")

    plt.clf()
    plt.figure()
    #plt.suptitle('x-Korrektur Prediction-Target ')
    plt.xlabel("$ \Delta \\alpha $")
    plt.ylabel("Response")
    dPhi = np.linspace(-np.pi/180*10, np.pi/180*10, 200)
    Response_pTperfect = np.cos(dPhi)
    mean_Response_CR('NN_LongZ', 'NN_PerpZ', 'NN', 0)
    mean_Response_CR('recoilslimmedMETs_LongZ', 'recoilslimmedMETs_PerpZ', 'PF', 1)
    plt.plot(np.linspace(-np.pi/180*10, np.pi/180*10, 2000), np.cos(np.linspace(-np.pi/180*10, np.pi/180*10, 2000)), linewidth=2, markersize=12, label='perfect guess $p_T$')
    plt.ylim(-5,5)
    plt.legend()
    plt.savefig("%sResponse_NN_Delta_Alpha_CR.png"%(plotsD), bbox_inches="tight")


    plt.clf()
    plt.figure()
    #plt.suptitle('x-Korrektur Prediction-Target ')
    plt.xlabel("$ \Delta \\alpha $")
    plt.ylabel("Response")
    PF_r, PF_phi = kar2pol(-DFName['recoilpatpfPUMET_LongZ'], DFName['recoilpatpfPUMET_PerpZ'])
    Response = np.divide(-DFName['recoilpatpfPUMET_LongZ'], DFName['Boson_Pt'] )
    heatmap, xedges, yedges = np.histogram2d(  PF_phi, Response,  bins=50,
                                             range=[[-np.pi,np.pi],
                                                    [0,2]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #plt.clf()
    HM =plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm(), aspect='auto')
    plt.colorbar(HM)
    plt.legend()
    plt.savefig("%sHM_Response_PF_Delta_Alpha.png"%(plotsD), bbox_inches="tight")

    '''
    plt.clf()
    plt.figure()
    plt.xlabel("$ \Delta \\alpha $")
    plt.ylabel("$|U|-|\mathrm{MET}|$")
    PF_r, PF_phi = kar2pol(-DFName['recoilpatpfPUMET_LongZ'], DFName['recoilpatpfPUMET_PerpZ'])
    Response = np.divide(-DFName['recoilpatpfPUMET_LongZ'], DFName['Boson_Pt'] )
    x=PF_phi
    y=DFName['recoilpatpfPUMET_Pt']-DFName['Boson_Pt']
    z=Response
    x=np.unique(x)
    y=np.unique(y)
    X,Y = np.meshgrid(x,y)
    Z=z.reshape(len(y),len(x))
    plt.pcolormesh(X,Y,Z)
    plt.show()
    plt.legend()
    plt.savefig("%sCM_PF_Response_Delta_Alpha_Delta_pT.png"%(plotsD), bbox_inches="tight")


    plt.clf()
    plt.figure()
    plt.xlabel("$ \Delta \\alpha $")
    plt.ylabel("$|U|-|\mathrm{MET}|$")
    PF_r, PF_phi = kar2pol(-DFName['NN_LongZ'], DFName['NN_PerpZ'])
    Response = np.divide(-DFName['NN_LongZ'], DFName['Boson_Pt'] )
    x=PF_phi
    y=np.sqrt(np.multiply(DFName['NN_LongZ'], DFName['NN_LongZ'])+ np.multiply(DFName['NN_PerpZ'], DFName['NN_PerpZ']))-DFName['Boson_Pt']
    z=Response
    x=np.unique(x)
    y=np.unique(y)
    X,Y = np.meshgrid(x,y)
    Z=z.reshape(len(y),len(x))
    plt.pcolormesh(X,Y,Z)
    plt.show()
    plt.legend()
    plt.savefig("%sCM_NN_Response_Delta_Alpha_Delta_pT.png"%(plotsD), bbox_inches="tight")
    '''
    '''
    plt.clf()
    plt.figure()
    plt.xlabel("$ \Delta \\alpha $")
    plt.ylabel("$|U|-|\mathrm{MET}|$")
    y=np.sqrt(np.multiply(DFName['recoilslimmedMETs_LongZ'], DFName['recoilslimmedMETs_LongZ'])+ np.multiply(DFName['recoilslimmedMETs_PerpZ'], DFName['recoilslimmedMETs_PerpZ']))-DFName['Boson_Pt']
    delta_alpha=np.linspace(-np.pi,np.pi,201)
    delta_pT=np.linspace(y.min(),y.max(),201)
    PF_r, PF_phi = kar2pol(-DFName['recoilslimmedMETs_LongZ'], DFName['recoilslimmedMETs_PerpZ'])
    Response = np.divide(-DFName['recoilslimmedMETs_LongZ'], DFName['Boson_Pt'] )
    x=PF_phi
    HM2=np.empty([200, 200])
    for i in range(0,200):
        for j in range(0,200):
            HM2[i,j]=np.mean(Response[(PF_phi>=delta_alpha[i]) & (PF_phi<delta_alpha[i+1]) & (y>=delta_pT[j]) & (y<delta_pT[j+1])])
    heatmap, xedges, yedges = HM2, delta_alpha, delta_pT
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #plt.clf()
    HM =plt.imshow(heatmap.T, extent=[-np.pi/2,np.pi/2, -50, 50], origin='lower', norm=LogNorm(), aspect='auto')
    plt.colorbar(HM)
    plt.xlim(-np.pi/2,np.pi/2)
    plt.ylim(-50,50)
    plt.show()
    plt.legend()
    plt.savefig("%sHM_PF_Response_Delta_Alpha_Delta_pT.png"%(plotsD), bbox_inches="tight")
    '''
    '''
    plt.clf()
    plt.figure()
    plt.xlabel("$ \Delta \\alpha $")
    plt.ylabel("$|U|-|\mathrm{MET}|$")
    delta_alpha=np.linspace(-np.pi,np.pi,201)
    y=np.sqrt(np.multiply(DFName['NN_LongZ'], DFName['NN_LongZ'])+ np.multiply(DFName['NN_PerpZ'], DFName['NN_PerpZ']))-DFName['Boson_Pt']
    delta_pT=np.linspace(y.min(),y.max(),201)
    PF_r, PF_phi = kar2pol(-DFName['NN_LongZ'], DFName['NN_PerpZ'])
    Response = np.divide(-DFName['NN_LongZ'], DFName['Boson_Pt'] )
    x=PF_phi
    HM2=np.empty([200, 200])
    for i in range(0,200):
        for j in range(0,200):
            HM2[i,j]=np.mean(Response[(PF_phi>=delta_alpha[i]) & (PF_phi<delta_alpha[i+1]) & (y>=delta_pT[j]) & (y<delta_pT[j+1])])
    heatmap, xedges, yedges = HM2, delta_alpha, delta_pT
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #plt.clf()
    HM =plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm(), aspect='auto')
    plt.xlim(-np.pi/2,np.pi/2)
    plt.ylim(-200,200)
    plt.colorbar(HM)
    plt.show()
    plt.legend()
    plt.savefig("%sHM_NN_Response_Delta_Alpha_Delta_pT.png"%(plotsD), bbox_inches="tight")
    '''

    plt.clf()
    plt.figure()
    plt.xlabel("$ \Delta \\alpha $")
    plt.ylabel("$|U|-|\mathrm{MET}|$")
    PF_r, PF_phi = kar2pol(-DFName['recoilpatpfPUMET_LongZ'], DFName['recoilpatpfPUMET_PerpZ'])
    DeltapT = DFName['recoilpatpfPUMET_Pt'] - DFName['Boson_Pt']
    heatmap, xedges, yedges = np.histogram2d(  PF_phi, DeltapT,  bins=50,
                                             range=[[-np.pi/2,np.pi/2],
                                                    [-10,10]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #plt.clf()
    HM =plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm(), aspect='auto')
    plt.colorbar(HM)
    plt.legend()
    plt.savefig("%sHM_PF_Delta_Alpha_Delta_pT.png"%(plotsD), bbox_inches="tight")


    plt.clf()
    plt.figure()
    plt.xlabel("$ \Delta \\alpha $")
    plt.ylabel("$|U|-|\mathrm{MET}|$")
    PF_r, PF_phi = kar2pol(-DFName['NN_LongZ'], DFName['NN_PerpZ'])
    NN_pT = np.sqrt(np.multiply(DFName['NN_LongZ'],DFName['NN_LongZ'])+ np.multiply(DFName['NN_PerpZ'],DFName['NN_PerpZ']))
    DeltapT = NN_pT - DFName['Boson_Pt']
    heatmap, xedges, yedges = np.histogram2d(  PF_phi, DeltapT,  bins=50,
                                             range=[[-np.pi/2,np.pi/2],
                                                    [-10,10]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #plt.clf()
    HM =plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm(), aspect='auto')
    plt.colorbar(HM)
    plt.legend()
    plt.savefig("%sHM_NN_Delta_Alpha_Delta_pT.png"%(plotsD), bbox_inches="tight")

    plt.clf()
    plt.figure()
    plt.xlabel("$ \Delta \\alpha $")
    plt.ylabel("Response")
    PF_r, PF_phi = kar2pol(-DFName['recoilpatpfPUMET_LongZ'], DFName['recoilpatpfPUMET_PerpZ'])
    Response = np.divide(-DFName['recoilpatpfPUMET_LongZ'], DFName['Boson_Pt'] )
    heatmap, xedges, yedges = np.histogram2d(  PF_phi, Response,  bins=50,
                                             range=[[-np.pi/2,np.pi/2],
                                                    [0,2]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #plt.clf()
    HM =plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm(), aspect='auto')
    plt.colorbar(HM)
    plt.legend()
    plt.savefig("%sHM_Response_PF_Delta_Alpha_CO.png"%(plotsD), bbox_inches="tight")

    plt.clf()
    plt.figure()
    #plt.suptitle('x-Korrektur Prediction-Target ')
    plt.xlabel("$ \Delta \\alpha $")
    plt.ylabel("Response")
    PF_r, PF_phi = kar2pol(-DFName['recoilpatpfPUMET_LongZ'], DFName['recoilpatpfPUMET_PerpZ'])
    Response = np.divide(-DFName['recoilpatpfPUMET_LongZ'], DFName['Boson_Pt'] )
    heatmap, xedges, yedges = np.histogram2d(  PF_phi, Response,  bins=[25,25],
                                             range=[[-np.pi/2,np.pi/2],
                                                    [0.75,1.25]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #plt.clf()
    HM =plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm(), aspect='auto')
    plt.colorbar(HM)
    plt.legend()
    plt.savefig("%sHM_Response_PF_Delta_Alpha_CR.png"%(plotsD), bbox_inches="tight")

    plt.clf()
    plt.figure()
    #plt.suptitle('x-Korrektur Prediction-Target ')
    plt.xlabel("$ \Delta \\alpha $")
    plt.ylabel("Response")
    PF_r, PF_phi = kar2pol(-DFName['NN_LongZ'], DFName['NN_PerpZ'])
    Response = np.divide(-DFName['NN_LongZ'], DFName['Boson_Pt'] )
    heatmap, xedges, yedges = np.histogram2d(  PF_phi, Response,  bins=[25,25],
                                             range=[[-np.pi/2,np.pi/2],
                                                    [0.75,1.25]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #plt.clf()
    HM =plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm(), aspect='auto')
    plt.colorbar(HM)
    plt.legend()
    plt.savefig("%sHM_Response_NN_Delta_Alpha_CR.png"%(plotsD), bbox_inches="tight")


    plt.clf()
    plt.figure()
    #plt.suptitle('x-Korrektur Prediction-Target ')
    plt.ylabel("$ \Delta \\alpha $")
    plt.xlabel("$p_T$")
    NN_r, NN_phi = kar2pol(-DFName['NN_LongZ'], DFName['NN_PerpZ'])
    heatmap, xedges, yedges = np.histogram2d(  PF_r, PF_phi,  bins=50,
                                             range=[[0,200], [-np.pi/2,np.pi/2]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #plt.clf()
    HM =plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm(), aspect='auto')
    plt.colorbar(HM)
    plt.legend()
    plt.savefig("%sHM_NN_Delta_Alpha_pT.png"%(plotsD), bbox_inches="tight")

    plt.clf()
    plt.figure()
    #plt.suptitle('x-Korrektur Prediction-Target ')
    plt.ylabel("$ \Delta \\alpha $")
    plt.xlabel("$p_T$")
    NN_r, NN_phi = kar2pol(-DFName['recoilpatpfPUMET_LongZ'], DFName['recoilpatpfPUMET_PerpZ'])
    heatmap, xedges, yedges = np.histogram2d(  PF_r, PF_phi,  bins=50,
                                             range=[[0,200], [-np.pi/2,np.pi/2]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #plt.clf()
    HM =plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm(), aspect='auto')
    plt.colorbar(HM)
    plt.legend()
    plt.savefig("%sHM_PF_Delta_Alpha_pT.png"%(plotsD), bbox_inches="tight")


    plt.clf()
    plt.figure()
    #plt.suptitle('x-Korrektur Prediction-Target ')
    plt.ylabel("$ \Delta \\alpha $")
    plt.xlabel("$p_T$")
    PF_r, PF_phi = kar2pol(-DFName['recoilpatpfPUMET_LongZ'], DFName['recoilpatpfPUMET_PerpZ'])
    heatmap, xedges, yedges = np.histogram2d(  PF_r, PF_phi,  bins=50,
                                             range=[[0,30], [-np.pi/2,np.pi/2]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #plt.clf()
    HM =plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm(), aspect='auto')
    plt.colorbar(HM)
    plt.legend()
    plt.savefig("%sHM_PF_Delta_Alpha_pT.png"%(plotsD), bbox_inches="tight")


    plt.clf()
    plt.figure()
    #plt.suptitle('x-Korrektur Prediction-Target ')
    plt.ylabel("$ \Delta \\alpha $")
    plt.xlabel("$p_T$")
    NN_r, NN_phi = kar2pol(-DFName['NN_LongZ'], DFName['NN_PerpZ'])
    heatmap, xedges, yedges = np.histogram2d(  NN_r, NN_phi,  bins=50,
                                             range=[[0,30], [-np.pi/2,np.pi/2]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #plt.clf()
    HM =plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm(), aspect='auto')
    plt.colorbar(HM)
    plt.legend()
    plt.savefig("%sHM_NN_Delta_Alpha_pT.png"%(plotsD), bbox_inches="tight")

    plt.clf()
    plt.figure()
    #plt.suptitle('x-Korrektur Prediction-Target ')
    plt.xlabel("$ p_T^Z $")
    plt.ylabel("Response")
    NN_r, NN_phi = kar2pol(-DFName['NN_LongZ'], DFName['NN_PerpZ'])
    Response = np.divide(-DFName['NN_LongZ'], DFName['Boson_Pt'] )
    heatmap, xedges, yedges = np.histogram2d(  DFName['Boson_Pt'], Response,  bins=[25,25],
                                             range=[[0,10],
                                                    [0,2]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #plt.clf()
    HM =plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm(), aspect='auto')
    plt.colorbar(HM)
    plt.legend()
    plt.savefig("%sHM_Response_NN_Delta_pT_10.png"%(plotsD), bbox_inches="tight")


    plt.clf()
    plt.figure()
    #plt.suptitle('x-Korrektur Prediction-Target ')
    plt.xlabel("$ p_T^Z $")
    plt.ylabel("Response")
    NN_r, NN_phi = kar2pol(-DFName['NN_LongZ'], DFName['NN_PerpZ'])
    Response = np.divide(-DFName['NN_LongZ'], DFName['Boson_Pt'] )
    heatmap, xedges, yedges = np.histogram2d(  DFName['Boson_Pt'], Response,  bins=[25,25],
                                             range=[[0,200],
                                                    [0,2]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #plt.clf()
    HM =plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm(), aspect='auto')
    plt.colorbar(HM)
    plt.legend()
    plt.savefig("%sHM_Response_NN_Delta_pT_0_200.png"%(plotsD), bbox_inches="tight")


    plt.clf()
    plt.figure()
    #plt.suptitle('x-Korrektur Prediction-Target ')
    plt.xlabel("$ \Delta \\alpha $")
    plt.ylabel("Response")
    NN_r, NN_phi = kar2pol(-DFName['NN_LongZ'], DFName['NN_PerpZ'])
    Response = np.divide(-DFName['NN_LongZ'], DFName['Boson_Pt'] )
    heatmap, xedges, yedges = np.histogram2d(  NN_phi, Response,  bins=[25,25],
                                             range=[[0,10],
                                                    [0,2]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #plt.clf()
    HM =plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm(), aspect='auto')
    plt.colorbar(HM)
    plt.legend()
    plt.savefig("%sHM_Response_NN_Delta_pT_10.png"%(plotsD), bbox_inches="tight")

    plt.clf()
    plt.figure()
    #plt.suptitle('x-Korrektur Prediction-Target ')
    plt.xlabel("$ p_T^Z $")
    plt.ylabel("Response")
    PF_r, PF_phi = kar2pol(-DFName['recoilpatpfPUMET_LongZ'], DFName['recoilpatpfPUMET_PerpZ'])
    Response = np.divide(-DFName['recoilpatpfPUMET_LongZ'], DFName['Boson_Pt'] )
    heatmap, xedges, yedges = np.histogram2d(  DFName['Boson_Pt'], Response,  bins=[25,25],
                                             range=[[0,10],
                                                    [0,2]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #plt.clf()
    HM =plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm(), aspect='auto')
    plt.colorbar(HM)
    plt.legend()
    plt.savefig("%sHM_Response_PF_Delta_pT_10.png"%(plotsD), bbox_inches="tight")

    plt.clf()
    plt.figure()
    #plt.suptitle('x-Korrektur Prediction-Target ')
    plt.xlabel("$ p_T^Z $")
    plt.ylabel("Response")
    PF_r, PF_phi = kar2pol(-DFName['recoilpatpfPUMET_LongZ'], DFName['recoilpatpfPUMET_PerpZ'])
    Response = np.divide(-DFName['recoilpatpfPUMET_LongZ'], DFName['Boson_Pt'] )
    heatmap, xedges, yedges = np.histogram2d(  DFName['Boson_Pt'], Response,  bins=[25,25],
                                             range=[[0,200],
                                                    [0,2]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #plt.clf()
    HM =plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm(), aspect='auto')
    plt.colorbar(HM)
    plt.legend()
    plt.savefig("%sHM_Response_PF_Delta_pT_0_200.png"%(plotsD), bbox_inches="tight")

    plt.clf()
    plt.figure()
    #plt.suptitle('x-Korrektur Prediction-Target ')
    plt.xlabel("$ p_T^Z $")
    plt.ylabel("Response")
    NN_r, NN_phi = kar2pol(-DFName['NN_LongZ'], DFName['NN_PerpZ'])
    Response = np.divide(-DFName['NN_LongZ'], DFName['Boson_Pt'] )
    heatmap, xedges, yedges = np.histogram2d(  DFName['Boson_Pt'], Response,  bins=[25,25],
                                             range=[[150,200],
                                                    [0,2]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #plt.clf()
    HM =plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm(), aspect='auto')
    plt.colorbar(HM)
    plt.legend()
    plt.savefig("%sHM_Response_NN_Delta_pT_150.png"%(plotsD), bbox_inches="tight")






    print('PF: Wie viele negative Responses', np.sum(DFName['recoilpatpfPUMET_LongZ']<0))
    print('GBRT: Wie viele negative Responses', np.sum(DFName['LongZCorrectedRecoil_LongZ']<0))
    print('NN: Wie viele negative Responses', np.sum(DFName['NN_LongZ']<0))
    print('BosonPt: Wie viele negative Responses', np.sum(DFName['Boson_Pt']<0))

if __name__ == "__main__":
    inputDir = sys.argv[1]
    filesDir =  sys.argv[2]
    plotDir = sys.argv[3]
    print(plotDir)
    DFName_plain = loadData(inputDir)
    DFName=DFName_plain[DFName_plain['Boson_Pt']<=200]
    DFName=DFName[DFName['Boson_Pt']>0]
    DFName=DFName[DFName['NVertex']<=50]
    DFName=DFName[DFName['NVertex']>=0]
    #DFName['NN_LongZ']=-(DFName['NN_LongZ'])

    DFName_nVertex = DFName
    #DFName_nVertex = DFName_nVertex[DFName_nVertex['Boson_Pt']>0]
    #DFName_nVertex = DFName_nVertex[DFName_nVertex['NVertex']<=50]
    #DFName_nVertex['NN_LongZ']=-(DFName_nVertex['NN_LongZ'])
    getPlotsOutput(inputDir, filesDir, plotDir, DFName, DFName_nVertex)
