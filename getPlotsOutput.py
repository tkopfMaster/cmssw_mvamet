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


fName ="/storage/b/tkopf/mvamet/skim/out.root"
nbins = 10
nbinsVertex = 5
nbinsHist = 250
nbinsHistBin =150
colors = cm.brg(np.linspace(0, 1, 3))
colors2 = cm.brg(np.linspace(0, 1, nbins))
#Data settings

def div0( a, b ):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
    return c

def loadData(inputD):
    #tfile = ROOT.TFile(inputD+'Summer17_NN_Output.root')
    #for key in tfile.GetListOfKeys():
    #    if key.GetName() == "MAPAnalyzer/t":
    #        tree = key.ReadObj()
    #arrayName = rnp.tree2array(tree, branches=['Boson_Pt'
    tree='t'
    arrayName = rnp.root2array(inputD,  branches=['Boson_Pt', 'Boson_Phi', 'NVertex' ,
        'recoilslimmedMETs_LongZ', 'recoilslimmedMETs_PerpZ',
        'recoilslimmedMETsPuppi_LongZ', 'recoilslimmedMETsPuppi_PerpZ',
        'LongZCorrectedRecoil_LongZ', 'LongZCorrectedRecoil_PerpZ',
        'NN_PerpZ', 'NN_LongZ'], )
    '''arrayName = rnp.root2array(inputD, treename='MAPAnalyzer/t', branches=['Boson_Pt', 'Boson_Phi', 'NVertex' ,
        'recoilslimmedMETsPuppi_Pt', 'recoilslimmedMETsPuppi_Phi', 'recoilslimmedMETsPuppi_sumEt',
        'recoilslimmedMETs_Pt', 'recoilslimmedMETs_Phi', 'recoilslimmedMETs_sumEt',
        'recoilpatpfNoPUMET_Pt','recoilpatpfNoPUMET_Phi', 'recoilpatpfNoPUMET_sumEt',
        'recoilpatpfPUCorrectedMET_Pt', 'recoilpatpfPUCorrectedMET_Phi', 'recoilpatpfPUCorrectedMET_sumEt',
        'recoilpatpfPUMET_Pt', 'recoilpatpfPUMET_Phi', 'recoilpatpfPUMET_sumEt',
        'recoilpatpfTrackMET_Pt', 'recoilpatpfTrackMET_Phi', 'recoilpatpfTrackMET_sumEt',
         'NN_PerpZ', 'NN_LongZ'], )'''
    DFName = pd.DataFrame.from_records(arrayName.view(np.recarray))
    return(DFName)

def plotMVAResponseOverpTZ_woutError(branchString, labelName, errbars_shift):
    binwidth = (DFName.Boson_Pt.values.max() - DFName.Boson_Pt.values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName.Boson_Pt, bins=nbins)
    sy, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=np.negative(DFName[branchString]/DFName.Boson_Pt))
    sy2, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(DFName[branchString]/DFName.Boson_Pt)**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    #plt.errorbar((_[:-1]+(_[1:]-_[:-1])/5*errbars_shift), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])
    #plt.plot((_[1:] + _[:-1])/2, mean, marker='.', label=labelName, linestyle="None", color=MVAcolors[errbars_shift])
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAResponseOverNVertex_woutError(branchString, labelName, errbars_shift):
    binwidth = (DFName_nVertex.NVertex.values.max() - DFName_nVertex.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=np.negative(DFName_nVertex[branchString]/DFName_nVertex.Boson_Pt))
    sy2, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(DFName_nVertex[branchString]/DFName_nVertex.Boson_Pt)**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    #plt.errorbar((_[:-1]+(_[1:]-_[:-1])/5*errbars_shift), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])
    #plt.plot((_[1:] + _[:-1])/2, mean, marker='.', label=labelName, linestyle="None", color=MVAcolors[errbars_shift])
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAResponseOverpTZ_wError(branchString, labelName, errbars_shift):
    binwidth = (DFName.Boson_Pt.values.max() - DFName.Boson_Pt.values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName.Boson_Pt, bins=nbins)
    sy, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=np.negative(DFName[branchString]/DFName.Boson_Pt))
    sy2, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(DFName[branchString]/DFName.Boson_Pt)**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[:-1]+(_[1:]-_[:-1])/3*errbars_shift), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])
    #plt.plot((_[1:] + _[:-1])/2, mean, marker='.', label=labelName, linestyle="None", color=MVAcolors[errbars_shift])
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAResponseOverNVertex_wError(branchString, labelName, errbars_shift):
    binwidth = (DFName_nVertex.NVertex.values.max() - DFName_nVertex.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=np.negative(DFName_nVertex[branchString]/DFName_nVertex.Boson_Pt))
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

def MeanDeviation_Pt(branchString, labelName, errbars_shift):
    binwidth = (DFName.Boson_Pt.values.max() - DFName.Boson_Pt.values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName.Boson_Pt, bins=nbins)
    sy, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=-(DFName[branchString]+DFName.Boson_Pt))
    sy2, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(DFName[branchString]+DFName.Boson_Pt)**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    #plt.errorbar((_[:-1]+(_[1:]-_[:-1])/5*errbars_shift), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])
    #plt.plot((_[1:] + _[:-1])/2, std, marker='.', label=labelName, linestyle="None", color=MVAcolors[errbars_shift])
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])


def MeanDeviation_PV(branchString, labelName, errbars_shift):
    binwidth = (DFName_nVertex.NVertex.values.max() - DFName_nVertex.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=-(DFName_nVertex[branchString]+DFName_nVertex.Boson_Pt))
    sy2, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(DFName_nVertex[branchString]+DFName_nVertex.Boson_Pt)**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    #plt.errorbar((_[:-1]+(_[1:]-_[:-1])/5*errbars_shift), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])
    #plt.plot((_[1:] + _[:-1])/2, std, marker='.', label=labelName, linestyle="None", color=MVAcolors[errbars_shift])
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

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

def plotMVAResolutionOverpTZ_woutError_perp_RC(branchString, labelName, errbars_shift):
    binwidth = (DFName.Boson_Pt.values.max() - DFName.Boson_Pt.values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName.Boson_Pt, bins=nbins)
    sy, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(DFName[branchString]))
    sy2, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(DFName[branchString])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    sy_resp, _resp = np.histogram(DFName.Boson_Pt, bins=nbins, weights=-(DFName[branchString]+DFName.Boson_Pt))
    mean_resp = sy_resp / n
    #plt.errorbar((_[:-1]+(_[1:]-_[:-1])/5*errbars_shift), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])
    #plt.plot((_[1:] + _[:-1])/2, std, marker='.', label=labelName, linestyle="None", color=MVAcolors[errbars_shift])
    plt.errorbar((_[1:] + _[:-1])/2, div0(std,mean_resp), marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])



def Histogram_Response(branchString, labelName, errbars_shift):
    binwidth = ((-(DFName[branchString])-DFName.Boson_Pt.values).max() - (-(DFName[branchString])-DFName.Boson_Pt.values).min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram((-(DFName[branchString])-DFName.Boson_Pt.values), bins=nbins)
    #sy, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=-(DFName[branchString]))
    #sy2, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(DFName[branchString])**2)
    #mean = sy / n
    #std = np.sqrt(sy2/n - mean*mean)
    #plt.errorbar((_[:-1]+(_[1:]-_[:-1])/5*errbars_shift), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])
    #plt.plot((_[1:] + _[:-1])/2, std, marker='.', label=labelName, linestyle="None", color=MVAcolors[errbars_shift])
    plt.hist((-(DFName[branchString])-DFName.Boson_Pt.values), bins=nbinsHist, range=[-200, 200], label=labelName, histtype='step', ec=colors[errbars_shift])

def Histogram_Response_Bin(branchString, labelName, bin):
    binwidth = ((-(DFName[branchString])-DFName.Boson_Pt.values).max() - (-(DFName[branchString])-DFName.Boson_Pt.values).min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram((-(DFName[branchString])-DFName.Boson_Pt.values), bins=nbins)
    #sy, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=-(DFName[branchString]))
    #sy2, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(DFName[branchString])**2)
    #mean = sy / n
    #std = np.sqrt(sy2/n - mean*mean)
    #plt.errorbar((_[:-1]+(_[1:]-_[:-1])/5*errbars_shift), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])
    #plt.plot((_[1:] + _[:-1])/2, std, marker='.', label=labelName, linestyle="None", color=MVAcolors[errbars_shift])
    plt.hist((-(DFName[branchString])-DFName.Boson_Pt.values), bins=nbinsHistBin, range=(_[bin], _[bin+1]], label=labelName, histtype='step', ec=colors2[bin])


def Histogram_Response_PV(branchString, labelName, errbars_shift):
    binwidth = ((-(DFName_nVertex[branchString])-DFName_nVertex.NVertex.values).max() - (-(DFName_nVertex[branchString])-DFName_nVertex.NVertex.values).min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram((-(DFName_nVertex[branchString])-DFName_nVertex.NVertex.values), bins=nbins)
    #sy, _ = np.histogram(DFName_nVertex.NVertex, bins=nbins, weights=-(DFName_nVertex[branchString]))
    #sy2, _ = np.histogram(DFName_nVertex.NVertex, bins=nbins, weights=(DFName_nVertex[branchString])**2)
    #mean = sy / n
    #std = np.sqrt(sy2/n - mean*mean)
    #plt.errorbar((_[:-1]+(_[1:]-_[:-1])/5*errbars_shift), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])
    #plt.plot((_[1:] + _[:-1])/2, std, marker='.', label=labelName, linestyle="None", color=MVAcolors[errbars_shift])
    plt.hist((-(DFName_nVertex[branchString])-DFName_nVertex.Boson_Pt.values), bins=nbinsHist, range=[-200, 200], label=labelName, histtype='step', ec=colors[errbars_shift])

def Histogram_relResponse(branchString, labelName, errbars_shift):
    binwidth = ((-(DFName[branchString])-DFName.Boson_Pt.values).max() - (-(DFName[branchString])-DFName.Boson_Pt.values).min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram((-(DFName[branchString])-DFName.Boson_Pt.values), bins=nbins_relR)
    #sy, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=-(DFName[branchString]))
    #sy2, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(DFName[branchString])**2)
    #mean = sy / n
    #std = np.sqrt(sy2/n - mean*mean)
    #plt.errorbar((_[:-1]+(_[1:]-_[:-1])/5*errbars_shift), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])
    #plt.plot((_[1:] + _[:-1])/2, std, marker='.', label=labelName, linestyle="None", color=MVAcolors[errbars_shift])
    plt.hist(div0((-(DFName[branchString])-DFName.Boson_Pt.values),DFName.Boson_Pt.values), bins=nbinsHist, range=[-200, 200], label=labelName, histtype='step', ec=colors[errbars_shift])

def Histogram_relResponse_PV(branchString, labelName, errbars_shift):
    binwidth = ((-(DFName_nVertex[branchString])-DFName_nVertex.Boson_Pt.values).max() - (-(DFName_nVertex[branchString])-DFName_nVertex.Boson_Pt.values).min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram((-(DFName_nVertex[branchString])-DFName_nVertex.Boson_Pt.values), bins=nbins)
    #sy, _ = np.histogram(DFName_nVertex.NVertex, bins=nbins, weights=-(DFName_nVertex[branchString]))
    #sy2, _ = np.histogram(DFName_nVertex.NVertex, bins=nbins, weights=(DFName_nVertex[branchString])**2)
    #mean = sy / n
    #std = np.sqrt(sy2/n - mean*mean)
    #plt.errorbar((_[:-1]+(_[1:]-_[:-1])/5*errbars_shift), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])
    #plt.plot((_[1:] + _[:-1])/2, std, marker='.', label=labelName, linestyle="None", color=MVAcolors[errbars_shift])
    plt.hist(div0((-(DFName_nVertex[branchString])-DFName_nVertex.Boson_Pt.values), DFName_nVertex.Boson_Pt.values), bins=nbinsHist, range=[-200, 200], label=labelName, histtype='step', ec=colors[errbars_shift])



def NN_Response_pT( labelName, errbars_shift):
    binwidth = (DFName.Boson_Pt.values.max() - DFName.Boson_Pt.values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName.Boson_Pt, bins=nbins)
    sy, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=-(NN_LongZ/DFName.Boson_Pt))
    sy2, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(NN_LongZ/DFName.Boson_Pt)**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    #plt.errorbar((_[:-1]+(_[1:]-_[:-1])/5*errbars_shift), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])
    #plt.plot((_[1:] + _[:-1])/2, mean, marker='.', label=labelName, linestyle="None", color=MVAcolors[errbars_shift])
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def NN_Response_PV(branchString, labelName, errbars_shift):
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
    sy, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=np.abs(DFName[branchString]/DFName.Boson_Pt))
    sy2, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(DFName[branchString]/DFName.Boson_Pt)**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[:-1]+(_[1:]-_[:-1])/3*errbars_shift), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])
    #plt.plot((_[1:] + _[:-1])/2, mean, marker='.', label=labelName, linestyle="None", color=MVAcolors[errbars_shift])
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAResponseOverNVertex_wError(branchString, labelName, errbars_shift):
    binwidth = (DFName_nVertex.NVertex.values.max() - DFName_nVertex.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=-(np.abs(DFName_nVertex[branchString])/np.abs(DFName_nVertex.Boson_Pt)))
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

def plotMVAResolutionOverpTZ_woutError_para_RC(branchString, labelName, errbars_shift):
    binwidth = (DFName.Boson_Pt.values.max() - DFName.Boson_Pt.values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName.Boson_Pt, bins=nbins)
    sy, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=-(DFName[branchString]+DFName.Boson_Pt))
    sy2, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(DFName[branchString]+DFName.Boson_Pt)**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    sy_resp, _resp = np.histogram(DFName.Boson_Pt, bins=nbins, weights=-(DFName[branchString]+DFName.Boson_Pt))
    mean_resp = sy_resp / n
    #plt.errorbar((_[:-1]+(_[1:]-_[:-1])/5*errbars_shift), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])
    #plt.plot((_[1:] + _[:-1])/2, std, marker='.', label=labelName, linestyle="None", color=MVAcolors[errbars_shift])
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
    #plt.errorbar((_[:-1]+(_[1:]-_[:-1])/5*errbars_shift), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])
    #plt.plot((_[1:] + _[:-1])/2, std, marker='.', label=labelName, linestyle="None", color=MVAcolors[errbars_shift])
    plt.errorbar((_[1:] + _[:-1])/2, div0(std,mean_resp), marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])


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

def plotMVAResolutionOverpTZ_woutError_perp_RC(branchString, labelName, errbars_shift):
    binwidth = (DFName.Boson_Pt.values.max() - DFName.Boson_Pt.values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName.Boson_Pt, bins=nbins)
    sy, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=-(DFName[branchString]))
    sy2, _ = np.histogram(DFName.Boson_Pt, bins=nbins, weights=(DFName[branchString])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    sy_resp, _resp = np.histogram(DFName.Boson_Pt, bins=nbins, weights=-(DFName[branchString]+DFName.Boson_Pt))
    mean_resp = sy_resp / n
    #plt.errorbar((_[:-1]+(_[1:]-_[:-1])/5*errbars_shift), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])
    #plt.plot((_[1:] + _[:-1])/2, std, marker='.', label=labelName, linestyle="None", color=MVAcolors[errbars_shift])
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
    #plt.errorbar((_[:-1]+(_[1:]-_[:-1])/5*errbars_shift), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])
    #plt.plot((_[1:] + _[:-1])/2, std, marker='.', label=labelName, linestyle="None", color=MVAcolors[errbars_shift])
    plt.errorbar((_[1:] + _[:-1])/2, div0(std,mean_resp), marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])









def getPlotsOutput(inputD, filesD, plotsD,DFName, DFName_nVertex):

    DFName = loadData(inputD)
    DFName=DFName[DFName['Boson_Pt']<=200]
    DFName=DFName[DFName['Boson_Pt']>0]

    DFName_nVertex = DFName[DFName['Boson_Pt']<=200]
    DFName_nVertex = DFName[DFName['Boson_Pt']>60]
    DFName_nVertex = DFName[DFName['NVertex']<=50]

    #Plot settings
    NPlotsLines = 6
    MVA_NPlotsLines = 3
    pTRangeString = '$0\ \mathrm{GeV} < p_{T}^Z \leq 200\ \mathrm{GeV}$'
    pTRangeStringNVertex = '$60\ \mathrm{GeV} < p_{T}^Z \leq 200\ \mathrm{GeV}$ \n $\mathrm{\# Vertex} \leq 50$'
    LegendTitle = '$\mathrm{Summer\ 17\ campaign}$' '\n'  '$\mathrm{Z \  \\rightarrow \ \mu \mu}$'
    colors = ['blue','green','red','cyan','magenta','yellow']
    MVAcolors =  colors
    ylimResMin, ylimResMax = 7.5 , 50
    ylimResMVAMin, ylimResMax = 5 , 35
    ylimResMVAMin_RC, ylimResMax_RC = 0 , 50
    ResponseMin, ResponseMax = 0.0, 2.0
    ResponseMinErr, ResponseMaxErr = -0.5, 2.5




    NPlotsLines=3
    colors = cm.brg(np.linspace(0, 1, NPlotsLines))


    #NN_mode='kart'

    ################################MVA Output ################################
    nbinsVertex = 10
    #########Response u_para/pTZ ###########
    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    plotMVAResponseOverpTZ_woutError('LongZCorrectedRecoil_LongZ', 'GBRT', 0)
    plotMVAResponseOverpTZ_woutError('NN_LongZ', 'NN', 2)
    plotMVAResponseOverpTZ_woutError('recoilslimmedMETs_LongZ', 'PF', 1)
    plt.plot([0, 200], [1, 1], color='k', linestyle='--', linewidth=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.xlabel('$p_{T}^Z $ in GeV')
    plt.ylabel('$\\langle \\frac{U_{\parallel}}{p_{T}^Z} \\rangle$ ')
    plt.title('Response $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    plt.ylim(ResponseMin, ResponseMax)
    plt.savefig("%sOutput_Response_pT.png"%(plotsD))

    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    plotMVAResponseOverpTZ_wError('LongZCorrectedRecoil_LongZ', 'GBRT', 0)
    plotMVAResponseOverpTZ_wError('NN_LongZ', 'NN', 2)
    plotMVAResponseOverpTZ_wError('recoilslimmedMETs_LongZ', 'PF', 1)
    plt.plot([0, 200], [1, 1], color='k', linestyle='--', linewidth=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.xlabel('$p_{T}^Z $ in GeV')
    plt.ylabel('$\\langle \\frac{U_{\parallel}}{p_{T}^Z} \\rangle$ ')
    plt.title('Response $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    plt.ylim(ResponseMinErr, ResponseMaxErr)
    plt.savefig("%sOutput_Response_pT_wErr.png"%(plotsD))



    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    plotMVAResponseOverNVertex_woutError('LongZCorrectedRecoil_LongZ', 'GBRT', 0)
    plotMVAResponseOverNVertex_woutError('NN_LongZ', 'NN', 2)
    plotMVAResponseOverNVertex_woutError('recoilslimmedMETs_LongZ', 'PF', 1)
    plt.plot([0, 50], [1, 1], color='k', linestyle='--', linewidth=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    plt.xlabel('#$ \mathrm{PV}$ ')
    plt.ylabel('$\\langle \\frac{U_{\parallel}}{p_{T}^Z} \\rangle$ ')
    plt.title('Response $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    plt.ylim(0, ResponseMax)
    plt.savefig("%sOutput_Response_PV.png"%(plotsD))

    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    plotMVAResponseOverNVertex_wError('LongZCorrectedRecoil_LongZ', 'GBRT', 0)
    plotMVAResponseOverNVertex_wError('NN_LongZ', 'NN', 2)
    plotMVAResponseOverNVertex_wError('recoilslimmedMETs_LongZ', 'PF', 1)
    plt.plot([0, 80], [1, 1], color='k', linestyle='--', linewidth=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    plt.xlabel('#$ \mathrm{PV}$ ')
    plt.ylabel('$\\langle \\frac{U_{\parallel}}{p_{T}^Z} \\rangle$ ')
    plt.title('Response $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    plt.ylim(ResponseMinErr, ResponseMaxErr)
    plt.savefig("%sOutput_Response_PV_wErr.png"%(plotsD))

    ##########Resolutions #########
    ######## u para ########
    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    plotMVAResolutionOverpTZ_woutError_para('LongZCorrectedRecoil_LongZ', 'GBRT', 0)
    plotMVAResolutionOverpTZ_woutError_para('NN_LongZ', 'NN', 2)
    plotMVAResolutionOverpTZ_woutError_para('recoilslimmedMETs_LongZ', 'PF', 1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.xlabel('$p_{T}^Z $ in GeV')
    plt.ylabel('$\sigma \\left( U_{\parallel}- p_{T}^Z \\right) $ in GeV')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\parallel}}{p_{T}^Z} \\right) $ in GeV')
    plt.title('Resolution $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    plt.ylim(ylimResMVAMin, ylimResMax)
    plt.savefig("%sOutput_Resolution_Long_pT.png"%(plotsD))

    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    plotMVAResolutionOverpTZ_woutError_para_RC('LongZCorrectedRecoil_LongZ', 'GBRT', 0)
    plotMVAResolutionOverpTZ_woutError_para_RC('NN_LongZ', 'NN', 2)
    plotMVAResolutionOverpTZ_woutError_para_RC('recoilslimmedMETs_LongZ', 'PF', 1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.xlabel('$p_{T}^Z $ in GeV')
    plt.ylabel('$\\frac{\sigma \\left( U_{\parallel}- p_{T}^Z \\right)}{\\frac{U_{\parallel}}{p_{T}^Z}} $ ')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\parallel}}{p_{T}^Z} \\right) $ in GeV')
    plt.title('Resolution $U_{\parallel}$, Response corrected')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(ylimResMVAMin_RC, ylimResMax_RC)
    plt.savefig("%sOutput_Resolution_Long_pT_RC.png"%(plotsD))

    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    MeanDeviation_Pt('LongZCorrectedRecoil_LongZ', 'GBRT', 0)
    MeanDeviation_Pt('NN_LongZ', 'NN', 2)
    MeanDeviation_Pt('recoilslimmedMETs_LongZ', 'PF', 1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.xlabel('$p_{T}^Z $ in GeV')
    plt.ylabel('$ \\langle U_{\parallel}- p_{T}^Z \\rangle $ in GeV')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\parallel}}{p_{T}^Z} \\right) $ in GeV')
    plt.title('Mean Deviation $U_{\parallel}-p_T^Z$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    plt.ylim(-20, 10)
    plt.savefig("%sMeanDeviation_pT.png"%(plotsD))


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    plotMVAResolutionOverNVertex_woutError_para('LongZCorrectedRecoil_LongZ', 'GBRT', 0)
    plotMVAResolutionOverNVertex_woutError_para('NN_LongZ', 'NN', 2)
    plotMVAResolutionOverNVertex_woutError_para('recoilslimmedMETs_LongZ', 'PF', 1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    plt.xlabel('#$ \mathrm{PV}$ ')
    plt.ylabel('$\sigma \\left( U_{\parallel}- p_{T}^Z \\right) $ in GeV')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\parallel}}{p_{T}^Z} \\right) $ in GeV')
    plt.title('Resolution $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    plt.ylim(ylimResMVAMin, ylimResMax)
    plt.savefig("%sOutput_Resolution_Long_PV.png"%(plotsD))

    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    plotMVAResolutionOverNVertex_woutError_para_RC('LongZCorrectedRecoil_LongZ', 'GBRT', 0)
    plotMVAResolutionOverNVertex_woutError_para_RC('NN_LongZ', 'NN', 2)
    plotMVAResolutionOverNVertex_woutError_para_RC('recoilslimmedMETs_LongZ', 'PF', 1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    plt.xlabel('#$ \mathrm{PV}$ ')
    plt.ylabel('$\\frac{\sigma \\left( U_{\parallel}- p_{T}^Z \\right)}{\\frac{U_{\parallel}}{p_{T}^Z}} $ ')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\parallel}}{p_{T}^Z} \\right) $ in GeV')
    plt.title('Resolution $U_{\parallel}$, Response corrected')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(ylimResMVAMin_RC, ylimResMax_RC)
    plt.savefig("%sOutput_Resolution_Long_PV_RC.png"%(plotsD))

    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    MeanDeviation_PV('LongZCorrectedRecoil_LongZ', 'GBRT', 0)
    MeanDeviation_PV('NN_LongZ', 'NN', 2)
    MeanDeviation_PV('recoilslimmedMETs_LongZ', 'PF', 1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    plt.xlabel('#$ \mathrm{PV}$ ')
    plt.ylabel('$ \\langle U_{\parallel}- p_{T}^Z \\rangle $ in GeV')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\parallel}}{p_{T}^Z} \\right) $ in GeV')
    plt.title('Mean Deviation $U_{\parallel}-p_T^Z$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    plt.ylim(-20, 20)
    plt.savefig("%sMeanDeviation_PV.png"%(plotsD))


    #######u perp ######
    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    plotMVAResolutionOverpTZ_woutError_perp('LongZCorrectedRecoil_PerpZ', 'GBRT', 0)
    plotMVAResolutionOverpTZ_woutError_perp('NN_PerpZ', 'NN', 2)
    plotMVAResolutionOverpTZ_woutError_perp('recoilslimmedMETs_PerpZ', 'PF', 1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.xlabel('$p_{T}^Z $ in GeV')
    plt.ylabel('$\sigma \\left( U_{\perp} \\right) $ in GeV')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_{T}^Z} \\right) $ in GeV')
    plt.title('Resolution $U_{\perp}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    plt.ylim(ylimResMVAMin, ylimResMax)
    plt.savefig("%sOutput_Resolution_Perp_pT.png"%(plotsD))

    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    plotMVAResolutionOverpTZ_woutError_perp_RC('LongZCorrectedRecoil_PerpZ', 'GBRT', 0)
    plotMVAResolutionOverpTZ_woutError_perp_RC('NN_PerpZ', 'NN', 2)
    plotMVAResolutionOverpTZ_woutError_perp_RC('recoilslimmedMETs_PerpZ', 'PF', 1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.xlabel('$p_{T}^Z $ in GeV')
    plt.ylabel('$\\frac{\sigma \\left( U_{\perp} \\right) }{\\frac{U_{\parallel}}{p_T^Z}} $ ')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_{T}^Z} \\right) $ in GeV')
    plt.title('Resolution $U_{\perp}$, Response corrected')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(ylimResMVAMin_RC, ylimResMax_RC)
    plt.savefig("%sOutput_Resolution_Perp_pT_RC.png"%(plotsD))



    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    plotMVAResolutionOverNVertex_woutError_perp('LongZCorrectedRecoil_PerpZ', 'GBRT', 0)
    plotMVAResolutionOverNVertex_woutError_perp('NN_PerpZ', 'NN', 2)
    plotMVAResolutionOverNVertex_woutError_perp('recoilslimmedMETs_PerpZ', 'PF', 1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    plt.xlabel('#$ \mathrm{PV}$ ')
    plt.ylabel('$\sigma \\left( U_{\perp} \\right) $ in GeV')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_{T}^Z} \\right) $ in GeV')
    plt.title('Resolution $U_{\perp}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    plt.ylim(ylimResMVAMin, ylimResMax)
    plt.savefig("%sOutput_Resolution_Perp_PV.png"%(plotsD))

    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    plotMVAResolutionOverNVertex_woutError_perp_RC('LongZCorrectedRecoil_PerpZ', 'GBRT', 0)
    plotMVAResolutionOverNVertex_woutError_perp_RC('NN_PerpZ', 'NN', 2)
    plotMVAResolutionOverNVertex_woutError_perp_RC('recoilslimmedMETs_PerpZ', 'PF', 1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    plt.xlabel('#$ \mathrm{PV}$ ')
    plt.ylabel('$\\frac{\sigma \\left( U_{\perp} \\right)}{\\frac{U_{\parallel}}{p_T^Z}} $ ')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_{T}^Z} \\right) $ in GeV')
    plt.title('Resolution $U_{\perp}$, Response corrected')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(ylimResMVAMin_RC, ylimResMax_RC)
    plt.savefig("%sOutput_Resolution_Perp_PV_RC.png"%(plotsD))


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    Histogram_Response('LongZCorrectedRecoil_LongZ', 'GBRT', 0)
    Histogram_Response('NN_LongZ', 'NN', 2)
    Histogram_Response('recoilslimmedMETs_LongZ', 'PF', 1)

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


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)
    for Bin in range(0,nbins):
            Histogram_Response_Bin('NN_LongZ', 'NN', Bin)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.ylabel('Counts')
    plt.xlabel('$ U_{\parallel} - p_T^Z  $ in GeV')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_{T}^Z} \\right) $ in GeV')
    plt.title('Response Histogram by Bin')
    #plt.text('$p_T$ range restriction')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(ylimResMVAMin, ylimResMax)
    plt.savefig("%sOutput_Response_Hist_Bin.png"%(plotsD))


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    Histogram_Response_PV('LongZCorrectedRecoil_LongZ', 'GBRT', 0)
    Histogram_Response_PV('NN_LongZ', 'NN', 2)
    Histogram_Response_PV('recoilslimmedMETs_LongZ', 'PF', 1)


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    plt.ylabel('Counts')
    plt.xlabel('$ U_{\parallel} - p_T^Z  $ in GeV')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_{T}^Z} \\right) $ in GeV')
    plt.title('Response Histogram')
    #plt.text('$p_T$ and $\# PV$ range restriction')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(ylimResMVAMin, ylimResMax)
    plt.savefig("%sOutput_Response_Hist_PV.png"%(plotsD))



    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    Histogram_relResponse('LongZCorrectedRecoil_LongZ', 'GBRT', 0)
    Histogram_relResponse('NN_LongZ', 'NN', 2)
    Histogram_relResponse('recoilslimmedMETs_LongZ', 'PF', 1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.ylabel('Counts')
    plt.xlabel('$ \\frac{U_{\parallel} - p_T^Z}{p_T^Z}  $')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_{T}^Z} \\right) $ in GeV')
    plt.title('relative Response Histogram')
    #plt.text('$p_T$ range restriction')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.xlim(-50, 50)
    plt.savefig("%sOutput_relResponse_Hist.png"%(plotsD))


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    Histogram_relResponse_PV('LongZCorrectedRecoil_LongZ', 'GBRT', 0)
    Histogram_relResponse_PV('NN_LongZ', 'NN', 2)
    Histogram_relResponse_PV('recoilslimmedMETs_LongZ', 'PF', 1)


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    plt.ylabel('Counts')
    plt.xlabel('$ \\frac{U_{\parallel} - p_T^Z}{p_T^Z}  $ in GeV')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_{T}^Z} \\right) $ in GeV')
    plt.title('relative Response Histogram')
    #plt.text('$p_T$ and $\# PV$ range restriction')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(-50, 50)
    plt.savefig("%sOutput_relResponse_Hist_PV.png"%(plotsD))



'''
    plt.clf()
    plt.figure()
    plt.suptitle('Histogram deviation over $p_T^Z$ ')
    plt.xlabel("$p_T^Z \mathrm{ in GeV}$")
    plt.ylabel('$ U_{\parallel} - p_T^Z  $ in GeV')
    heatmap, xedges, yedges = np.histogram2d(
        map(np.subtract, np.negative(DFName['NN_LongZ']),  DFName.Boson_Pt) ,
        DFName.Boson_Pt, bins=50,
        range=[[-5,200],
               [ -10, 10]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    #plt.colorbar(heatmap.T)
    plt.legend()
    plt.savefig("%sOutput_Response_HM_pTZ.png"%(plotsD))

    plt.clf()
    plt.figure()
    plt.suptitle('Histogram deviation over $p_T^Z$ ')
    plt.xlabel("$p_T^Z \mathrm{ in GeV}$")
    plt.ylabel('$ U_{\parallel} - p_T^Z  $ in GeV')
    heatmap, xedges, yedges = np.histogram2d(
        map(np.subtract, np.negative(['NN_LongZ']),  DFName.Boson_Pt) ,
        DFName.Boson_Pt, bins=50,
        range=[[-5,200],
               [ -5, 5]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    #plt.colorbar(heatmap.T)
    plt.legend()
    plt.savefig("%sOutput_Response_HM_pTZ_CR.png"%(plotsD))

    plt.clf()
    plt.figure()
    plt.suptitle('Histogram deviation over $\# \mathrm{PV}$ ')
    plt.xlabel("$\# \mathrm{PV}$")
    plt.ylabel('$ U_{\parallel} - p_T^Z  $ in GeV')
    heatmap, xedges, yedges = np.histogram2d(
        map(np.subtract, np.negative(DFName_nVertex['NN_LongZ']),  DFName_nVertex.NVertex) ,
        DFName_nVertex.NVertex, bins=50,
        range=[[0,50],
               [ -10, 10]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    #plt.colorbar(heatmap.T)
    plt.legend()
    plt.savefig("%sOutput_Response_HM_PV.png"%(plotsD))
    '''




if __name__ == "__main__":
    inputDir = sys.argv[1]
    filesDir =  sys.argv[2]
    plotDir = sys.argv[3]
    print(plotDir)
    DFName = loadData(inputDir)
    DFName=DFName[DFName['Boson_Pt']<=200]
    DFName=DFName[DFName['Boson_Pt']>0]
    #DFName['NN_LongZ']=-DFName['NN_LongZ']

    DFName_nVertex = DFName[DFName['Boson_Pt']<=200]
    DFName_nVertex = DFName[DFName['Boson_Pt']>60]
    DFName_nVertex = DFName[DFName['NVertex']<=50]
    #DFName_nVertex['NN_LongZ']=-DFName_nVertex['NN_LongZ']
    getPlotsOutput(inputDir, filesDir, plotDir, DFName, DFName_nVertex)
