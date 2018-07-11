
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
import matplotlib.ticker as mtick
from prepareInput import pol2kar_x, pol2kar_y, kar2pol, pol2kar, angularrange, loadData
import h5py
import sys


fName ="/storage/b/tkopf/mvamet/skim/out.root"
nbins = 10
nbinsVertex = 5
nbinsHist = 40
nbinsHistBin = 40
nbins_relR = 10
colors = cm.brg(np.linspace(0, 1, 3))

colors_InOut = cm.brg(np.linspace(0, 1, 8))
colors2 = cm.brg(np.linspace(0, 1, nbins))
HistLimMin, HistLimMax = -50, 50
#Data settings

def div0( a, b ):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
    return c





def loadData_woutGBRT(filesDir, rootInput, Target_Pt, Target_Phi, NN_mode):
    '''
    tfile = ROOT.TFile(rootInput)
    for key in tfile.GetListOfKeys():
            print('key.GetName()', key.GetName())
            if key.GetName() in  ["MAPAnalyzer/t", "MAPAnalyzer/1"]:
                tree = key.ReadObj()
                arrayName = rnp.tree2array(tree, branches=[Target_Pt, Target_Phi, 'NVertex' ,
                'recoilslimmedMETsPuppi_Pt', 'recoilslimmedMETsPuppi_Phi',
                'recoilslimmedMETs_Pt', 'recoilslimmedMETs_Phi',
                'recoilpatpfNoPUMET_Pt','recoilpatpfNoPUMET_Phi',
                'recoilpatpfPUCorrectedMET_Pt', 'recoilpatpfPUCorrectedMET_Phi',
                'recoilpatpfPUMET_Pt', 'recoilpatpfPUMET_Phi',
                'recoilpatpfTrackMET_Pt', 'recoilpatpfTrackMET_Phi',
                'recoilslimmedMETs_LongZ', 'recoilslimmedMETs_PerpZ',
                'recoilpatpfNoPUMET_LongZ','recoilpatpfNoPUMET_PerpZ',
                'recoilpatpfPUCorrectedMET_LongZ', 'recoilpatpfPUCorrectedMET_PerpZ',
                'recoilpatpfPUMET_LongZ', 'recoilpatpfPUMET_PerpZ',
                'recoilpatpfTrackMET_LongZ', 'recoilpatpfTrackMET_PerpZ',
                'recoilslimmedMETsPuppi_LongZ', 'recoilslimmedMETsPuppi_PerpZ'
                 ],)
                arrayNameNN=rnp.root2array(fName, treename='tree', branches=['NN_LongZ', 'NN_PerpZ', 'NN_Phi', 'NN_Pt'  ],)
            else:
    '''
    NN_MVA = h5py.File("%s/NN_MVA_%s.h5"%(filesDir,NN_mode), "r+")
    print("Keys: %s" % NN_MVA.keys())
    DFNameNN = pd.DataFrame()
    keys = NN_MVA.keys()
    values = [NN_MVA[k] for k in keys]
    for k, v in zip(keys, values):
        DFNameNN[k] = v
                #DFNameNN = NN_MVA.create_dataset([('NN_LongZ', 'f8'), ('NN_PerpZ', 'f8'), ('NN_Pt', 'f8'), ('NN_Phi', 'f8')])



    DFNameInput = loadData(rootInput, Target_Pt, Target_Phi)
    if Target_Pt == 'genMet_Pt':
        DFNameInput['recoilslimmedMETs_LongZ'] = np.cos(DFNameInput['recoilslimmedMETs_Phi']-DFNameInput[Target_Phi])*DFNameInput['recoilslimmedMETs_Pt']
        DFNameInput['recoilslimmedMETs_PerpZ'] = np.sin(DFNameInput['recoilslimmedMETs_Phi']-DFNameInput[Target_Phi])*DFNameInput['recoilslimmedMETs_Pt']
        #DFNameNN = pd.DataFrame.from_records(arrayNameNN.view(np.recarray))
    DFName = pd.concat([DFNameInput, DFNameNN], axis=1, join_axes=[DFNameInput.index])
    return(DFName)


def plotMVAResponseOverpTZ_woutError(branchString, labelName, errbars_shift):
    binwidth = (DFName[Target_Pt].values.max() - DFName[Target_Pt].values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName[Target_Pt], bins=nbins)
    sy, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=(-(DFName[branchString])/DFName[Target_Pt]))
    sy2, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=(DFName[branchString]/DFName[Target_Pt])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAResponseOverNVertex_woutError(branchString, labelName, errbars_shift):
    binwidth = (DFName_nVertex.NVertex.values.max() - DFName_nVertex.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=-(DFName_nVertex[branchString])/DFName_nVertex[Target_Pt])
    sy2, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(DFName_nVertex[branchString]/DFName_nVertex[Target_Pt])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAResponseOverpTZ_wError(branchString, labelName, errbars_shift):
    binwidth = (DFName[Target_Pt].values.max() - DFName[Target_Pt].values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName[Target_Pt], bins=nbins)
    sy, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=-(DFName[branchString]/DFName[Target_Pt]))
    sy2, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=(DFName[branchString]/DFName[Target_Pt])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[:-1]+(_[1:]-_[:-1])/3*errbars_shift), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAResponseOverNVertex_wError(branchString, labelName, errbars_shift):
    binwidth = (DFName_nVertex.NVertex.values.max() - DFName_nVertex.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=-(DFName_nVertex[branchString])/DFName_nVertex[Target_Pt])
    sy2, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(DFName_nVertex[branchString]/DFName_nVertex[Target_Pt])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[:-1]+(_[1:]-_[:-1])/3*errbars_shift), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])


def plotMVAResolutionOverpTZ_woutError_para(branchString, labelName, errbars_shift):
    binwidth = (DFName[Target_Pt].values.max() - DFName[Target_Pt].values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName[Target_Pt], bins=nbins)
    sy, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=-(DFName[branchString]+DFName[Target_Pt]))
    sy2, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=(DFName[branchString]+DFName[Target_Pt])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, std, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAResolutionOverNVertex_woutError_para(branchString, labelName, errbars_shift):
    binwidth = (DFName_nVertex.NVertex.values.max() - DFName_nVertex.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=-(DFName_nVertex[branchString]+DFName_nVertex[Target_Pt]))
    sy2, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(DFName_nVertex[branchString]+DFName_nVertex[Target_Pt])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, std, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def MeanDeviation_Pt(branchString, labelName, errbars_shift):
    binwidth = (DFName[Target_Pt].values.max() - DFName[Target_Pt].values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName[Target_Pt], bins=nbins)
    sy, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=-(DFName[branchString]+DFName[Target_Pt]))
    sy2, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=(DFName[branchString]+DFName[Target_Pt])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])


def MeanDeviation_PV(branchString, labelName, errbars_shift):
    binwidth = (DFName_nVertex.NVertex.values.max() - DFName_nVertex.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=-(DFName_nVertex[branchString]+DFName_nVertex[Target_Pt]))
    sy2, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(DFName_nVertex[branchString]+DFName_nVertex[Target_Pt])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAResolutionOverpTZ_woutError_perp(branchString, labelName, errbars_shift):
    binwidth = (DFName[Target_Pt].values.max() - DFName[Target_Pt].values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName[Target_Pt], bins=nbins)
    sy, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=-(DFName[branchString]))
    sy2, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=(DFName[branchString])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, std, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAResolutionOverpTZ_woutError_perp_RC(branchString, labelName, errbars_shift):
    binwidth = (DFName[Target_Pt].values.max() - DFName[Target_Pt].values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName[Target_Pt], bins=nbins)
    sy, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=(DFName[branchString]))
    sy2, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=(DFName[branchString])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    sy_resp, _resp = np.histogram(DFName[Target_Pt], bins=nbins, weights=-(DFName[branchString]+DFName[Target_Pt]))
    mean_resp = sy_resp / n
    plt.errorbar((_[1:] + _[:-1])/2, div0(std,mean_resp), marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])



def Histogram_Deviation_para_pT(branchString, labelName, errbars_shift):
    Mean = np.mean(-(DFName[branchString])-DFName[Target_Pt].values)
    Std = np.std(-(DFName[branchString])-DFName[Target_Pt].values)
    if branchString in ['NN_LongZ', 'recoilslimmedMETs_LongZ']:
        plt.hist((-(DFName[branchString])-DFName[Target_Pt].values), bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift], linewidth=1.5)
    else:
        plt.hist((-(DFName[branchString])-DFName[Target_Pt].values), bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift])


def Histogram_Deviation_perp_pT(branchString, labelName, errbars_shift):
    Mean = np.mean((DFName[branchString]))
    Std = np.std((DFName[branchString]))
    if branchString in ['NN_PerpZ', 'recoilslimmedMETs_PerpZ']:
        plt.hist(((DFName[branchString])), bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift], linewidth=1.5)
    else:
        plt.hist(((DFName[branchString])), bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift])




def Histogram_Deviation_para_Bin(branchString, labelName, bin):
    Mean = np.mean(-(DFName[branchString])-DFName[Target_Pt].values)
    Std = np.std(-(DFName[branchString])-DFName[Target_Pt].values)
    n, _ = np.histogram(-(DFName[branchString])-DFName[Target_Pt].values, bins=nbinsHistBin)
    plt.hist((-(DFName[branchString])-DFName[Target_Pt].values), bins=nbinsHistBin, range=(_[bin], _[bin+1]), label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors2[bin])

def Histogram_Norm_Comparison(branchStringLong, branchStringPerp, labelName, errbars_shift):
    Norm_ = np.sqrt(np.square(DFName[branchStringLong])+np.square(DFName[branchStringPerp]))
    Mean = np.mean(Norm_-DFName[Target_Pt].values)
    Std = np.std(Norm_-DFName[Target_Pt].values)
    if branchStringLong in ['NN_LongZ', 'recoilslimmedMETs_LongZ']:
        plt.hist(Norm_-DFName[Target_Pt].values, bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift], linewidth=1.5)
    else:
        plt.hist(Norm_-DFName[Target_Pt].values, bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift])

def Histogram_Angle_Dev(branchStringLong, branchStringPerp, labelName, errbars_shift):
    if branchStringLong == 'NN_LongZ':
        r_, phi_ = kar2pol(DFName[branchStringLong], DFName[branchStringPerp])
        phi_ = angularrange(phi_ +DFName[Target_Phi])
        Mean = np.mean(phi_)
        Std = np.std(phi_)
    else:
        r_, phi_ = kar2pol(DFName[branchStringLong], DFName[branchStringPerp])
        phi_ = angularrange(phi_ -np.pi)
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
    Norm_ = DFName[Target_Pt].values
    Mean = np.mean(DFName[Target_Pt].values)
    Std = np.std(Norm_)
    plt.hist(Norm_, bins=nbinsHist, range=[0, 75], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift], linewidth=1.5)


def Histogram_Deviation_para_PV(branchString, labelName, errbars_shift):
    Mean = np.mean(-(DFName_nVertex[branchString])-DFName_nVertex[Target_Pt].values)
    Std = np.std(-(DFName_nVertex[branchString])-DFName_nVertex[Target_Pt].values)
    if branchString in ['NN_LongZ', 'recoilslimmedMETs_LongZ']:
        plt.hist((-(DFName_nVertex[branchString])-DFName_nVertex[Target_Pt].values), bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift], linewidth=1.5)
    else:
        plt.hist((-(DFName_nVertex[branchString])-DFName_nVertex[Target_Pt].values), bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift])


def Histogram_Deviation_perp_PV(branchString, labelName, errbars_shift):
    Mean = np.mean((DFName_nVertex[branchString]))
    Std = np.std((DFName_nVertex[branchString]))
    if branchString in ['NN_PerpZ', 'recoilslimmedMETs_PerpZ']:
        plt.hist(((DFName_nVertex[branchString])), bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift], linewidth=1.5)
    else:
        plt.hist(((DFName_nVertex[branchString])), bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift])

def Hist_LongZ(branchString, labelName, errbars_shift):
    if branchString == Target_Pt:
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
    binwidth = ((-(DFName[branchString])-DFName[Target_Pt].values).max() - (-(DFName[branchString])-DFName[Target_Pt].values).min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram((-(DFName[branchString])-DFName[Target_Pt].values), bins=nbins_relR)
    plt.hist(div0((-(DFName[branchString])-DFName[Target_Pt].values),DFName[Target_Pt].values), bins=nbinsHist, range=[-20, 20], label=labelName, histtype='step', ec=colors[errbars_shift])

def Histogram_relResponse_PV(branchString, labelName, errbars_shift):
    binwidth = ((-(DFName_nVertex[branchString])-DFName_nVertex[Target_Pt].values).max() - (-(DFName_nVertex[branchString])-DFName_nVertex[Target_Pt].values).min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram((-(DFName_nVertex[branchString])-DFName_nVertex[Target_Pt].values), bins=nbins)
    plt.hist(div0((-(DFName_nVertex[branchString])-DFName_nVertex[Target_Pt].values), DFName_nVertex[Target_Pt].values), bins=nbinsHist, range=[-20, 20], label=labelName, histtype='step', ec=colors[errbars_shift])



def NN_Response_pT( labelName, errbars_shift):
    binwidth = (DFName[Target_Pt].values.max() - DFName[Target_Pt].values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName[Target_Pt], bins=nbins)
    sy, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=-(NN_LongZ/DFName[Target_Pt]))
    sy2, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=(NN_LongZ/DFName[Target_Pt])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def NN_Response_PV(branchString, labelName, errbars_shift):
    binwidth = (DFName_nVertex.NVertex.values.max() - DFName_nVertex.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=-(DFName_nVertex[branchString]/DFName_nVertex[Target_Pt]))
    sy2, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(DFName_nVertex[branchString]/DFName_nVertex[Target_Pt])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAResponseOverpTZ_wError(branchString, labelName, errbars_shift):
    binwidth = (DFName[Target_Pt].values.max() - DFName[Target_Pt].values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName[Target_Pt], bins=nbins)
    sy, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=(-(DFName[branchString])/DFName[Target_Pt]))
    sy2, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=(DFName[branchString]/DFName[Target_Pt])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[:-1]+(_[1:]-_[:-1])/3*errbars_shift), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAResponseOverNVertex_wError(branchString, labelName, errbars_shift):
    binwidth = (DFName_nVertex.NVertex.values.max() - DFName_nVertex.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(-(DFName_nVertex[branchString])/np.abs(DFName_nVertex[Target_Pt])))
    sy2, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(DFName_nVertex[branchString]/DFName_nVertex[Target_Pt])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[:-1]+(_[1:]-_[:-1])/3*errbars_shift), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])
    #plt.plot((_[1:] + _[:-1])/2, mean, marker='.', label=labelName, linestyle="None", color=MVAcolors[errbars_shift])
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])


def plotMVAResolutionOverpTZ_woutError_para(branchString, labelName, errbars_shift):
    binwidth = (DFName[Target_Pt].values.max() - DFName[Target_Pt].values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName[Target_Pt], bins=nbins)
    sy, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=-(DFName[branchString]+DFName[Target_Pt]))
    sy2, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=(DFName[branchString]+DFName[Target_Pt])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, std, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAResolutionOverNVertex_woutError_para(branchString, labelName, errbars_shift):
    binwidth = (DFName_nVertex.NVertex.values.max() - DFName_nVertex.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=-(DFName_nVertex[branchString]+DFName_nVertex[Target_Pt]))
    sy2, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(DFName_nVertex[branchString]+DFName_nVertex[Target_Pt])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, std, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAResolutionOverpTZ_woutError_para_RC(branchString, labelName, errbars_shift):
    binwidth = (DFName[Target_Pt].values.max() - DFName[Target_Pt].values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName[Target_Pt], bins=nbins)
    sy, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=-(DFName[branchString]+DFName[Target_Pt]))
    sy2, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=(DFName[branchString]+DFName[Target_Pt])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    sy_resp, _resp = np.histogram(DFName[Target_Pt], bins=nbins, weights=-(DFName[branchString]+DFName[Target_Pt]))
    mean_resp = sy_resp / n
    plt.errorbar((_[1:] + _[:-1])/2, div0(std,mean_resp), marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAResolutionOverNVertex_woutError_para_RC(branchString, labelName, errbars_shift):
    binwidth = (DFName_nVertex.NVertex.values.max() - DFName_nVertex.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=-(DFName_nVertex[branchString]+DFName_nVertex[Target_Pt]))
    sy2, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(DFName_nVertex[branchString]+DFName_nVertex[Target_Pt])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    sy_resp, _resp = np.histogram(DFName_nVertex[Target_Pt], bins=nbinsVertex, weights=-(DFName_nVertex[branchString]+DFName_nVertex[Target_Pt]))
    mean_resp = sy_resp / n
    plt.errorbar((_[1:] + _[:-1])/2, div0(std,mean_resp), marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])


def plotMVAResolutionOverpTZ_woutError_perp(branchString, labelName, errbars_shift):
    binwidth = (DFName[Target_Pt].values.max() - DFName[Target_Pt].values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName[Target_Pt], bins=nbins)
    sy, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=-(DFName[branchString]))
    sy2, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=(DFName[branchString])**2)
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
    binwidth = (DFName[Target_Pt].values.max() - DFName[Target_Pt].values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName[Target_Pt], bins=nbins)
    sy, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=-(DFName[branchString]))
    sy2, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=(DFName[branchString])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    sy_resp, _resp = np.histogram(DFName[Target_Pt], bins=nbins, weights=-(DFName[branchString]+DFName[Target_Pt]))
    mean_resp = sy_resp / n
    plt.errorbar((_[1:] + _[:-1])/2, div0(std,mean_resp), marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAResolutionOverNVertex_woutError_perp_RC(branchString, labelName, errbars_shift):
    binwidth = (DFName_nVertex.NVertex.values.max() - DFName_nVertex.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(DFName_nVertex[branchString]))
    sy2, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(DFName_nVertex[branchString])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    sy_resp, _resp = np.histogram(DFName_nVertex[Target_Pt], bins=nbinsVertex, weights=-(DFName_nVertex[branchString]+DFName_nVertex[Target_Pt]))
    mean_resp = sy_resp / n
    plt.errorbar((_[1:] + _[:-1])/2, div0(std,mean_resp), marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def Histogram_Deviation_perp(branchString, labelName, errbars_shift):
    Mean = np.mean(DFName[branchString])
    Std = np.std(DFName[branchString])
    plt.hist(DFName[branchString], bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors[errbars_shift])

def Mean_Std_Deviation_pTZ_para(branchString, labelName, errbars_shift):
    binwidth = (DFName[Target_Pt].values.max() - DFName[Target_Pt].values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName[Target_Pt], bins=nbins)
    sy, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=(-(DFName[branchString])-DFName[Target_Pt].values))
    sy2, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=(-(DFName[branchString])-DFName[Target_Pt].values)**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])
    plt.errorbar((_[:-1]+(_[1:]-_[:-1])/3*errbars_shift), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])

def Mean_Std_Deviation_pTZ_perp(branchString, labelName, errbars_shift):
    binwidth = (DFName[Target_Pt].values.max() - DFName[Target_Pt].values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName[Target_Pt], bins=nbins)
    sy, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=(DFName[branchString]))
    sy2, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=(DFName[branchString])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])
    plt.errorbar((_[:-1]+(_[1:]-_[:-1])/3*errbars_shift), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])

def Mean_Std_Deviation_PV_para(branchString, labelName, errbars_shift):
    binwidth = (DFName_nVertex.NVertex.values.max() - DFName_nVertex.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(-(DFName_nVertex[branchString])-DFName_nVertex.NVertex.values))
    sy2, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(-(DFName_nVertex[branchString])-DFName_nVertex.NVertex.values)**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])
    plt.errorbar((_[:-1]+(_[1:]-_[:-1])/3*errbars_shift), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])

def Mean_Std_Deviation_PV_perp(branchString, labelName, errbars_shift):
    binwidth = (DFName_nVertex.NVertex.values.max() - DFName_nVertex.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(DFName_nVertex[branchString]))
    sy2, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(DFName_nVertex[branchString])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])
    plt.errorbar((_[:-1]+(_[1:]-_[:-1])/3*errbars_shift), mean, yerr=std, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])




def getPlotsOutput(inputD, filesD, plotsD,DFName, DFName_nVertex, Target_Pt, Target_Phi):



    #Plot settings
    NPlotsLines = 6
    MVA_NPlotsLines = 3
    if Target_Pt=='Boson_Pt':
        LegendTitle = '$\mathrm{Summer\ 17\ campaign}$' '\n'  '$\mathrm{Z \  \\rightarrow \ \mu \mu}$'
    else:
        LegendTitle = '$\mathrm{Summer\ 17\ campaign}$' '\n'  '$\mathrm{Z \  \\rightarrow \ \\tau \\tau  \\rightarrow \ \mu \mu }$'
    pTRangeString_Err = '$0\ \mathrm{GeV} < |\\vec{\mathrm{MET}}| \leq 200\ \mathrm{GeV}$ \n $\mathrm{\# Vertex} \leq 50$'
    pTRangeString= '$0\ \mathrm{GeV} < |\\vec{\mathrm{MET}}| \leq 200\ \mathrm{GeV}$ \n $\mathrm{\# Vertex} \leq 50$'
    pTRangeString_low= '$0\ \mathrm{GeV} < |\\vec{\mathrm{MET}}| \leq %8.2f \ \mathrm{GeV}$ \n $\mathrm{\# Vertex} \leq 50$'%(np.percentile(DFName[Target_Pt],0.3333*100))
    pTRangeString_mid= '$%8.2f\ \mathrm{GeV} < |\\vec{\mathrm{MET}}| \leq %8.2f\ \mathrm{GeV}$ \n $\mathrm{\# Vertex} \leq 50$'%(np.percentile(DFName[Target_Pt],0.3333*100), np.percentile(DFName[Target_Pt],0.6666*100))
    pTRangeString_high= '$%8.2f\ \mathrm{GeV} < |\\vec{\mathrm{MET}}| \leq 200\ \mathrm{GeV}$ \n $\mathrm{\# Vertex} \leq 50$'%(np.percentile(DFName[Target_Pt],0.6666*100))
    pTRangeStringNVertex = pTRangeString
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

    #plotMVAResponseOverpTZ_woutError('LongZCorrectedRecoil_LongZ', 'GBRT', 0)
    plotMVAResponseOverpTZ_woutError('NN_LongZ', 'NN', 2)
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

    #plotMVAResponseOverpTZ_wError('LongZCorrectedRecoil_LongZ', 'GBRT', 0)
    plotMVAResponseOverpTZ_wError('NN_LongZ', 'NN', 2)
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

    #plotMVAResponseOverNVertex_woutError('LongZCorrectedRecoil_LongZ', 'GBRT', 0)
    plotMVAResponseOverNVertex_woutError('NN_LongZ', 'NN', 2)
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

    #plotMVAResponseOverNVertex_wError('LongZCorrectedRecoil_LongZ', 'GBRT', 0)
    plotMVAResponseOverNVertex_wError('NN_LongZ', 'NN', 2)
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

    #plotMVAResolutionOverpTZ_woutError_para('LongZCorrectedRecoil_LongZ', 'GBRT', 0)
    plotMVAResolutionOverpTZ_woutError_para('NN_LongZ', 'NN', 2)
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





    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    #MeanDeviation_Pt('LongZCorrectedRecoil_LongZ', 'GBRT', 0)
    MeanDeviation_Pt('NN_LongZ', 'NN', 2)
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

    #plotMVAResolutionOverNVertex_woutError_para('LongZCorrectedRecoil_LongZ', 'GBRT', 0)
    plotMVAResolutionOverNVertex_woutError_para('NN_LongZ', 'NN', 2)
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

    #MeanDeviation_PV('LongZCorrectedRecoil_LongZ', 'GBRT', 0)
    MeanDeviation_PV('NN_LongZ', 'NN', 2)
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

    #plotMVAResolutionOverpTZ_woutError_perp('LongZCorrectedRecoil_PerpZ', 'GBRT', 0)
    plotMVAResolutionOverpTZ_woutError_perp('NN_PerpZ', 'NN', 2)
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

    #plotMVAResolutionOverNVertex_woutError_perp('LongZCorrectedRecoil_PerpZ', 'GBRT', 0)
    plotMVAResolutionOverNVertex_woutError_perp('NN_PerpZ', 'NN', 2)
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
    #Histogram_Deviation_para_PV('LongZCorrectedRecoil_LongZ', 'GBRT MET', 5)
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


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)


    Histogram_Norm_Comparison('recoilpatpfNoPUMET_LongZ','recoilpatpfNoPUMET_PerpZ', 'No PU MET',0)
    Histogram_Norm_Comparison('recoilpatpfPUCorrectedMET_LongZ', 'recoilpatpfPUCorrectedMET_PerpZ','PU corrected MET',1)
    Histogram_Norm_Comparison( 'recoilpatpfPUMET_LongZ', 'recoilpatpfPUMET_PerpZ', 'PU MET',2)
    Histogram_Norm_Comparison('recoilpatpfTrackMET_LongZ', 'recoilpatpfTrackMET_PerpZ', 'Track MET',3)
    Histogram_Norm_Comparison('recoilslimmedMETsPuppi_LongZ', 'recoilslimmedMETsPuppi_PerpZ', 'Puppi MET',4)
    #Histogram_Norm_Comparison('LongZCorrectedRecoil_LongZ', 'LongZCorrectedRecoil_PerpZ', 'GBRT MET', 5)
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


    Histogram_Norm_Pt(Target_Pt, 'Target MET',7)


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.ylabel('Counts')
    plt.xlabel('$ |\\vec{\mathrm{MET}}|   $ in GeV')
    plt.xlim(0,75)
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_{T}^Z} \\right) $ in GeV')
    #plt.title('Deviation Histogram norm')
    #plt.text('$p_T$ range restriction')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    ##plt.ylim(ylimResMVAMin, ylimResMax)
    plt.savefig("%sHist_norm_pT.png"%(plotsD), bbox_inches="tight")
    plt.close()



    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)


    Histogram_Norm('recoilpatpfNoPUMET_LongZ','recoilpatpfNoPUMET_PerpZ', 'No PU MET',0)
    Histogram_Norm('recoilpatpfPUCorrectedMET_LongZ', 'recoilpatpfPUCorrectedMET_PerpZ','PU corrected MET',1)
    Histogram_Norm( 'recoilpatpfPUMET_LongZ', 'recoilpatpfPUMET_PerpZ', 'PU MET',2)
    Histogram_Norm('recoilpatpfTrackMET_LongZ', 'recoilpatpfTrackMET_PerpZ', 'Track MET',3)
    Histogram_Norm('recoilslimmedMETsPuppi_LongZ', 'recoilslimmedMETsPuppi_PerpZ', 'Puppi MET',4)
    #Histogram_Norm('LongZCorrectedRecoil_LongZ', 'LongZCorrectedRecoil_PerpZ', 'GBRT MET', 5)
    Histogram_Norm('recoilslimmedMETs_LongZ', 'recoilslimmedMETs_PerpZ', 'PF MET', 1)
    Histogram_Norm_Pt(Target_Pt, 'Target MET',7)
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
    #Histogram_Deviation_perp_PV('LongZCorrectedRecoil_PerpZ', 'GBRT', 5)
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
    #Hist_LongZ('LongZCorrectedRecoil_LongZ', 'GBRT MET', 5)
    Hist_LongZ('recoilslimmedMETs_LongZ', 'PF MET', 1)
    Hist_LongZ(Target_Pt, 'Target MET',4)
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
    #Hist_PerpZ('LongZCorrectedRecoil_PerpZ', 'GBRT MET', 5)
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
    #Histogram_Deviation_para_PV('LongZCorrectedRecoil_LongZ', 'GBRT MET', 5)
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
    #Histogram_Angle_Dev('LongZCorrectedRecoil_LongZ', 'LongZCorrectedRecoil_PerpZ', 'GBRT MET', 5)
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

    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    #Mean_Std_Deviation_pTZ_para('LongZCorrectedRecoil_LongZ', 'GBRT', 0)
    Mean_Std_Deviation_pTZ_para('NN_LongZ', 'NN', 2)
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


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    #Mean_Std_Deviation_pTZ_perp('LongZCorrectedRecoil_PerpZ', 'GBRT', 0)
    Mean_Std_Deviation_pTZ_perp('NN_PerpZ', 'NN', 2)
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

    #Mean_Std_Deviation_PV_para('LongZCorrectedRecoil_LongZ', 'GBRT', 0)
    Mean_Std_Deviation_PV_para('NN_LongZ', 'NN', 2)
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

    #Mean_Std_Deviation_PV_perp('LongZCorrectedRecoil_PerpZ', 'GBRT', 0)
    Mean_Std_Deviation_PV_perp('NN_PerpZ', 'NN', 2)
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







if __name__ == "__main__":
    inputDir = sys.argv[1]
    filesDir =  sys.argv[2]
    plotDir = sys.argv[3]
    PhysicsProcess = sys.argv[4]
    rootInput = sys.argv[5]
    NN_mode = sys.argv[6]
    if PhysicsProcess == 'Tau':
        Target_Pt = 'genMet_Pt'
        Target_Phi = 'genMet_Phi'
        DFName_plain = loadData_woutGBRT(filesDir, rootInput, Target_Pt, Target_Phi, NN_mode)
    else:
        Target_Pt = 'Boson_Pt'
        Target_Phi = 'Boson_Phi'
        DFName_plain = loadData(rootInput, Target_Pt, Target_Phi)
    print(plotDir)
    DFName=DFName_plain[DFName_plain[Target_Pt]<=200]
    DFName=DFName[DFName[Target_Pt]>0]
    DFName=DFName[DFName['NVertex']<=50]
    DFName=DFName[DFName['NVertex']>=0]

    DFName_nVertex = DFName

    getPlotsOutput(inputDir, filesDir, plotDir, DFName, DFName_nVertex, Target_Pt, Target_Phi)
