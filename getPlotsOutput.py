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


fName = "/storage/b/tkopf/mvamet/Gridoutput/data1.root"
#Data settings

def loadData(fName):
    treeName = 't'
    arrayName = rnp.root2array(fName, branches=['Boson_Pt', 'Boson_Phi', 'NVertex' ,
        'recoilslimmedMETsPuppi_Pt', 'recoilslimmedMETsPuppi_Phi', 'recoilslimmedMETsPuppi_sumEt',
        'recoilslimmedMETs_Pt', 'recoilslimmedMETs_Phi', 'recoilslimmedMETs_sumEt',
        'recoilpatpfNoPUMET_Pt','recoilpatpfNoPUMET_Phi', 'recoilpatpfNoPUMET_sumEt',
        'recoilpatpfPUCorrectedMET_Pt', 'recoilpatpfPUCorrectedMET_Phi', 'recoilpatpfPUCorrectedMET_sumEt',
        'recoilpatpfPUMET_Pt', 'recoilpatpfPUMET_Phi', 'recoilpatpfPUMET_sumEt',
        'recoilpatpfTrackMET_Pt', 'recoilpatpfTrackMET_Phi', 'recoilpatpfTrackMET_sumEt',
        'NN_LongZ', 'NN_PerpZ' ],)
    DFName = pd.DataFrame.from_records(arrayName.view(np.recarray))
    return(DFName)

DFName = loadData(fName)
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
colors = ['blue','green','red','cyan','magenta','yellow']
MVAcolors =  colors
ylimResMin, ylimResMax = 7.5 , 50
ylimResMVAMin, ylimResMax = 12 , 50
ResponseMin, ResponseMax = 0, 1.05

nbins=12
nbinsVertex = 5



NPlotsLines=3
colors = cm.brg(np.linspace(0, 1, NPlotsLines))


NN_Mode='kart'

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











def getPlotsOutput(inputD, filesD, plotD):
	NN_ = h5py.File("%sNN_MVA.h5" % (inputD), "r")
	NN_LongZ = NN_['NN_LongZ']
	NN_LongZ = NN_['NN_PerpZ']
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
	plt.ylim(0.6, ResponseMax)
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
	plt.ylim(0.6, ResponseMax)
	plt.savefig("%sOutput_Response_pT_wErr.png"%(plotsD))



	fig=plt.figure(figsize=(10,6))
	fig.patch.set_facecolor('white')
	ax = plt.subplot(111)

	plotMVAResponseOverNVertex_woutError('LongZCorrectedRecoil_LongZ', 'GBRT', 0)
	plotMVAResponseOverNVertex_woutError('NN_LongZ', 'NN', 2)
	plotMVAResponseOverNVertex_woutError('recoilslimmedMETs_LongZ', 'PF', 1)
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
	plt.ylim(0.6, ResponseMax)
	plt.savefig("%sOutput_Response_PV.png"%(plotsD))

	fig=plt.figure(figsize=(10,6))
	fig.patch.set_facecolor('white')
	ax = plt.subplot(111)

	plotMVAResponseOverNVertex_wError('LongZCorrectedRecoil_LongZ', 'MVA', 0)
	plotMVAResponseOverNVertex_wError('recoilslimmedMETsPuppi_LongZ', 'Puppi', 2)
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
	plt.ylim(0.6, ResponseMax)
	plt.savefig("%Output_Response_PV_wErr.png"%(plotsD))

	##########Resolutions #########
	######## u para ########
	fig=plt.figure(figsize=(10,6))
	fig.patch.set_facecolor('white')
	ax = plt.subplot(111)

	plotMVAResolutionOverpTZ_woutError_para('LongZCorrectedRecoil_LongZ', 'MVA', 0)
	plotMVAResolutionOverpTZ_woutError_para('recoilslimmedMETsPuppi_LongZ', 'Puppi', 2)
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
	plt.savefig("%Output_Resolution_Long_pT.png"%(plotsD))



	fig=plt.figure(figsize=(10,6))
	fig.patch.set_facecolor('white')
	ax = plt.subplot(111)

	plotMVAResolutionOverNVertex_woutError_para('LongZCorrectedRecoil_LongZ', 'MVA', 0)
	plotMVAResolutionOverNVertex_woutError_para('recoilslimmedMETsPuppi_LongZ', 'Puppi', 2)
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
	plt.savefig("%Output_Resolution_Long_PV.png"%(plotsD))




	#######u perp ######
	fig=plt.figure(figsize=(10,6))
	fig.patch.set_facecolor('white')
	ax = plt.subplot(111)

	plotMVAResolutionOverpTZ_woutError_perp('LongZCorrectedRecoil_PerpZ', 'MVA', 0)
	plotMVAResolutionOverpTZ_woutError_perp('recoilslimmedMETsPuppi_PerpZ', 'Puppi', 2)
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
	plt.savefig("%Output_Resolution_Perp_pT.png"%(plotsD))



	fig=plt.figure(figsize=(10,6))
	fig.patch.set_facecolor('white')
	ax = plt.subplot(111)

	plotMVAResolutionOverNVertex_woutError_perp('LongZCorrectedRecoil_PerpZ', 'MVA', 0)
	plotMVAResolutionOverNVertex_woutError_perp('recoilslimmedMETsPuppi_PerpZ', 'Puppi', 2)
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
	plt.savefig("%Output_Resolution_Perp_PV.png"%(plotsD))






    
if __name__ == "__main__":
	inputDir = sys.argv[1]
	filesDir =  sys.argv[2]
	plotDir = sys.argv[3]
	print(plotDir)
	getPlotsOutput(inputDir, filesDir, plotDir)	   
