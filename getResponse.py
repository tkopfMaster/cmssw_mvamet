
import numpy as np
from prepareInput import pol2kar_x, pol2kar_y, kar2pol, pol2kar, angularrange
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
from matplotlib.ticker import NullFormatter, MaxNLocator
from getPlotsOutputclean import loadData, loadData_woutGBRT
import h5py
import sys


fName ="/storage/b/tkopf/mvamet/skim/out.root"
nbins = 7
binsAngle = 7
nbinsVertex = 5
nbinsHist = 40
nbinsHistBin = 40
nbins_relR = 10
colors = cm.brg(np.linspace(0, 1, 8))

colors_InOut = cm.brg(np.linspace(0, 1, 8))
colors2 = cm.brg(np.linspace(0, 1, nbins))
HistLimMin, HistLimMax = -50, 50
ResponseMin, ResponseMax = -1,3
ResolutionParaMin, ResolutionParaMax = -40, 40
ResolutionPerpMin, ResolutionPerpMax = -40, 40

errbars_shift2 = 10

pTTresh = 10

#Data settings

def getResponse(METlong):
    Response = -DFName[METlong]/DFName[Target_Pt]
    Response = Response[~np.isnan(Response)]
    return Response
def getResponse_pTRange( METlong, rangemin, rangemax):
    Response = -DFName[METlong][(DFName[Target_Pt]>rangemin) & (DFName[Target_Pt]<=rangemax) ]/DFName[Target_Pt][(DFName[Target_Pt]>rangemin) & (DFName[Target_Pt]<=rangemax) ]
    PhiStr = METlong.replace('LongZ','Phi')
    array=getAngle(PhiStr)
    return array[(DFName[Target_Pt]>rangemin) & (DFName[Target_Pt]<=rangemax) ], Response


def getResponseIdx(METlong):
    Response = -DFName[METlong]/DFName[Target_Pt]
    Response = Response[~np.isnan(Response)]
    return ~np.isnan(Response)

def getAngle(METPhi):
    if METPhi=='NN_Phi':
        NN_r, deltaPhi = kar2pol(-DFName['NN_LongZ'], DFName['NN_PerpZ'])
    else:
        deltaPhi = angularrange(DFName[METPhi]+np.pi-DFName[Target_Phi])
    return deltaPhi




def div0( a, b ):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
    return c


'''
def loadData(inputD):

    tree='t'
    arrayName = rnp.root2array(inputD,  branches=[Target_Pt, Target_Phi, 'NVertex' ,
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
'''


# Define a function to make the ellipses
def ellipse(ra,rb,ang,x0,y0,Nb=100):
    xpos,ypos=x0,y0
    radm,radn=ra,rb
    an=ang
    co,si=np.cos(an),np.sin(an)
    the=np.linspace(0,2*np.pi,Nb)
    X=radm*np.cos(the)*co-si*radn*np.sin(the)+xpos
    Y=radm*np.cos(the)*si+co*radn*np.sin(the)+ypos
    return X,Y

def plotHM(x,y, xlabel, ylabel, xmin, xmax, ymin, ymax):

    # Set up default x and y limits
    xlims = [xmin,xmax]
    ylims = [ymin,ymax]


    # Define the locations for the axes
    left, width = 0.12, 0.55
    bottom, height = 0.12, 0.55
    bottom_h = left_h = left+width+0.02

    # Set up the geometry of the three plots
    rect_temperature = [left, bottom, width, height] # dimensions of temp plot
    rect_histx = [left, bottom_h, width, 0.25] # dimensions of x-histogram
    rect_histy = [left_h, bottom, 0.25, height] # dimensions of y-histogram

    # Set up the size of the figure
    fig = plt.figure(1, figsize=(9.5,9))

    # Make the three plots
    axTemperature = plt.axes(rect_temperature) # temperature plot
    axHistx = plt.axes(rect_histx) # x histogram
    axHisty = plt.axes(rect_histy) # y histogram

    # Remove the inner axes numbers of the histograms
    nullfmt = NullFormatter()
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)



    # Make the 'main' temperature plot
    # Define the number of bins
    nxbins = 50
    nybins = 50
    nbinsHM = 100

    xbins = np.linspace(start = xmin, stop = xmax, num = nxbins)
    ybins = np.linspace(start = ymin, stop = ymax, num = nybins)
    xcenter = (xbins[0:-1]+xbins[1:])/2.0
    ycenter = (ybins[0:-1]+ybins[1:])/2.0
    aspectratio = 1.0*(xmax - 0)/(1.0*ymax - 0)

    H, xedges,yedges = np.histogram2d(y,x,bins=(ybins,xbins))
    X = xcenter
    Y = ycenter
    Z = H

    # Plot the temperature data
    cax = (axTemperature.imshow(H, extent=[xmin,xmax,ymin,ymax],
           interpolation='nearest', origin='lower',aspect=aspectratio))

    # Plot the temperature plot contours
    contourcolor = 'white'
    xcenter = np.mean(x)
    ycenter = np.mean(y)
    ra = np.std(x)
    rb = np.std(y)
    ang = 0

    X,Y=ellipse(ra,rb,ang,xcenter,ycenter)
    axTemperature.plot(X,Y,"k:",ms=1,linewidth=2.0)
    axTemperature.annotate('$1\\sigma$', xy=(X[15], Y[15]), xycoords='data',xytext=(10, 10),
                           textcoords='offset points', horizontalalignment='right',
                           verticalalignment='bottom',fontsize=25)

    X,Y=ellipse(2*ra,2*rb,ang,xcenter,ycenter)
    axTemperature.plot(X,Y,"k:",color = contourcolor,ms=1,linewidth=2.0)
    axTemperature.annotate('$2\\sigma$', xy=(X[15], Y[15]), xycoords='data',xytext=(10, 10),
                           textcoords='offset points',horizontalalignment='right',
                           verticalalignment='bottom',fontsize=25, color = contourcolor)

    X,Y=ellipse(3*ra,3*rb,ang,xcenter,ycenter)
    axTemperature.plot(X,Y,"k:",color = contourcolor, ms=1,linewidth=2.0)
    axTemperature.annotate('$3\\sigma$', xy=(X[15], Y[15]), xycoords='data',xytext=(10, 10),
                           textcoords='offset points',horizontalalignment='right',
                           verticalalignment='bottom',fontsize=25, color = contourcolor)

    #Plot the axes labels
    axTemperature.set_xlabel(xlabel,fontsize=25)
    axTemperature.set_ylabel(ylabel,fontsize=25)

    #Make the tickmarks pretty
    ticklabels = axTemperature.get_xticklabels()
    for label in ticklabels:
        label.set_fontsize(18)
        label.set_family('serif')

    ticklabels = axTemperature.get_yticklabels()
    for label in ticklabels:
        label.set_fontsize(18)
        label.set_family('serif')

    #Set up the plot limits
    axTemperature.set_xlim(xlims)
    axTemperature.set_ylim(ylims)

    #Set up the histogram bins
    nbinsHM=50
    xbins = np.arange(xmin, xmax, (xmax-xmin)/nbinsHM)
    ybins = np.arange(ymin, ymax, (ymax-ymin)/nbinsHM)

    #Plot the histograms
    axHistx.hist(x, bins=xbins, color = 'blue')
    axHisty.hist(y, bins=ybins, orientation='horizontal', color = 'red')

    #Set up the histogram limits
    axHistx.set_xlim( xmin, xmax )
    axHisty.set_ylim( ymin, ymax )

    #Make the tickmarks pretty
    ticklabels = axHistx.get_yticklabels()
    for label in ticklabels:
        label.set_fontsize(12)
        label.set_family('serif')

    #Make the tickmarks pretty
    ticklabels = axHisty.get_xticklabels()
    for label in ticklabels:
        label.set_fontsize(12)
        label.set_family('serif')

    #Cool trick that changes the number of tickmarks for the histogram axes
    axHisty.xaxis.set_major_locator(MaxNLocator(4))
    axHistx.yaxis.set_major_locator(MaxNLocator(4))

    #Show the plot
    plt.draw()

    # Save to a File
    #filename = 'myplot'
    #plt.savefig(filename + '.pdf',format = 'pdf', transparent=True)



def plotMVAResponseOverpTZ_woutError(branchString, labelName, errbars_shift, ScaleErr):
    binwidth = (DFName[Target_Pt].values.max() - DFName[Target_Pt].values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName[Target_Pt], bins=nbins)
    sy, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=(getResponse(branchString)))
    sy2, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=(DFName[branchString]/DFName[Target_Pt])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    meanc = np.mean((getResponse(branchString)))
    stdc = np.std((getResponse(branchString)))
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName+'%8.2f $\pm$ %8.2f'%(meanc, stdc), linestyle="None", capsize=0,  color=colors[errbars_shift])


def plotMVANormOverpTZ(branchString, labelName, errbars_shift, ScaleErr):
    binwidth = (DFName[Target_Pt].values.max() - DFName[Target_Pt].values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName[Target_Pt], bins=nbins)
    if branchString=='NN_LongZ':
        NN_Pt = np.sqrt(np.multiply(DFName['NN_LongZ'],DFName['NN_LongZ'])+np.multiply(DFName['NN_PerpZ'],DFName['NN_PerpZ']))
        sy, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=(NN_Pt/DFName[Target_Pt]))
        sy2, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=(NN_Pt/DFName[Target_Pt])**2)
        meanc = np.mean(NN_Pt/DFName[Target_Pt])
        stdc = np.std(NN_Pt/DFName[Target_Pt])
    else:
        sy, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=((DFName[branchString])/DFName[Target_Pt]))
        sy2, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=(DFName[branchString]/DFName[Target_Pt])**2)
        meanc = np.mean(DFName[branchString]/DFName[Target_Pt])
        stdc = np.std(DFName[branchString]/DFName[Target_Pt])
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName+'%8.2f $\pm$ %8.2f'%(meanc, stdc), linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAAngularOverpTZ(branchStringPhi, labelName, errbars_shift, ScaleErr):
    binwidth = (DFName[Target_Pt].values.max() - DFName[Target_Pt].values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName[Target_Pt], bins=nbins)
    sy, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=getAngle(branchStringPhi))
    sy2, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=(getAngle(branchStringPhi))**2)
    meanc = np.mean(getAngle(branchStringPhi))
    stdc = np.std(getAngle(branchStringPhi))
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName+'%8.2f $\pm$ %8.2f'%(meanc, stdc), linestyle="None", capsize=0,  color=colors[errbars_shift])



def plotMVANormOverpTZ_wErr(branchString, labelName, errbars_shift, ScaleErr):
    binwidth = (DFName[Target_Pt].values.max() - DFName[Target_Pt].values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName[Target_Pt][getResponseIdx(branchString)], bins=nbins)
    sy, _ = np.histogram(DFName[Target_Pt][getResponseIdx(branchString)], bins=nbins, weights=getResponse(branchString))
    sy2, _ = np.histogram(DFName[Target_Pt][getResponseIdx(branchString)], bins=nbins, weights=getResponse(branchString)**2)
    meanc = np.mean(getResponse(branchString))
    stdc = np.std(getResponse(branchString))
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName+'%8.2f $\pm$ %8.2f'%(meanc, stdc), linestyle="None", capsize=0,  color=colors[errbars_shift])
    if errbars_shift==5: errbars_shift2 = 0
    elif errbars_shift==6: errbars_shift2 = 2
    else: errbars_shift2 = errbars_shift
    plt.errorbar((_[:-1]+(_[1:]-_[:-1])/3*errbars_shift2), mean, yerr=std*ScaleErr, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAAngularOverpTZ_wErr(branchStringPhi, labelName, errbars_shift, ScaleErr):
    nbins=1000
    ScaleErr=1
    binwidth = (DFName[Target_Pt].values.max() - DFName[Target_Pt].values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName[Target_Pt], bins=nbins)
    sy, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=(getAngle(branchStringPhi)))
    sy2, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=(getAngle(branchStringPhi))**2)
    meanc = np.mean(getAngle(branchStringPhi))
    stdc = np.std(getAngle(branchStringPhi))
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    #print('std', std)
    Bool45Degrees = std<45*np.pi/180

    indSmaller45Degrees = [i for i,x in enumerate(Bool45Degrees) if x==True]
    #print(std<45*np.pi/180)
    #print(np.where(std<45*np.pi/180))
    #print(indSmaller45Degrees)
    print('MET definition std under 45 degrees', labelName)
    print('pT bin Start with std under 45 degrees', _[indSmaller45Degrees[0:10]])
    print('pT bin End with std under 45 degrees', _[indSmaller45Degrees[0:10]]+binwidth)
    print('crosscheck std with std under 45 degrees', std[indSmaller45Degrees[0:10]])
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName+'%8.2f $\pm$ %8.2f'%(meanc, stdc), linestyle="None", capsize=0,  color=colors[errbars_shift])
    if errbars_shift==5: errbars_shift2 = 0
    elif errbars_shift==6: errbars_shift2 = 2
    else: errbars_shift2 = errbars_shift
    plt.errorbar((_[:-1]+(_[1:]-_[:-1])/3*errbars_shift2), mean, yerr=std*ScaleErr, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])



def plotMVAResponseOverpTZ_woutError_Tresh(branchString, labelName, errbars_shift, ScaleErr):
    DFName_Tresh = DFName[DFName[Target_Pt]>pTTresh]
    binwidth = (DFName_Tresh.Boson_Pt.values.max() - DFName_Tresh.Boson_Pt.values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName_Tresh.Boson_Pt, bins=nbins)
    sy, _ = np.histogram(DFName_Tresh.Boson_Pt, bins=nbins, weights=(-(DFName_Tresh[branchString])/DFName_Tresh.Boson_Pt))
    sy2, _ = np.histogram(DFName_Tresh.Boson_Pt, bins=nbins, weights=(DFName_Tresh[branchString]/DFName_Tresh.Boson_Pt)**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])


def plotMVAResponseOverNVertex_woutError(branchString, labelName, errbars_shift, ScaleErr, rangemin, rangemax):
    IndexRange = (DFName_nVertex[Target_Pt]>rangemin) & (DFName_nVertex[Target_Pt]<rangemax)
    DF_Response_PV = DFName_nVertex[(DFName_nVertex[Target_Pt]>rangemin) & (DFName_nVertex[Target_Pt]<rangemax)]
    binwidth = (DF_Response_PV.NVertex.values.max() - DF_Response_PV.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DF_Response_PV.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DF_Response_PV.NVertex, bins=nbinsVertex, weights=getResponse(branchString)[IndexRange])
    sy2, _ = np.histogram(DF_Response_PV.NVertex, bins=nbinsVertex, weights=(getResponse(branchString)[IndexRange])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    meanc=np.mean(getResponse(branchString))
    stdc=np.std(getResponse(branchString))
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName+'%8.2f $\pm$ %8.2f'%(meanc, stdc), linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAResponseOverNVertex_woutError_Tresh(branchString, labelName, errbars_shift, ScaleErr):
    DFName_nVertex_Tresh = DFName_nVertex[DFName_nVertex[Target_Pt]>pTTresh]
    binwidth = (DFName_nVertex_Tresh.NVertex.values.max() - DFName_nVertex_Tresh.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName_nVertex_Tresh.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName_nVertex_Tresh.NVertex, bins=nbinsVertex, weights=-(DFName_nVertex_Tresh[branchString])/DFName_nVertex_Tresh[Target_Pt])
    sy2, _ = np.histogram(DFName_nVertex_Tresh.NVertex, bins=nbinsVertex, weights=(DFName_nVertex_Tresh[branchString]/DFName_nVertex_Tresh[Target_Pt])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def pT_PVbins(branchString, labelName, errbars_shift, ScaleErr, rangemin, rangemax):
    #IdxPVbins = (DFName_nVertex[Target_Pt]>=rangemin) & (DFName_nVertex[Target_Pt]<rangemax)
    DV_PVbins = DFName_nVertex[(DFName_nVertex[Target_Pt]>=rangemin) & (DFName_nVertex[Target_Pt]<rangemax)]
    binwidth = (DV_PVbins.NVertex.values.max() - DV_PVbins.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DV_PVbins.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DV_PVbins.NVertex, bins=nbinsVertex, weights=DV_PVbins[Target_Pt])
    sy2, _ = np.histogram(DV_PVbins.NVertex, bins=nbinsVertex, weights=(DV_PVbins[Target_Pt])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    meanc= np.mean(DV_PVbins[Target_Pt])
    stdc=np.std(DV_PVbins[Target_Pt])
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName+', %8.2f $\pm$ %8.2f'%(meanc, stdc), linestyle="None", capsize=0,  color=colors[errbars_shift])
    plt.errorbar((_[:-1]+(_[1:]-_[:-1])/2), mean, yerr=std*ScaleErr, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])


def plotMVAResponseOverpTZ_wError(branchString, labelName, errbars_shift, ScaleErr):
    binwidth = (DFName[Target_Pt].values.max() - DFName[Target_Pt].values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName[Target_Pt], bins=nbins)
    sy, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=getResponse(branchString))
    sy2, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=(getResponse(branchString))**2)
    mean = sy / n
    std = np.divide(np.sqrt(sy2/n - mean*mean), n)
    if errbars_shift==5: errbars_shift2 = 0
    elif errbars_shift==6: errbars_shift2 = 2
    else: errbars_shift2 = errbars_shift
    plt.fill_between((_[1:] + _[:-1])/2, mean-std*ScaleErr, mean+std*ScaleErr, alpha=0.2, edgecolor=colors[errbars_shift], facecolor=colors[errbars_shift])

    #plt.errorbar((_[:-1]+(_[1:]-_[:-1])/3*errbars_shift2), mean, yerr=std*ScaleErr, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])



def plotMVAResponseOverNVertex_wError(branchString, labelName, errbars_shift, ScaleErr):
    binwidth = (DFName_nVertex.NVertex.values.max() - DFName_nVertex.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=getResponse(branchString))
    sy2, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(getResponse(branchString))**2)
    mean = sy / n
    std = np.divide(np.sqrt(sy2/n - mean*mean), n)
    if errbars_shift==5: errbars_shift2 = 0
    elif errbars_shift==6: errbars_shift2 = 2
    else: errbars_shift2 = errbars_shift
    plt.fill_between((_[1:] + _[:-1])/2, mean-std*ScaleErr, mean+std*ScaleErr, alpha=0.2, edgecolor=colors[errbars_shift], facecolor=colors[errbars_shift])


    #plt.errorbar((_[:-1]+(_[1:]-_[:-1])/3*errbars_shift2), mean, yerr=std*ScaleErr, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])


def plotMVAResolutionOverpTZ_woutError_para(branchString, labelName, errbars_shift, ScaleErr):
    binwidth = (DFName[Target_Pt].values.max() - DFName[Target_Pt].values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName[Target_Pt], bins=nbins)
    sy, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=-(DFName[branchString]+DFName[Target_Pt]))
    sy2, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=(DFName[branchString]+DFName[Target_Pt])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, std, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])



def mean_Response(branchString_Long, branchString_Phi, labelName, errbars_shift, ScaleErr):
    binwidth = (-np.pi - np.pi)/binsAngle
    n, _ = np.histogram(DFName[Target_Pt], bins=nbins)
    NN_phi = getAngle(branchString_Phi)[getResponseIdx(branchString_Long)]

    n, _ = np.histogram(NN_phi, bins=20)
    sy, _ = np.histogram(NN_phi, bins=20, weights=getResponse(branchString_Long))
    sy2, _ = np.histogram(NN_phi, bins=20, weights=(getResponse(branchString_Long))**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])
    if errbars_shift==5: errbars_shift2 = 0
    elif errbars_shift==6: errbars_shift2 = 2
    else: errbars_shift2 = errbars_shift
    #plt.errorbar((_[:-1]+(_[1:]-_[:-1])/2*errbars_shift2), mean, yerr=std*ScaleErr, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift], linewidth=1.0)

def mean_Response_wErr(branchString_Long, branchString_Phi, labelName, errbars_shift, ScaleErr, rangemin, rangemax, nbins):
    binwidth = (-np.pi - np.pi)/(binsAngle)
    IndexRangeResponse = (getResponseIdx(branchString_Long)) & (DFName[Target_Pt]>rangemin) & (DFName[Target_Pt]<rangemax)
    IndexRange = (DFName[Target_Pt]>rangemin) & (DFName[Target_Pt]<rangemax)
    NN_phi = getAngle(branchString_Phi)[IndexRangeResponse]

    n, _ = np.histogram(NN_phi, bins=nbins)
    sy, _ = np.histogram(NN_phi, bins=nbins, weights=getResponse(branchString_Long)[IndexRange])
    sy2, _ = np.histogram(NN_phi, bins=nbins, weights=(getResponse(branchString_Long)[IndexRange])**2)
    mean = sy / n
    #std = np.sqrt(sy2/n - mean*mean)
    #print('pT von grossen Errors von ', branchString_Long, ' ist ', DFName[Target_Pt][getResponseIdx(branchString_Long) and np.abs(getResponse(branchString_Long))>10])
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])
    std = 1. / n
    print('std', std)
    print('n', n)
    if errbars_shift==5: errbars_shift2 = 0
    elif errbars_shift==6: errbars_shift2 = 2
    else: errbars_shift2 = errbars_shift
    plt.fill_between((_[1:] + _[:-1])/2, mean-std*ScaleErr, mean+std*ScaleErr, alpha=0.2, edgecolor=colors[errbars_shift], facecolor=colors[errbars_shift])

    #plt.errorbar((_[:-1]+(_[1:]-_[:-1])/2*errbars_shift2), mean, yerr=std*0.1, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift], linewidth=1.0)



def mean_Response_CR(branchString_Long, branchString_Phi, labelName, errbars_shift, ScaleErr):
    binwidth = (-np.pi/180*10 - np.pi/180*10)/binsAngle
    n, _ = np.histogram(DFName[Target_Pt], bins=nbins)
    NN_phi = getAngle(branchString_Phi)[getResponseIdx(branchString_Long)]

    n, _ = np.histogram(NN_phi, bins=20, range=[-np.pi/180*10, np.pi/180*10])
    sy, _ = np.histogram(NN_phi, bins=20, weights=getResponse(branchString_Long), range=[-np.pi/180*10, np.pi/180*10])
    sy2, _ = np.histogram(NN_phi, bins=20, weights=getResponse(branchString_Long)**2, range=[-np.pi/180*10, np.pi/180*10])
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])
    if errbars_shift==5: errbars_shift2 = 0
    elif errbars_shift==6: errbars_shift2 = 2
    else: errbars_shift2 = errbars_shift
    plt.errorbar((_[:-1]+(_[1:]-_[:-1])/2*errbars_shift2), mean, yerr=std*ScaleErr, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift], linewidth=1.0)


def plotMVAResolutionOverNVertex_woutError_para(branchString, labelName, errbars_shift, ScaleErr, rangemin, rangemax):
    #IdxRange = (DFName_nVertex[Target_Pt]>=rangemin) & (DFName_nVertex[Target_Pt]<rangemin)
    DF_Resolution_PV = DFName_nVertex[(DFName_nVertex[Target_Pt]>=rangemin) & (DFName_nVertex[Target_Pt]<rangemax)]
    binwidth = (50)/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DF_Resolution_PV.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DF_Resolution_PV.NVertex, bins=nbinsVertex, weights=-(DF_Resolution_PV[branchString]+DF_Resolution_PV[Target_Pt]))
    sy2, _ = np.histogram(DF_Resolution_PV.NVertex, bins=nbinsVertex, weights=(DF_Resolution_PV[branchString]+DF_Resolution_PV[Target_Pt])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, std, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def MeanDeviation_Pt(branchString, labelName, errbars_shift, ScaleErr):
    binwidth = (DFName[Target_Pt].values.max() - DFName[Target_Pt].values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName[Target_Pt], bins=nbins)
    sy, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=-(DFName[branchString]+DFName[Target_Pt]))
    sy2, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=(DFName[branchString]+DFName[Target_Pt])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])


def MeanDeviation_PV(branchString, labelName, errbars_shift, ScaleErr):
    binwidth = (DFName_nVertex.NVertex.values.max() - DFName_nVertex.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=-(DFName_nVertex[branchString]+DFName_nVertex[Target_Pt]))
    sy2, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(DFName_nVertex[branchString]+DFName_nVertex[Target_Pt])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAResolutionOverpTZ_woutError_perp(branchString, labelName, errbars_shift, ScaleErr):
    binwidth = (DFName[Target_Pt].values.max() - DFName[Target_Pt].values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName[Target_Pt], bins=nbins)
    sy, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=-(DFName[branchString]))
    sy2, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=(DFName[branchString])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, std, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAResolutionOverpTZ_woutError_perp_RC(branchString, labelName, errbars_shift, ScaleErr):
    binwidth = (DFName[Target_Pt].values.max() - DFName[Target_Pt].values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName[Target_Pt], bins=nbins)
    sy, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=(DFName[branchString]))
    sy2, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=(DFName[branchString])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    sy_resp, _resp = np.histogram(DFName[Target_Pt], bins=nbins, weights=-(DFName[branchString]+DFName[Target_Pt]))
    mean_resp = sy_resp / n
    plt.errorbar((_[1:] + _[:-1])/2, div0(std,mean_resp), marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])



def Histogram_Deviation_para_pT(branchString, labelName, errbars_shift, ScaleErr):
    Mean = np.mean(-(DFName[branchString])-DFName[Target_Pt].values)
    Std = np.std(-(DFName[branchString])-DFName[Target_Pt].values)
    if branchString in ['NN_LongZ', 'recoilslimmedMETs_LongZ']:
        plt.hist((-(DFName[branchString])-DFName[Target_Pt].values), bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift], linewidth=1.5)
    else:
        plt.hist((-(DFName[branchString])-DFName[Target_Pt].values), bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift])


def Histogram_Deviation_perp_pT(branchString, labelName, errbars_shift, ScaleErr):
    Mean = np.mean((DFName[branchString]))
    Std = np.std((DFName[branchString]))
    if branchString in ['NN_PerpZ', 'recoilslimmedMETs_PerpZ']:
        plt.hist(((DFName[branchString])), bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift], linewidth=1.5)
    else:
        plt.hist(((DFName[branchString])), bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift])

def Hist_Response(branchString, labelName, errbars_shift, ScaleErr, rangemin, rangemax):
    #DFName.loc[DFName[Target_Pt]]
    Response = -(DFName[branchString][(DFName[Target_Pt]>rangemin) & (DFName[Target_Pt]<rangemax)])/DFName[Target_Pt][(DFName[Target_Pt]>rangemin) & (DFName[Target_Pt]<rangemax)]
    Response = Response[~np.isnan(Response)]
    Mean = np.mean(Response)
    Std = np.std(Response)

    plt.hist(Response, bins=nbinsHist, range=[ResponseMin, ResponseMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift])

def Hist_InvMET(branchString, labelName, errbars_shift, ScaleErr):
    Response2 = np.divide(-(DFName[branchString]), DFName[Target_Pt])
    Response = np.divide(1, DFName[Target_Pt])
    Response = Response[~np.isnan(Response2)]
    Mean = np.mean(Response)
    Std = np.std(Response)

    plt.hist(Response, bins=nbinsHist, range=[ResponseMin, ResponseMax], label='$\\frac{1}{p_T^Z}$, %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift])


def Hist_Resolution_para(branchString, labelName, errbars_shift, ScaleErr, rangemin, rangemax):
    Mean = np.mean((-(DFName[branchString][(DFName[Target_Pt]>rangemin) & (DFName[Target_Pt]<rangemax)])-DFName[Target_Pt][(DFName[Target_Pt]>rangemin) & (DFName[Target_Pt]<rangemax)]))
    Std = np.std((-(DFName[branchString][(DFName[Target_Pt]>rangemin) & (DFName[Target_Pt]<rangemax)])-DFName[Target_Pt][(DFName[Target_Pt]>rangemin) & (DFName[Target_Pt]<rangemax)]))
    plt.hist((-(DFName[branchString][(DFName[Target_Pt]>rangemin) & (DFName[Target_Pt]<rangemax)])-DFName[Target_Pt][(DFName[Target_Pt]>rangemin) & (DFName[Target_Pt]<rangemax)]), bins=nbinsHist, range=[ResolutionParaMin, ResolutionParaMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift])

def Hist_Resolution_para_0_100(branchString, labelName, errbars_shift, ScaleErr):
    DFName_DC = DFName
    DFName_DC = DFName_DC[DFName_DC[Target_Pt]<=100]
    Mean = np.mean((-(DFName_DC[branchString])-DFName_DC[Target_Pt]))
    Std = np.std((-(DFName_DC[branchString])-DFName_DC[Target_Pt]))
    plt.hist((-(DFName_DC[branchString])-DFName_DC[Target_Pt]), bins=nbinsHist, range=[ResolutionParaMin, ResolutionParaMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift])

def Hist_Resolution_para_100_150(branchString, labelName, errbars_shift, ScaleErr):
    DFName_DC = DFName
    DFName_DC = DFName_DC[DFName_DC[Target_Pt]>100]
    DFName_DC = DFName_DC[DFName_DC[Target_Pt]<=150]
    Mean = np.mean((-(DFName_DC[branchString])-DFName_DC[Target_Pt]))
    Std = np.std((-(DFName_DC[branchString])-DFName_DC[Target_Pt]))
    plt.hist((-(DFName_DC[branchString])-DFName_DC[Target_Pt]), bins=nbinsHist, range=[ResolutionParaMin, ResolutionParaMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift])

def Hist_Resolution_para_150_200(branchString, labelName, errbars_shift, ScaleErr):
    DFName_DC = DFName
    DFName_DC = DFName_DC[DFName_DC[Target_Pt]>150]
    DFName_DC = DFName_DC[DFName_DC[Target_Pt]<=200]
    Mean = np.mean((-(DFName_DC[branchString])-DFName_DC[Target_Pt]))
    Std = np.std((-(DFName_DC[branchString])-DFName_DC[Target_Pt]))
    plt.hist((-(DFName_DC[branchString])-DFName_DC[Target_Pt]), bins=nbinsHist, range=[ResolutionParaMin, ResolutionParaMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift])


def Hist_Resolution_perp(branchString, labelName, errbars_shift, ScaleErr, rangemin, rangemax):
    Mean = np.mean(DFName[branchString][(DFName[Target_Pt]>rangemin) & (DFName[Target_Pt]<rangemax)])
    Std = np.std(DFName[branchString][(DFName[Target_Pt]>rangemin) & (DFName[Target_Pt]<rangemax)])
    plt.hist(DFName[branchString][(DFName[Target_Pt]>rangemin) & (DFName[Target_Pt]<rangemax)], bins=nbinsHist, range=[ResolutionPerpMin, ResolutionPerpMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift])


def Histogram_Deviation_para_Bin(branchString, labelName, bin):
    Mean = np.mean(-(DFName[branchString])-DFName[Target_Pt].values)
    Std = np.std(-(DFName[branchString])-DFName[Target_Pt].values)
    n, _ = np.histogram(-(DFName[branchString])-DFName[Target_Pt].values, bins=nbinsHistBin)
    plt.hist((-(DFName[branchString])-DFName[Target_Pt].values), bins=nbinsHistBin, range=(HistLimMin, HistLimMax), label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors2[bin])


def Histogram_Response(branchString, labelName, bin, ScaleErr):
    Mean = np.mean(np.divide(-(DFName[branchString]),DFName[Target_Pt].values))
    Std = np.std(np.divide(-(DFName[branchString]),DFName[Target_Pt].values))
    Reso = np.divide(-(DFName[branchString]),DFName[Target_Pt].values)
    #n, _ = np.histogram(-(DFName[branchString])-DFName[Target_Pt].values, bins=nbinsHistBin)
    plt.hist(Reso[~np.isnan(Reso)], bins=nbinsHistBin, range=[ResponseMin, ResponseMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors2[bin])



def Histogram_Norm_Comparison(branchStringPt,  labelName, errbars_shift, ScaleErr):
    Norm_ = DFName[branchStringPt]
    Mean = np.mean(Norm_-DFName[Target_Pt].values)
    Std = np.std(Norm_-DFName[Target_Pt].values)
    if branchStringPt in ['NN_Pt', 'recoilslimmedMETs_Pt']:
        plt.hist(Norm_-DFName[Target_Pt].values, bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift], linewidth=1.5)
    else:
        plt.hist(Norm_-DFName[Target_Pt].values, bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift])

def Histogram_Angle_Dev(branchStringPhi, labelName, errbars_shift, ScaleErr):
    print('getAngle(branchStringPhi).shape', getAngle(branchStringPhi).shape)
    if branchStringPhi in ['NN_Phi', 'recoilslimmedMETs_Phi']:
        plt.hist(getAngle(branchStringPhi), bins=nbinsHist, range=[-np.pi, np.pi], label=labelName+', %8.2f $\pm$ %8.2f'%(np.mean(getAngle(branchStringPhi)), np.std(getAngle(branchStringPhi))), histtype='step', ec=colors_InOut[errbars_shift], linewidth=1.5)
    else:
        plt.hist(getAngle(branchStringPhi), bins=nbinsHist, range=[-np.pi, np.pi], label=labelName+', %8.2f $\pm$ %8.2f'%(np.mean(getAngle(branchStringPhi)), np.std(getAngle(branchStringPhi))), histtype='step', ec=colors_InOut[errbars_shift])


def Histogram_Norm(branchStringNorm,  labelName, errbars_shift, ScaleErr):
    Norm_ = DFName[branchStringNorm]
    Mean = np.mean(Norm_)
    Std = np.std(Norm_)
    if branchStringNorm in ['NN_Pt', 'recoilslimmedMETs_Pt']:
        plt.hist(Norm_, bins=nbinsHist, range=[0, 75], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift], linewidth=1.5)
    else:
        plt.hist(Norm_, bins=nbinsHist, range=[0, 75], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift])

def Histogram_Norm_Pt(branchStringLong, labelName, errbars_shift, ScaleErr):
    Norm_ = DFName[Target_Pt].values
    Mean = np.mean(DFName[Target_Pt].values)
    Std = np.std(Norm_)
    plt.hist(Norm_, bins=nbinsHist, range=[0, 75], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift], linewidth=1.5)


def Histogram_Deviation_para_PV(branchString, labelName, errbars_shift, ScaleErr):
    Mean = np.mean(-(DFName_nVertex[branchString])-DFName_nVertex[Target_Pt].values)
    Std = np.std(-(DFName_nVertex[branchString])-DFName_nVertex[Target_Pt].values)
    if branchString in ['NN_LongZ', 'recoilslimmedMETs_LongZ']:
        plt.hist((-(DFName_nVertex[branchString])-DFName_nVertex[Target_Pt].values), bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift], linewidth=1.5)
    else:
        plt.hist((-(DFName_nVertex[branchString])-DFName_nVertex[Target_Pt].values), bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift])


def Histogram_Deviation_perp_PV(branchString, labelName, errbars_shift, ScaleErr):
    Mean = np.mean((DFName_nVertex[branchString]))
    Std = np.std((DFName_nVertex[branchString]))
    if branchString in ['NN_PerpZ', 'recoilslimmedMETs_PerpZ']:
        plt.hist(((DFName_nVertex[branchString])), bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift], linewidth=1.5)
    else:
        plt.hist(((DFName_nVertex[branchString])), bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift])

def Hist_LongZ(branchString, labelName, errbars_shift, ScaleErr):
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


def Hist_PerpZ(branchString, labelName, errbars_shift, ScaleErr):
    Mean = np.mean((DFName_nVertex[branchString]))
    Std = np.std((DFName_nVertex[branchString]))
    if branchString in ['NN_PerpZ', 'recoilslimmedMETs_PerpZ']:
        plt.hist(((DFName_nVertex[branchString])), bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift], linewidth=1.5)
    else:
        plt.hist(((DFName_nVertex[branchString])), bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift])


def Histogram_relResponse(branchString, labelName, errbars_shift, ScaleErr):
    binwidth = ((-(DFName[branchString])-DFName[Target_Pt].values).max() - (-(DFName[branchString])-DFName[Target_Pt].values).min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram((-(DFName[branchString])-DFName[Target_Pt].values), bins=nbins_relR)
    plt.hist(div0((-(DFName[branchString])-DFName[Target_Pt].values),DFName[Target_Pt].values), bins=nbinsHist, range=[-20, 20], label=labelName, histtype='step', ec=colors[errbars_shift])

def Histogram_relResponse_PV(branchString, labelName, errbars_shift, ScaleErr):
    binwidth = ((-(DFName_nVertex[branchString])-DFName_nVertex[Target_Pt].values).max() - (-(DFName_nVertex[branchString])-DFName_nVertex[Target_Pt].values).min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram((-(DFName_nVertex[branchString])-DFName_nVertex[Target_Pt].values), bins=nbins)
    plt.hist(div0((-(DFName_nVertex[branchString])-DFName_nVertex[Target_Pt].values), DFName_nVertex[Target_Pt].values), bins=nbinsHist, range=[-20, 20], label=labelName, histtype='step', ec=colors[errbars_shift])



def NN_Response_pT( labelName, errbars_shift, ScaleErr):
    binwidth = (DFName[Target_Pt].values.max() - DFName[Target_Pt].values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName[Target_Pt], bins=nbins)
    sy, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=getResponse('NN_LongZ'))
    sy2, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=getResponse('NN_LongZ')**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def NN_Response_PV(branchString, labelName, errbars_shift, ScaleErr):
    binwidth = (DFName_nVertex.NVertex.values.max() - DFName_nVertex.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=getResponse(branchString))
    sy2, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(getResponse(branchString))**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAResponseOverpTZ_wError(branchString, labelName, errbars_shift, ScaleErr):
    binwidth = (DFName[Target_Pt].values.max() - DFName[Target_Pt].values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName[Target_Pt], bins=nbins)
    sy, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=(getResponse(branchString)))
    sy2, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=(getResponse(branchString))**2)
    mean = sy / n
    std = np.divide(np.sqrt(sy2/n - mean*mean), n)
    if errbars_shift==5: errbars_shift2 = 0
    elif errbars_shift==6: errbars_shift2 = 2
    else: errbars_shift2 = errbars_shift
    plt.fill_between((_[1:] + _[:-1])/2, mean-std*ScaleErr, mean+std*ScaleErr, alpha=0.2, edgecolor=colors[errbars_shift], facecolor=colors[errbars_shift])

    #plt.errorbar((_[:-1]+(_[1:]-_[:-1])/3*errbars_shift2), mean, yerr=std*ScaleErr, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAResponseOverNVertex_wError(branchString, labelName, errbars_shift, ScaleErr):
    binwidth = (DFName_nVertex.NVertex.values.max() - DFName_nVertex.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=getResponse(branchString))
    sy2, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(getResponse(branchString))**2)
    mean = sy / n
    std = np.divide(np.sqrt(sy2/n - mean*mean), n)
    if errbars_shift==5: errbars_shift2 = 0
    elif errbars_shift==6: errbars_shift2 = 2
    else: errbars_shift2 = errbars_shift
    plt.fill_between((_[1:] + _[:-1])/2, mean-std*ScaleErr, mean+std*ScaleErr, alpha=0.2, edgecolor=colors[errbars_shift], facecolor=colors[errbars_shift])

    #plt.errorbar((_[:-1]+(_[1:]-_[:-1])/3*errbars_shift2), mean, yerr=std*ScaleErr, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])
    #plt.plot((_[1:] + _[:-1])/2, mean, marker='.', label=labelName, linestyle="None", color=MVAcolors[errbars_shift])
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])


def plotMVAResolutionOverpTZ_woutError_para(branchString, labelName, errbars_shift, ScaleErr):
    binwidth = (DFName[Target_Pt].values.max() - DFName[Target_Pt].values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName[Target_Pt], bins=nbins)
    sy, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=-(DFName[branchString]+DFName[Target_Pt]))
    sy2, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=(DFName[branchString]+DFName[Target_Pt])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, std, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])


def plotMVAResolutionOverpTZ_woutError_para_RC(branchString, labelName, errbars_shift, ScaleErr):
    binwidth = (DFName[Target_Pt].values.max() - DFName[Target_Pt].values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName[Target_Pt], bins=nbins)
    sy, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=-(DFName[branchString]+DFName[Target_Pt]))
    sy2, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=(DFName[branchString]+DFName[Target_Pt])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    sy_resp, _resp = np.histogram(DFName[Target_Pt], bins=nbins, weights=-(DFName[branchString]+DFName[Target_Pt]))
    mean_resp = sy_resp / n
    plt.errorbar((_[1:] + _[:-1])/2, div0(std,mean_resp), marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAResolutionOverNVertex_woutError_para_RC(branchString, labelName, errbars_shift, ScaleErr):
    binwidth = (DFName_nVertex.NVertex.values.max() - DFName_nVertex.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=-(DFName_nVertex[branchString]+DFName_nVertex[Target_Pt]))
    sy2, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(DFName_nVertex[branchString]+DFName_nVertex[Target_Pt])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    sy_resp, _resp = np.histogram(DFName_nVertex[Target_Pt], bins=nbinsVertex, weights=-(DFName_nVertex[branchString]+DFName_nVertex[Target_Pt]))
    mean_resp = sy_resp / n
    plt.errorbar((_[1:] + _[:-1])/2, div0(std,mean_resp), marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])


def plotMVAResolutionOverpTZ_woutError_perp(branchString, labelName, errbars_shift, ScaleErr):
    binwidth = (DFName[Target_Pt].values.max() - DFName[Target_Pt].values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName[Target_Pt], bins=nbins)
    sy, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=-(DFName[branchString]))
    sy2, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=(DFName[branchString])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, std, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAResolutionOverNVertex_woutError_perp(branchString, labelName, errbars_shift, ScaleErr, rangemin, rangemax):
    #IdxRange = (DFName_nVertex[Target_Pt]>=rangemin) & (DFName_nVertex[Target_Pt]<rangemin)
    DF_Resolution_pe_PV = DFName_nVertex[(DFName_nVertex[Target_Pt]>=rangemin) & (DFName_nVertex[Target_Pt]<rangemax)]
    binwidth = (50)/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DF_Resolution_pe_PV.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DF_Resolution_pe_PV.NVertex, bins=nbinsVertex, weights=-(DF_Resolution_pe_PV[branchString]))
    sy2, _ = np.histogram(DF_Resolution_pe_PV.NVertex, bins=nbinsVertex, weights=(DF_Resolution_pe_PV[branchString])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, std, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAResolutionOverpTZ_woutError_perp_RC(branchString, labelName, errbars_shift, ScaleErr):
    binwidth = (DFName[Target_Pt].values.max() - DFName[Target_Pt].values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName[Target_Pt], bins=nbins)
    sy, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=-(DFName[branchString]))
    sy2, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=(DFName[branchString])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    sy_resp, _resp = np.histogram(DFName[Target_Pt], bins=nbins, weights=-(DFName[branchString]+DFName[Target_Pt]))
    mean_resp = sy_resp / n
    plt.errorbar((_[1:] + _[:-1])/2, div0(std,mean_resp), marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAResolutionOverNVertex_woutError_perp_RC(branchString, labelName, errbars_shift, ScaleErr):
    binwidth = (DFName_nVertex.NVertex.values.max() - DFName_nVertex.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(DFName_nVertex[branchString]))
    sy2, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(DFName_nVertex[branchString])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    sy_resp, _resp = np.histogram(DFName_nVertex[Target_Pt], bins=nbinsVertex, weights=-(DFName_nVertex[branchString]+DFName_nVertex[Target_Pt]))
    mean_resp = sy_resp / n
    plt.errorbar((_[1:] + _[:-1])/2, div0(std,mean_resp), marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def Histogram_Deviation_perp(branchString, labelName, errbars_shift, ScaleErr):
    Mean = np.mean(DFName[branchString])
    Std = np.std(DFName[branchString])
    plt.hist(DFName[branchString], bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors[errbars_shift])

def Mean_Std_Deviation_pTZ_para(branchString, labelName, errbars_shift, ScaleErr):
    binwidth = (DFName[Target_Pt].values.max() - DFName[Target_Pt].values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName[Target_Pt], bins=nbins)
    sy, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=(-(DFName[branchString])-DFName[Target_Pt].values))
    sy2, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=(-(DFName[branchString])-DFName[Target_Pt].values)**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])
    if errbars_shift==5: errbars_shift2 = 0
    elif errbars_shift==6: errbars_shift2 = 2
    else: errbars_shift2 = errbars_shift
    plt.errorbar((_[:-1]+(_[1:]-_[:-1])/3*errbars_shift2), mean, yerr=std*ScaleErr, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])

def Mean_Std_Deviation_pTZ_perp(branchString, labelName, errbars_shift, ScaleErr):
    binwidth = (DFName[Target_Pt].values.max() - DFName[Target_Pt].values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName[Target_Pt], bins=nbins)
    sy, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=(DFName[branchString]))
    sy2, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=(DFName[branchString])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])
    if errbars_shift==5: errbars_shift2 = 0
    elif errbars_shift==6: errbars_shift2 = 2
    else: errbars_shift2 = errbars_shift
    plt.errorbar((_[:-1]+(_[1:]-_[:-1])/3*errbars_shift2), mean, yerr=std*ScaleErr, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])

def Mean_Std_Deviation_PV_para(branchString, labelName, errbars_shift, ScaleErr):
    binwidth = (DFName_nVertex.NVertex.values.max() - DFName_nVertex.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(-(DFName_nVertex[branchString])-DFName_nVertex.NVertex.values))
    sy2, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(-(DFName_nVertex[branchString])-DFName_nVertex.NVertex.values)**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])
    if errbars_shift==5: errbars_shift2 = 0
    elif errbars_shift==6: errbars_shift2 = 2
    else: errbars_shift2 = errbars_shift
    plt.errorbar((_[:-1]+(_[1:]-_[:-1])/3*errbars_shift2), mean, yerr=std*ScaleErr, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])

def Mean_Std_Deviation_PV_perp(branchString, labelName, errbars_shift, ScaleErr):
    binwidth = (DFName_nVertex.NVertex.values.max() - DFName_nVertex.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(DFName_nVertex[branchString]))
    sy2, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(DFName_nVertex[branchString])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])
    if errbars_shift==5: errbars_shift2 = 0
    elif errbars_shift==6: errbars_shift2 = 2
    else: errbars_shift2 = errbars_shift
    plt.errorbar((_[:-1]+(_[1:]-_[:-1])/3*errbars_shift2), mean, yerr=std*ScaleErr, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])





def getPlotsOutput(inputD, filesD, plotsD,DFName, DFName_nVertex, Target_Pt, Target_Phi):



    #Plot settings
    ScaleErr = 1
    NPlotsLines = 6
    MVA_NPlotsLines = 3
    pTRangeString_Err = '$20\ \mathrm{GeV} < |-\\vec{p}_T^Z| \leq 200\ \mathrm{GeV}$ \n $\mathrm{\# Vertex} \leq 50$ \\ Error scaled: '+str(ScaleErr)
    pTRangeString= '$20\ \mathrm{GeV} < |-\\vec{p}_T^Z| \leq 200\ \mathrm{GeV}$ \n $\mathrm{\# Vertex} \leq 50$'
    pTRangeString_low= '$0\ \mathrm{GeV} < |-\\vec{p}_T^Z| \leq %8.2f \ \mathrm{GeV}$ \n $\mathrm{\# Vertex} \leq 50$'%(np.percentile(DFName[Target_Pt],0.3333*100))
    pTRangeString_mid= '$%8.2f\ \mathrm{GeV} < |-\\vec{p}_T^Z| \leq %8.2f\ \mathrm{GeV}$ \n $\mathrm{\# Vertex} \leq 50$'%(np.percentile(DFName[Target_Pt],0.3333*100), np.percentile(DFName[Target_Pt],0.6666*100))
    pTRangeString_high= '$20\ \mathrm{GeV} < |-\\vec{p}_T^Z| \leq 200\ \mathrm{GeV}$ \n $\mathrm{\# Vertex} \leq 50$'
    ptMin_low, ptMax_low = 0, np.percentile(DFName[Target_Pt], 33.3333)
    ptMin_mid, ptMax_mid = np.percentile(DFName[Target_Pt], 33.3333), np.percentile(DFName[Target_Pt], 66.6666)
    ptMin_high, ptMax_high = 20, 200


    pTRangeString_Tresh = '$1\ \mathrm{GeV} < |-\\vec{p}_T^Z| \leq 200\ \mathrm{GeV}$ \n $\mathrm{\# Vertex} \leq 50$'
    pTRangeStringNVertex = '$0\ \mathrm{GeV} < |-\\vec{p}_T^Z| \leq 200\ \mathrm{GeV}$ \n $\mathrm{\# Vertex} \leq 50$'
    if Target_Pt=='Boson_Pt':
        LegendTitle = '$\mathrm{Summer\ 17\ campaign}$' '\n'  '$\mathrm{Z \  \\rightarrow \ \mu \mu}$'
    else:
        LegendTitle = '$\mathrm{Summer\ 17\ campaign}$' '\n'  '$\mathrm{Z \  \\rightarrow \ \\tau \\tau   \\rightarrow \ \mu \mu}$'
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

    PF_Delta_pT, PF_Delta_Phi = kar2pol(DFName['recoilslimmedMETs_LongZ'],DFName['recoilslimmedMETs_PerpZ'])



    '''

    #NN_mode='kart'

    ################################MVA Output ################################
    nbinsVertex = 10
    #########Response u_para/pTZ ###########

    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    #plotMVAResponseOverpTZ_woutError('LongZCorrectedRecoil_LongZ', 'GBRT', 5, ScaleErr)
    plotMVAResponseOverpTZ_woutError('NN_LongZ', 'NN', 6, ScaleErr)
    plotMVAResponseOverpTZ_woutError('recoilslimmedMETs_LongZ', 'PF', 1, ScaleErr)
    #plotMVAResponseOverpTZ_woutError('ScaledNN_LongZ', 'NN scaled', 0, ScaleErr)
    plt.plot([0, 200], [1, 1], color='k', linestyle='--', linewidth=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.xlabel('$|-\\vec{p}_T^Z| $ in GeV')
    plt.ylabel('$\\langle \\frac{U_{\parallel}}{|-\\vec{p}_T^Z|} \\rangle$ ')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    plt.savefig("%sResponse_pT.png"%(plotsD), bbox_inches="tight")
    plt.close()


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    #plotMVANormOverpTZ('LongZCorrectedRecoil_Pt', 'GBRT', 5, ScaleErr)
    plotMVANormOverpTZ('NN_LongZ', 'NN', 6, ScaleErr)
    plotMVANormOverpTZ('recoilslimmedMETs_Pt', 'PF', 1, ScaleErr)
    plt.plot([0, 200], [1, 1], color='k', linestyle='--', linewidth=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.xlabel('$|-\\vec{p}_T^Z| $ in GeV')
    plt.ylabel('$\\langle \\frac{U}{|-\\vec{p}_T^Z|} \\rangle$ ')
    #plt.title('Response $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(ResponseMin, ResponseMax)
    plt.savefig("%sNorm_pT.png"%(plotsD), bbox_inches="tight")
    plt.close()


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    METpT_NN = DFName[Target_Pt].values[getResponseIdx('NN_LongZ')]
    #METpT_GBRT = DFName[Target_Pt].values[getResponseIdx('LongZCorrectedRecoil_LongZ')]
    METpT_PF = DFName[Target_Pt].values[getResponseIdx('recoilslimmedMETs_LongZ')]
    plt.hist(METpT_NN[np.abs(getResponse('NN_LongZ'))>10], bins=nbinsHist, range=[0, 200], label='NN', histtype='step', ec=colors_InOut[6], linewidth=1.5)
    #plt.hist(METpT_GBRT[np.abs(getResponse('LongZCorrectedRecoil_LongZ'))>10], bins=nbinsHist, range=[0, 200], label='GBRT', histtype='step', ec=colors_InOut[5], linewidth=1)
    plt.hist(METpT_PF[np.abs(getResponse('recoilslimmedMETs_LongZ'))>10], bins=nbinsHist, range=[0, 200], label='PF', histtype='step', ec=colors_InOut[1], linewidth=1)


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label='$ |Response|>10$'))

    plt.xlabel('$ |p_T^Z|$')
    plt.ylabel('Count')
    #plt.title('Response $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    plt.xlim(0, 200)
    plt.savefig("%sResponse_Ausreisser.png"%(plotsD), bbox_inches="tight")
    plt.close()





    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    plt.hist(getAngle('NN_Phi'), bins=nbinsHist, range=[-np.pi, np.pi], label='NN', histtype='step', ec=colors_InOut[0], linewidth=1.5)
    plt.hist(getAngle('recoilslimmedMETs_Phi'), bins=nbinsHist, range=[-np.pi, np.pi], label='GBRT', histtype='step', ec=colors_InOut[1], linewidth=1)
    #plt.hist(getAngle('LongZCorrectedRecoil_Phi'), bins=nbinsHist, range=[-np.pi, np.pi], label='PF', histtype='step', ec=colors_InOut[2], linewidth=1)
    PF_r, PF_phi = kar2pol(-DFName['recoilslimmedMETs_LongZ'], DFName['recoilslimmedMETs_PerpZ'])
    NN_r, NN_phi = kar2pol(-DFName['NN_LongZ'], DFName['NN_PerpZ'])
    #GBRT_r, GBRT_phi = kar2pol(-DFName['LongZCorrectedRecoil_LongZ'], DFName['LongZCorrectedRecoil_PerpZ'])
    plt.hist(NN_phi, bins=nbinsHist, range=[-np.pi, np.pi], label='CC NN', histtype='step', ec=colors_InOut[3], linewidth=1.5)
    plt.hist(PF_phi, bins=nbinsHist, range=[-np.pi, np.pi], label='CC GBRT', histtype='step', ec=colors_InOut[4], linewidth=1)
    #plt.hist(GBRT_phi, bins=nbinsHist, range=[-np.pi, np.pi], label='CC PF', histtype='step', ec=colors_InOut[5], linewidth=1)


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.xlabel('$ \Delta \\alpha$')
    plt.ylabel('Count')
    #plt.title('Response $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(ResponseMin, ResponseMax)
    plt.savefig("%sCrosscheck_Delta_Alpha.png"%(plotsD), bbox_inches="tight")
    plt.close()

    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    #plotMVANormOverpTZ_wErr('LongZCorrectedRecoil_Pt', 'GBRT', 5, ScaleErr)
    plotMVANormOverpTZ_wErr('NN_LongZ', 'NN', 6, ScaleErr)
    plotMVANormOverpTZ_wErr('recoilslimmedMETs_Pt', 'PF', 1, ScaleErr)
    plt.plot([0, 200], [1, 1], color='k', linestyle='--', linewidth=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString_Err))

    plt.xlabel('$|-\\vec{p}_T^Z| $ in GeV')
    plt.ylabel('$\\langle \\frac{U}{|-\\vec{p}_T^Z|} \\rangle$ ')
    #plt.title('Response $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(ResponseMin, ResponseMax)
    plt.savefig("%sNorm_pT_werr.png"%(plotsD), bbox_inches="tight")
    plt.close()


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    #plotMVAAngularOverpTZ('LongZCorrectedRecoil_Phi', 'GBRT', 5, ScaleErr)
    plotMVAAngularOverpTZ('NN_Phi', 'NN', 6, ScaleErr)
    plotMVAAngularOverpTZ('recoilslimmedMETs_Phi', 'PF', 1, ScaleErr)
    plt.plot([0, 200], [0, 0], color='k', linestyle='--', linewidth=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.xlabel('$|-\\vec{p}_T^Z| $ in GeV')
    plt.ylabel('$\\langle \Delta \\alpha \\rangle$ ')
    #plt.title('Response $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(ResponseMin, ResponseMax)
    plt.savefig("%sDelta_Alpha_pT.png"%(plotsD), bbox_inches="tight")
    plt.close()
    '''

    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    #plotMVAAngularOverpTZ_wErr('LongZCorrectedRecoil_Phi', 'GBRT', 5, ScaleErr)
    plotMVAAngularOverpTZ_wErr('NN_Phi', 'NN', 6, ScaleErr)
    plotMVAAngularOverpTZ_wErr('recoilslimmedMETs_Phi','PF', 1, ScaleErr)
    plt.plot([0, 200], [0, 0], color='k', linestyle='--', linewidth=1)


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString_Err))

    plt.xlabel('$|-\\vec{p}_T^Z| $ in GeV')
    plt.ylabel('$\\langle \Delta \\alpha \\rangle$ ')
    #plt.title('Response $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(ResponseMin, ResponseMax)
    plt.savefig("%sDelta_Alpha_pT_werr.png"%(plotsD), bbox_inches="tight")
    plt.close()


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    #Hist_Response('LongZCorrectedRecoil_LongZ', 'GBRT', 5, ScaleErr, 0, 200)
    Hist_Response('NN_LongZ', 'NN', 6, ScaleErr, 0, 200)
    Hist_Response('recoilslimmedMETs_LongZ', 'PF', 1, ScaleErr, 0, 200)


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.xlabel('$\\frac{U_{\parallel}}{p_T^Z}$')
    plt.ylabel('Counts')
    plt.xlim(ResponseMin, ResponseMax)
    #plt.ylim(0, 90000)
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

    #Hist_Response('LongZCorrectedRecoil_LongZ', 'GBRT', 5, ScaleErr, ptMin_low, ptMax_low)
    Hist_Response('NN_LongZ', 'NN', 6, ScaleErr, ptMin_low, ptMax_low)
    Hist_Response('recoilslimmedMETs_LongZ', 'PF', 1, ScaleErr, ptMin_low, ptMax_low)
    #Hist_Response('ScaledNN_LongZ', 'NN', 0, ScaleErr)
    #plt.plot([1, 1], [0, 90000], color='k', linestyle='--', linewidth=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString_low))

    plt.xlabel('$\\frac{U_{\parallel}}{p_T^Z}$')
    plt.ylabel('Counts')
    plt.xlim(ResponseMin, ResponseMax)
    #plt.ylim(0, 90000)
    #plt.title('Response $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    plt.xlim(ResponseMin, ResponseMax)
    plt.savefig("%sResponse_low.png"%(plotsD), bbox_inches="tight")
    plt.close()



    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    #Hist_Response('LongZCorrectedRecoil_LongZ', 'GBRT', 5, ScaleErr, ptMin_mid, ptMax_mid)
    Hist_Response('NN_LongZ', 'NN', 6, ScaleErr, ptMin_mid, ptMax_mid)
    Hist_Response('recoilslimmedMETs_LongZ', 'PF', 1, ScaleErr, ptMin_mid, ptMax_mid)
    #Hist_Response('ScaledNN_LongZ', 'NN', 0, ScaleErr)
    #plt.plot([1, 1], [0, 90000], color='k', linestyle='--', linewidth=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString_mid))

    plt.xlabel('$\\frac{U_{\parallel}}{p_T^Z}$')
    plt.ylabel('Counts')
    plt.xlim(ResponseMin, ResponseMax)
    #plt.ylim(0, 90000)
    #plt.title('Response $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    plt.xlim(ResponseMin, ResponseMax)
    plt.savefig("%sResponse_mid.png"%(plotsD), bbox_inches="tight")
    plt.close()


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    #Hist_Response('LongZCorrectedRecoil_LongZ', 'GBRT', 5, ScaleErr, ptMin_high, ptMax_high)
    Hist_Response('NN_LongZ', 'NN', 6, ScaleErr, ptMin_high, ptMax_high)
    Hist_Response('recoilslimmedMETs_LongZ', 'PF', 1, ScaleErr, ptMin_high, ptMax_high)
    #Hist_Response('ScaledNN_LongZ', 'NN', 0, ScaleErr)
    #plt.plot([1, 1], [0, 85000], color='k', linestyle='--', linewidth=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString_high))

    plt.xlabel('$\\frac{U_{\parallel}}{p_T^Z}$')
    plt.ylabel('Counts')
    plt.xlim(ResponseMin, ResponseMax)
    #plt.ylim(0, 85000)
    #plt.title('Response $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    plt.xlim(ResponseMin, ResponseMax)
    plt.savefig("%sResponse_high.png"%(plotsD), bbox_inches="tight")
    plt.close()


    '''







    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    #plotMVAResponseOverpTZ_wError('LongZCorrectedRecoil_LongZ', 'GBRT', 5, ScaleErr)
    plotMVAResponseOverpTZ_wError('NN_LongZ', 'NN', 6, ScaleErr)
    plotMVAResponseOverpTZ_wError('recoilslimmedMETs_LongZ', 'PF', 1, ScaleErr)
    plt.plot([0, 200], [1, 1], color='k', linestyle='--', linewidth=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.xlabel('$|-\\vec{p}_T^Z| $ in GeV')
    plt.ylabel('$\\langle \\frac{U_{\parallel}}{|-\\vec{p}_T^Z|} \\rangle$ ')
    #plt.title('Response $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(ResponseMinErr, ResponseMaxErr)
    plt.savefig("%sResponse_pT_wErr.png"%(plotsD), bbox_inches="tight")
    plt.close()


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)
    ScaleErr_Response_PV=1
    #plotMVAResponseOverNVertex_woutError('LongZCorrectedRecoil_LongZ', 'GBRT', 5, ScaleErr)
    plotMVAResponseOverNVertex_woutError('NN_LongZ', 'NN', 6, ScaleErr_Response_PV, 0, 200)
    #plotMVAResponseOverNVertex_woutError('ScaledNN_LongZ', 'NN scaled', 0, ScaleErr_Response_PV)
    plotMVAResponseOverNVertex_woutError('recoilslimmedMETs_LongZ', 'PF', 1, ScaleErr_Response_PV, 0, 200)
    plt.plot([0, 50], [1, 1], color='k', linestyle='--', linewidth=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    plt.xlabel('#$ \mathrm{PV}$ ')
    plt.ylabel('$\\langle \\frac{U_{\parallel}}{|-\\vec{p}_T^Z|} \\rangle$ ')
    #plt.title('Response $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(0, ResponseMax)
    plt.savefig("%sResponse_PV.png"%(plotsD), bbox_inches="tight")
    plt.close()


    '''
    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)
    ScaleErr_Response_PV=1
    #plotMVAResponseOverNVertex_woutError('LongZCorrectedRecoil_LongZ', 'GBRT', 5, ScaleErr)
    plotMVAResponseOverNVertex_woutError('NN_LongZ', 'NN', 6, ScaleErr_Response_PV, ptMin_low, ptMax_low)
    #plotMVAResponseOverNVertex_woutError('ScaledNN_LongZ', 'NN scaled', 0, ScaleErr_Response_PV)
    plotMVAResponseOverNVertex_woutError('recoilslimmedMETs_LongZ', 'PF', 1, ScaleErr_Response_PV, ptMin_low, ptMax_low)
    plt.plot([0, 50], [1, 1], color='k', linestyle='--', linewidth=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString_low))

    plt.xlabel('#$ \mathrm{PV}$ ')
    plt.ylabel('$\\langle \\frac{U_{\parallel}}{|-\\vec{p}_T^Z|} \\rangle$ ')
    #plt.title('Response $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(0, ResponseMax)
    plt.savefig("%sResponse_PV_low.png"%(plotsD), bbox_inches="tight")
    plt.close()


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    #plotMVAResponseOverNVertex_woutError('LongZCorrectedRecoil_LongZ', 'GBRT', 5, ScaleErr)
    plotMVAResponseOverNVertex_woutError('NN_LongZ', 'NN', 6, ScaleErr_Response_PV, ptMin_mid, ptMax_mid)
    #plotMVAResponseOverNVertex_woutError('ScaledNN_LongZ', 'NN scaled', 0, ScaleErr_Response_PV)
    plotMVAResponseOverNVertex_woutError('recoilslimmedMETs_LongZ', 'PF', 1, ScaleErr_Response_PV, ptMin_mid, ptMax_mid)
    plt.plot([0, 50], [1, 1], color='k', linestyle='--', linewidth=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString_mid))

    plt.xlabel('#$ \mathrm{PV}$ ')
    plt.ylabel('$\\langle \\frac{U_{\parallel}}{|-\\vec{p}_T^Z|} \\rangle$ ')
    #plt.title('Response $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(0, ResponseMax)
    plt.savefig("%sResponse_PV_mid.png"%(plotsD), bbox_inches="tight")
    plt.close()
    '''

    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    #plotMVAResponseOverNVertex_woutError('LongZCorrectedRecoil_LongZ', 'GBRT', 5, ScaleErr)
    plotMVAResponseOverNVertex_woutError('NN_LongZ', 'NN', 6, ScaleErr_Response_PV, ptMin_high, ptMax_high)
    #plotMVAResponseOverNVertex_woutError('ScaledNN_LongZ', 'NN scaled', 0, ScaleErr_Response_PV)
    plotMVAResponseOverNVertex_woutError('recoilslimmedMETs_LongZ', 'PF', 1, ScaleErr_Response_PV, ptMin_high, ptMax_high)
    plt.plot([0, 50], [1, 1], color='k', linestyle='--', linewidth=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString_high))

    plt.xlabel('#$ \mathrm{PV}$ ')
    plt.ylabel('$\\langle \\frac{U_{\parallel}}{|-\\vec{p}_T^Z|} \\rangle$ ')
    #plt.title('Response $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(0, ResponseMax)
    plt.savefig("%sResponse_PV_high.png"%(plotsD), bbox_inches="tight")
    plt.close()


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    #plotMVAResponseOverNVertex_woutError_Tresh('LongZCorrectedRecoil_LongZ', 'GBRT', 5, ScaleErr)
    plotMVAResponseOverNVertex_woutError_Tresh('NN_LongZ', 'NN', 6, ScaleErr)
    plotMVAResponseOverNVertex_woutError_Tresh('recoilslimmedMETs_LongZ', 'PF', 1, ScaleErr)
    plt.plot([0, 50], [1, 1], color='k', linestyle='--', linewidth=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString_Tresh))

    plt.xlabel('#$ \mathrm{PV}$ ')
    plt.ylabel('$\\langle \\frac{U_{\parallel}}{|-\\vec{p}_T^Z|} \\rangle$ ')
    #plt.title('Response $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(0, ResponseMax)
    plt.savefig("%sResponse_PV_Tresh.png"%(plotsD), bbox_inches="tight")
    plt.close()



    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    #plotMVAResponseOverNVertex_wError('LongZCorrectedRecoil_LongZ', 'GBRT', 5, ScaleErr)
    plotMVAResponseOverNVertex_wError('NN_LongZ', 'NN', 6, ScaleErr)
    plotMVAResponseOverNVertex_wError('recoilslimmedMETs_LongZ', 'PF', 1, ScaleErr)
    plt.plot([0, 50], [1, 1], color='k', linestyle='--', linewidth=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    plt.xlabel('#$ \mathrm{PV}$ ')
    plt.ylabel('$\\langle \\frac{U_{\parallel}}{|-\\vec{p}_T^Z|} \\rangle$ ')
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

    #plotMVAResolutionOverpTZ_woutError_para('LongZCorrectedRecoil_LongZ', 'GBRT', 5, ScaleErr)
    plotMVAResolutionOverpTZ_woutError_para('NN_LongZ', 'NN', 6, ScaleErr)
    #plotMVAResolutionOverpTZ_woutError_para('ScaledNN_LongZ', 'NN scaled', 0, ScaleErr)
    plotMVAResolutionOverpTZ_woutError_para('recoilslimmedMETs_LongZ', 'PF', 1, ScaleErr)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.xlabel('$|-\\vec{p}_T^Z| $ in GeV')
    plt.ylabel('$\sigma \\left( U_{\parallel}- |-\\vec{p}_T^Z| \\right) $ in GeV')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\parallel}}{|-\\vec{p}_T^Z|} \\right) $ in GeV')
    #plt.title('Resolution $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(ylimResMVAMin, ylimResMax)
    plt.savefig("%sResolution_para_pT.png"%(plotsD), bbox_inches="tight")
    plt.close()


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)
    ScaleErr_pTPV=0.1
    pT_PVbins(Target_Pt, 'Target MET', 4, ScaleErr_pTPV, 0, 200)
    plt.plot([0, 50], [np.mean(DFName_nVertex[Target_Pt]), np.mean(DFName_nVertex[Target_Pt])], color='k', linestyle='--', linewidth=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.xlabel('#$ \mathrm{PV}$ ')
    plt.ylabel('$\\langle |-\\vec{p}_T^Z| \\rangle$ in GeV ')
    #plt.title('Response $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    plt.ylim(10, 30)
    plt.xlim(0, 50)
    plt.savefig("%spT_over_PV.png"%(plotsD), bbox_inches="tight")
    plt.close()

    '''
    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    pT_PVbins(Target_Pt, 'Target MET', 4, ScaleErr_pTPV, ptMin_low, ptMax_low)
    plt.plot([0, 50], [np.mean(DFName_nVertex[Target_Pt]), np.mean(DFName_nVertex[Target_Pt])], color='k', linestyle='--', linewidth=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString_low))

    plt.xlabel('#$ \mathrm{PV}$ ')
    plt.ylabel('$\\langle |-\\vec{p}_T^Z| \\rangle$ in GeV ')
    #plt.title('Response $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    plt.ylim(10, 30)
    plt.xlim(0, 50)
    plt.savefig("%spT_over_PV_low.png"%(plotsD), bbox_inches="tight")
    plt.close()


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    pT_PVbins(Target_Pt, 'Target MET', 4, ScaleErr_pTPV, ptMin_mid, ptMax_mid)
    plt.plot([0, 50], [np.mean(DFName_nVertex[Target_Pt]), np.mean(DFName_nVertex[Target_Pt])], color='k', linestyle='--', linewidth=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString_mid))

    plt.xlabel('#$ \mathrm{PV}$ ')
    plt.ylabel('$\\langle |-\\vec{p}_T^Z| \\rangle$ in GeV ')
    #plt.title('Response $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    plt.ylim(10, 30)
    plt.xlim(0, 50)
    plt.savefig("%spT_over_PV_mid.png"%(plotsD), bbox_inches="tight")
    plt.close()
    '''

    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    pT_PVbins(Target_Pt, 'Target MET', 4, ScaleErr_pTPV, ptMin_high, ptMax_high)
    plt.plot([0, 50], [np.mean(DFName_nVertex[Target_Pt]), np.mean(DFName_nVertex[Target_Pt])], color='k', linestyle='--', linewidth=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString_high))

    plt.xlabel('#$ \mathrm{PV}$ ')
    plt.ylabel('$\\langle |-\\vec{p}_T^Z| \\rangle$ in GeV ')
    #plt.title('Response $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    plt.ylim(10, 30)
    plt.xlim(0, 50)
    plt.savefig("%spT_over_PV_high.png"%(plotsD), bbox_inches="tight")
    plt.close()


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    #Hist_Resolution_para('LongZCorrectedRecoil_LongZ', 'GBRT', 5, ScaleErr, 0, 200)
    Hist_Resolution_para('NN_LongZ', 'NN', 6, ScaleErr, 0, 200)
    Hist_Resolution_para('recoilslimmedMETsPuppi_LongZ', 'Puppi', 4, ScaleErr, 0, 200)
    Hist_Resolution_para('recoilslimmedMETs_LongZ', 'PF', 1, ScaleErr, 0, 200)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.xlabel('$U_{\parallel}-p_T^Z$')
    plt.ylabel('Counts')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\parallel}}{|-\\vec{p}_T^Z|} \\right) $ in GeV')
    #plt.title('Resolution $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    plt.xlim(ResolutionParaMin, ResolutionParaMax )
    plt.savefig("%sHist_Resolution_para.png"%(plotsD), bbox_inches="tight")
    plt.close()

    '''
    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    #Hist_Resolution_para('LongZCorrectedRecoil_LongZ', 'GBRT', 5, ScaleErr, ptMin_low, ptMax_low)
    Hist_Resolution_para('NN_LongZ', 'NN', 6, ScaleErr, ptMin_low, ptMax_low)
    #Hist_Resolution_para('ScaledNN_LongZ', 'NN scaled', 0, ScaleErr)
    Hist_Resolution_para('recoilslimmedMETs_LongZ', 'PF', 1, ScaleErr, ptMin_low, ptMax_low)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString_low))

    plt.xlabel('$U_{\parallel}-p_T^Z$')
    plt.ylabel('Counts')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\parallel}}{|-\\vec{p}_T^Z|} \\right) $ in GeV')
    #plt.title('Resolution $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    plt.xlim(ResolutionParaMin, ResolutionParaMax )
    plt.savefig("%sHist_Resolution_para_low.png"%(plotsD), bbox_inches="tight")
    plt.close()
    '''

    '''
    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    #Hist_Resolution_para('LongZCorrectedRecoil_LongZ', 'GBRT', 5, ScaleErr, ptMin_mid, ptMax_mid)
    Hist_Resolution_para('NN_LongZ', 'NN', 6, ScaleErr, ptMin_mid, ptMax_mid)
    #Hist_Resolution_para('ScaledNN_LongZ', 'NN scaled', 0, ScaleErr)
    Hist_Resolution_para('recoilslimmedMETs_LongZ', 'PF', 1, ScaleErr, ptMin_mid, ptMax_mid)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString_mid))

    plt.xlabel('$U_{\parallel}-p_T^Z$')
    plt.ylabel('Counts')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\parallel}}{|-\\vec{p}_T^Z|} \\right) $ in GeV')
    #plt.title('Resolution $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    plt.xlim(ResolutionParaMin, ResolutionParaMax )
    plt.savefig("%sHist_Resolution_para_mid.png"%(plotsD), bbox_inches="tight")
    plt.close()
    '''



    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    #Hist_Resolution_para('LongZCorrectedRecoil_LongZ', 'GBRT', 5, ScaleErr, ptMin_high, ptMax_high)
    Hist_Resolution_para('NN_LongZ', 'NN', 6, ScaleErr, ptMin_high, ptMax_high)
    #Hist_Resolution_para('ScaledNN_LongZ', 'NN scaled', 0, ScaleErr)
    Hist_Resolution_para('recoilslimmedMETs_LongZ', 'PF', 1, ScaleErr, ptMin_high, ptMax_high)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString_high))

    plt.xlabel('$U_{\parallel}-p_T^Z$')
    plt.ylabel('Counts')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\parallel}}{|-\\vec{p}_T^Z|} \\right) $ in GeV')
    #plt.title('Resolution $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    plt.xlim(ResolutionParaMin, ResolutionParaMax )
    plt.savefig("%sHist_Resolution_para_high.png"%(plotsD), bbox_inches="tight")
    plt.close()


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)
    pTRangeString_0_100 = '$0\ \mathrm{GeV} < |-\\vec{p}_T^Z| \leq 100\ \mathrm{GeV}$ \n $\mathrm{\# Vertex} \leq 50$'
    #Hist_Resolution_para_0_100('LongZCorrectedRecoil_LongZ', 'GBRT', 5, ScaleErr)
    Hist_Resolution_para_0_100('NN_LongZ', 'NN', 6, ScaleErr)
    #Hist_Resolution_para_0_100('ScaledNN_LongZ', 'NN scaled', 0, ScaleErr)
    Hist_Resolution_para_0_100('recoilslimmedMETs_LongZ', 'PF', 1, ScaleErr)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString_0_100))

    plt.xlabel('$U_{\parallel}-p_T^Z$')
    plt.ylabel('Counts')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\parallel}}{|-\\vec{p}_T^Z|} \\right) $ in GeV')
    #plt.title('Resolution $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    plt.xlim(ResolutionParaMin, ResolutionParaMax )
    plt.savefig("%sHist_Resolution_para_0-100.png"%(plotsD), bbox_inches="tight")
    plt.close()

    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    #Hist_Resolution_para_100_150('LongZCorrectedRecoil_LongZ', 'GBRT', 5, ScaleErr)
    Hist_Resolution_para_100_150('NN_LongZ', 'NN', 6, ScaleErr)
    Hist_Resolution_para_100_150('recoilslimmedMETs_LongZ', 'PF', 1, ScaleErr)
    pTRangeString_100_150 = '$100\ \mathrm{GeV} < |-\\vec{p}_T^Z| \leq 150\ \mathrm{GeV}$ \n $\mathrm{\# Vertex} \leq 50$'
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString_100_150))

    plt.xlabel('$U_{\parallel}-p_T^Z$')
    plt.ylabel('Counts')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\parallel}}{|-\\vec{p}_T^Z|} \\right) $ in GeV')
    #plt.title('Resolution $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    plt.xlim(ResolutionParaMin, ResolutionParaMax )
    plt.savefig("%sHist_Resolution_para_100-150.png"%(plotsD), bbox_inches="tight")
    plt.close()

    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)
    pTRangeString_150_200 = '$150\ \mathrm{GeV} < |-\\vec{p}_T^Z| \leq 200\ \mathrm{GeV}$ \n $\mathrm{\# Vertex} \leq 50$'
    #Hist_Resolution_para_150_200('LongZCorrectedRecoil_LongZ', 'GBRT', 5, ScaleErr)
    Hist_Resolution_para_150_200('NN_LongZ', 'NN', 6, ScaleErr)
    Hist_Resolution_para_150_200('recoilslimmedMETs_LongZ', 'PF', 1, ScaleErr)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString_150_200))

    plt.xlabel('$U_{\parallel}-p_T^Z$')
    plt.ylabel('Counts')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\parallel}}{|-\\vec{p}_T^Z|} \\right) $ in GeV')
    #plt.title('Resolution $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    plt.xlim(ResolutionParaMin, ResolutionParaMax )
    plt.savefig("%sHist_Resolution_para_150-200.png"%(plotsD), bbox_inches="tight")
    plt.close()

    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    Hist_Resolution_perp('recoilslimmedMETsPuppi_PerpZ', 'Puppi', 4, ScaleErr, 0, 200)
    Hist_Resolution_perp('NN_PerpZ', 'NN', 6, ScaleErr, 0, 200)
    Hist_Resolution_perp('recoilslimmedMETs_PerpZ', 'PF', 1, ScaleErr, 0, 200)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.xlabel('$U_{\perp}$ in GeV')
    plt.ylabel('Counts')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\parallel}}{|-\\vec{p}_T^Z|} \\right) $ in GeV')
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

    Hist_Resolution_perp('recoilslimmedMETsPuppi_PerpZ', 'Puppi', 4, ScaleErr, ptMin_low, ptMax_low)
    Hist_Resolution_perp('NN_PerpZ', 'NN', 6, ScaleErr, ptMin_low, ptMax_low)
    Hist_Resolution_perp('recoilslimmedMETs_PerpZ', 'PF', 1, ScaleErr, ptMin_low, ptMax_low)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString_low))

    plt.xlabel('$U_{\perp}$ in GeV')
    plt.ylabel('Counts')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\parallel}}{|-\\vec{p}_T^Z|} \\right) $ in GeV')
    #plt.title('Resolution $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    plt.xlim(ResolutionPerpMin, ResolutionPerpMax )
    plt.savefig("%sHist_Resolution_perp_low.png"%(plotsD), bbox_inches="tight")
    plt.close()
    '''

    '''
    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    Hist_Resolution_perp('recoilslimmedMETsPuppi_PerpZ', 'Puppi', 4, ScaleErr, ptMin_mid, ptMax_mid)
    Hist_Resolution_perp('NN_PerpZ', 'NN', 6, ScaleErr, ptMin_mid, ptMax_mid)
    Hist_Resolution_perp('recoilslimmedMETs_PerpZ', 'PF', 1, ScaleErr, ptMin_mid, ptMax_mid)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString_mid))

    plt.xlabel('$U_{\perp}$ in GeV')
    plt.ylabel('Counts')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\parallel}}{|-\\vec{p}_T^Z|} \\right) $ in GeV')
    #plt.title('Resolution $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    plt.xlim(ResolutionPerpMin, ResolutionPerpMax )
    plt.savefig("%sHist_Resolution_perp_mid.png"%(plotsD), bbox_inches="tight")
    plt.close()
    '''

    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    Hist_Resolution_perp('recoilslimmedMETsPuppi_PerpZ', 'Puppi', 4, ScaleErr, ptMin_high, ptMax_high)
    Hist_Resolution_perp('NN_PerpZ', 'NN', 6, ScaleErr, ptMin_high, ptMax_high)
    Hist_Resolution_perp('recoilslimmedMETs_PerpZ', 'PF', 1, ScaleErr, ptMin_high, ptMax_high)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString_high))

    plt.xlabel('$U_{\perp}$ in GeV')
    plt.ylabel('Counts')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\parallel}}{|-\\vec{p}_T^Z|} \\right) $ in GeV')
    #plt.title('Resolution $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    plt.xlim(ResolutionPerpMin, ResolutionPerpMax )
    plt.savefig("%sHist_Resolution_perp_high.png"%(plotsD), bbox_inches="tight")
    plt.close()




    '''
    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    #MeanDeviation_Pt('LongZCorrectedRecoil_LongZ', 'GBRT', 5, ScaleErr)
    MeanDeviation_Pt('NN_LongZ', 'NN', 6, ScaleErr)
    MeanDeviation_Pt('recoilslimmedMETs_LongZ', 'PF', 1, ScaleErr)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.xlabel('$|-\\vec{p}_T^Z| $ in GeV')
    plt.ylabel('$\\langle U_{\parallel}- |-\\vec{p}_T^Z| \\rangle$ in GeV')
    #plt.title('Mean Deviation $U_{\parallel}-|-\\vec{p}_T^Z|$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(-20, 10)
    plt.savefig("%sDelta_para_pT.png"%(plotsD), bbox_inches="tight")
    plt.close()

    '''
    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    plotMVAResolutionOverNVertex_woutError_para('recoilslimmedMETsPuppi_LongZ', 'Puppi', 4, ScaleErr, 0, 200)
    plotMVAResolutionOverNVertex_woutError_para('NN_LongZ', 'NN', 6, ScaleErr, 0, 200)
    plotMVAResolutionOverNVertex_woutError_para('recoilslimmedMETs_LongZ', 'PF', 1, ScaleErr, 0, 200)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    plt.xlabel('#$ \mathrm{PV}$ ')
    plt.ylabel('$\sigma \\left( U_{\parallel}- |-\\vec{p}_T^Z| \\right) $ in GeV')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\parallel}}{|-\\vec{p}_T^Z|} \\right) $ in GeV')
    #plt.title('Resolution $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(ylimResMVAMin, ylimResMax)
    plt.savefig("%sResolution_para_PV.png"%(plotsD), bbox_inches="tight")
    plt.close()
    '''
    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    plotMVAResolutionOverNVertex_woutError_para('recoilslimmedMETsPuppi_LongZ', 'Puppi', 4, ScaleErr)
    plotMVAResolutionOverNVertex_woutError_para('NN_LongZ', 'NN', 6, ScaleErr, ptMin_low, ptMax_low)
    plotMVAResolutionOverNVertex_woutError_para('recoilslimmedMETs_LongZ', 'PF', 1, ScaleErr, ptMin_low, ptMax_low)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString_low))

    plt.xlabel('#$ \mathrm{PV}$ ')
    plt.ylabel('$\sigma \\left( U_{\parallel}- |-\\vec{p}_T^Z| \\right) $ in GeV')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\parallel}}{|-\\vec{p}_T^Z|} \\right) $ in GeV')
    #plt.title('Resolution $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(ylimResMVAMin, ylimResMax)
    plt.savefig("%sResolution_para_PV_low.png"%(plotsD), bbox_inches="tight")
    plt.close()

    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    plotMVAResolutionOverNVertex_woutError_para('recoilslimmedMETsPuppi_LongZ', 'Puppi', 4,  ScaleErr, ptMin_mid, ptMax_mid)
    plotMVAResolutionOverNVertex_woutError_para('NN_LongZ', 'NN', 6, ScaleErr, ptMin_mid, ptMax_mid)
    plotMVAResolutionOverNVertex_woutError_para('recoilslimmedMETs_LongZ', 'PF', 1, ScaleErr, ptMin_mid, ptMax_mid)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString_mid))

    plt.xlabel('#$ \mathrm{PV}$ ')
    plt.ylabel('$\sigma \\left( U_{\parallel}- |-\\vec{p}_T^Z| \\right) $ in GeV')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\parallel}}{|-\\vec{p}_T^Z|} \\right) $ in GeV')
    #plt.title('Resolution $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(ylimResMVAMin, ylimResMax)
    plt.savefig("%sResolution_para_PV_mid.png"%(plotsD), bbox_inches="tight")
    plt.close()
    '''

    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    plotMVAResolutionOverNVertex_woutError_para('recoilslimmedMETsPuppi_LongZ', 'Puppi', 5, ScaleErr, ptMin_high, ptMax_high)
    plotMVAResolutionOverNVertex_woutError_para('NN_LongZ', 'NN', 6, ScaleErr, ptMin_high, ptMax_high)
    plotMVAResolutionOverNVertex_woutError_para('recoilslimmedMETs_LongZ', 'PF', 1, ScaleErr, ptMin_high, ptMax_high)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString_high))

    plt.xlabel('#$ \mathrm{PV}$ ')
    plt.ylabel('$\sigma \\left( U_{\parallel}- |-\\vec{p}_T^Z| \\right) $ in GeV')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\parallel}}{|-\\vec{p}_T^Z|} \\right) $ in GeV')
    #plt.title('Resolution $U_{\parallel}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(ylimResMVAMin, ylimResMax)
    plt.savefig("%sResolution_para_PV_high.png"%(plotsD), bbox_inches="tight")
    plt.close()


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    plotMVAResolutionOverNVertex_woutError_perp('recoilslimmedMETsPuppi_PerpZ', 'Puppi', 4, ScaleErr, 0, 200)
    plotMVAResolutionOverNVertex_woutError_perp('NN_PerpZ', 'NN', 6, ScaleErr, 0, 200)
    plotMVAResolutionOverNVertex_woutError_perp('recoilslimmedMETs_PerpZ', 'PF', 1, ScaleErr, 0, 200)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    plt.xlabel('#$ \mathrm{PV}$ ')
    plt.ylabel('$\sigma \\left( U_{\perp}- |-\\vec{p}_T^Z| \\right) $ in GeV')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{|-\\vec{p}_T^Z|} \\right) $ in GeV')
    #plt.title('Resolution $U_{\perp}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(ylimResMVAMin, ylimResMax)
    plt.savefig("%sResolution_perp_PV.png"%(plotsD), bbox_inches="tight")
    plt.close()
    '''
    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    #plotMVAResolutionOverNVertex_woutError_perp('LongZCorrectedRecoil_LongZ', 'GBRT', 5, ScaleErr)
    plotMVAResolutionOverNVertex_woutError_perp('NN_LongZ', 'NN', 6, ScaleErr, ptMin_low, ptMax_low)
    plotMVAResolutionOverNVertex_woutError_perp('recoilslimmedMETs_LongZ', 'PF', 1, ScaleErr, ptMin_low, ptMax_low)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString_low))

    plt.xlabel('#$ \mathrm{PV}$ ')
    plt.ylabel('$\sigma \\left( U_{\perp}- |-\\vec{p}_T^Z| \\right) $ in GeV')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{|-\\vec{p}_T^Z|} \\right) $ in GeV')
    #plt.title('Resolution $U_{\perp}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(ylimResMVAMin, ylimResMax)
    plt.savefig("%sResolution_perp_PV_low.png"%(plotsD), bbox_inches="tight")
    plt.close()

    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    plotMVAResolutionOverNVertex_woutError_perp('recoilslimmedMETsPuppi_PerpZ', 'Puppi', 4, ScaleErr, ptMin_mid, ptMax_mid)
    plotMVAResolutionOverNVertex_woutError_perp('NN_PerpZ', 'NN', 6, ScaleErr, ptMin_mid, ptMax_mid)
    plotMVAResolutionOverNVertex_woutError_perp('recoilslimmedMETs_PerpZ', 'PF', 1, ScaleErr, ptMin_mid, ptMax_mid)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString_mid))

    plt.xlabel('#$ \mathrm{PV}$ ')
    plt.ylabel('$\sigma \\left( U_{\perp}- |-\\vec{p}_T^Z| \\right) $ in GeV')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{|-\\vec{p}_T^Z|} \\right) $ in GeV')
    #plt.title('Resolution $U_{\perp}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(ylimResMVAMin, ylimResMax)
    plt.savefig("%sResolution_perp_PV_mid.png"%(plotsD), bbox_inches="tight")
    plt.close()
    '''
    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    plotMVAResolutionOverNVertex_woutError_perp('recoilslimmedMETsPuppi_PerpZ', 'Puppi', 4, ScaleErr, ptMin_high, ptMax_high)
    plotMVAResolutionOverNVertex_woutError_perp('NN_PerpZ', 'NN', 6, ScaleErr, ptMin_high, ptMax_high)
    plotMVAResolutionOverNVertex_woutError_perp('recoilslimmedMETs_PerpZ', 'PF', 1, ScaleErr, ptMin_high, ptMax_high)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString_high))

    plt.xlabel('#$ \mathrm{PV}$ ')
    plt.ylabel('$\sigma \\left( U_{\perp}- |-\\vec{p}_T^Z| \\right) $ in GeV')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{|-\\vec{p}_T^Z|} \\right) $ in GeV')
    #plt.title('Resolution $U_{\perp}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(ylimResMVAMin, ylimResMax)
    plt.savefig("%sResolution_perp_PV_high.png"%(plotsD), bbox_inches="tight")
    plt.close()


    '''
    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    #MeanDeviation_PV('LongZCorrectedRecoil_LongZ', 'GBRT', 5, ScaleErr)
    MeanDeviation_PV('NN_LongZ', 'NN', 6, ScaleErr)
    MeanDeviation_PV('recoilslimmedMETs_LongZ', 'PF', 1, ScaleErr)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    plt.xlabel('#$ \mathrm{PV}$ ')
    plt.ylabel('$\\langle U_{\parallel} - p_T^Z \\rangle$ in GeV')
    #plt.title('Mean Deviation $U_{\parallel}-|-\\vec{p}_T^Z|$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(-20, 20)
    plt.savefig("%sDelta_para_PV.png"%(plotsD), bbox_inches="tight")
    plt.close()
    '''

    #######u perp ######
    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    plotMVAResolutionOverpTZ_woutError_perp('recoilslimmedMETsPuppi_PerpZ', 'Puppi', 4, ScaleErr)
    plotMVAResolutionOverpTZ_woutError_perp('NN_PerpZ', 'NN', 6, ScaleErr)
    plotMVAResolutionOverpTZ_woutError_perp('recoilslimmedMETs_PerpZ', 'PF', 1, ScaleErr)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.xlabel('$|-\\vec{p}_T^Z| $ in GeV')
    plt.ylabel('$\sigma \\left( U_{\perp} \\right) $ in GeV')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{|-\\vec{p}_T^Z|} \\right) $ in GeV')
    #plt.title('Resolution $U_{\perp}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(ylimResMVAMin, ylimResMax)
    plt.savefig("%sResolution_perp_pT.png"%(plotsD), bbox_inches="tight")
    plt.close()






    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    plotMVAResolutionOverNVertex_woutError_perp('recoilslimmedMETsPuppi_PerpZ', 'Puppi', 4, ScaleErr, 0, 200)
    plotMVAResolutionOverNVertex_woutError_perp('NN_PerpZ', 'NN', 6, ScaleErr, 0, 200)
    plotMVAResolutionOverNVertex_woutError_perp('recoilslimmedMETs_PerpZ', 'PF', 1, ScaleErr, 0, 200)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    plt.xlabel('#$ \mathrm{PV}$ ')
    plt.ylabel('$\sigma \\left( U_{\perp} \\right) $ in GeV')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{|-\\vec{p}_T^Z|} \\right) $ in GeV')
    #plt.title('Resolution $U_{\perp}$')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(ylimResMVAMin, ylimResMax)
    plt.savefig("%sResolution_perp_PV.png"%(plotsD), bbox_inches="tight")
    plt.close()












    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)


    Histogram_Deviation_para_PV('recoilpatpfNoPUMET_LongZ', 'No PU MET', 0, ScaleErr)
    Histogram_Deviation_para_PV('recoilpatpfPUCorrectedMET_LongZ', 'PU corrected MET', 7, ScaleErr)
    Histogram_Deviation_para_PV( 'recoilpatpfPUMET_LongZ', 'PU MET', 2, ScaleErr)
    Histogram_Deviation_para_PV('recoilpatpfTrackMET_LongZ', 'Track MET', 3, ScaleErr)
    Histogram_Deviation_para_PV('recoilslimmedMETsPuppi_LongZ', 'Puppi MET', 4, ScaleErr)
    #Histogram_Deviation_para_PV('LongZCorrectedRecoil_LongZ', 'GBRT MET', 5, ScaleErr)
    Histogram_Deviation_para_PV('recoilslimmedMETs_LongZ', 'PF MET', 1, ScaleErr)
    Histogram_Deviation_para_PV('NN_LongZ', 'NN MET', 6, ScaleErr)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.ylabel('Counts')
    plt.xlabel('$ U_{\parallel} - p_T^Z  $ in GeV')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{|-\\vec{p}_T^Z|} \\right) $ in GeV')
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


    Histogram_Norm_Comparison('recoilpatpfNoPUMET_Pt', 'No PU MET', 0, ScaleErr)
    Histogram_Norm_Comparison('recoilpatpfPUCorrectedMET_Pt', 'PU corrected MET', 7, ScaleErr)
    Histogram_Norm_Comparison( 'recoilpatpfPUMET_Pt',  'PU MET', 2, ScaleErr)
    Histogram_Norm_Comparison('recoilpatpfTrackMET_Pt',  'Track MET', 3, ScaleErr)
    Histogram_Norm_Comparison('recoilslimmedMETsPuppi_Pt',  'Puppi MET', 4, ScaleErr)
    #Histogram_Norm_Comparison('LongZCorrectedRecoil_Pt',  'GBRT MET', 5, ScaleErr)
    Histogram_Norm_Comparison('recoilslimmedMETs_Pt',  'PF MET', 1, ScaleErr)
    Histogram_Norm_Comparison('NN_Pt', 'NN MET', 6, ScaleErr)


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.ylabel('Counts')
    plt.xlabel('$ |\\vec{U}| - |-\\vec{p}_T^Z|  $ in GeV')
    plt.xlim(HistLimMin,HistLimMax)
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{|-\\vec{p}_T^Z|} \\right) $ in GeV')
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


    Histogram_Norm('recoilpatpfNoPUMET_Pt', 'No PU MET', 0, ScaleErr)
    Histogram_Norm('recoilpatpfPUCorrectedMET_Pt', 'PU corrected MET', 7, ScaleErr)
    Histogram_Norm( 'recoilpatpfPUMET_Pt',  'PU MET', 2, ScaleErr)
    Histogram_Norm('recoilpatpfTrackMET_Pt',  'Track MET', 3, ScaleErr)
    Histogram_Norm('recoilslimmedMETsPuppi_Pt',  'Puppi MET', 4, ScaleErr)
    #Histogram_Norm('LongZCorrectedRecoil_Pt',  'GBRT MET', 5, ScaleErr)
    Histogram_Norm('recoilslimmedMETs_Pt',  'PF MET', 1, ScaleErr)
    Histogram_Norm('NN_Pt', 'NN MET', 6, ScaleErr)




    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.ylabel('Counts')
    plt.xlabel('$ |U|   $ in GeV')
    plt.xlim(0,75)
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{|-\\vec{p}_T^Z|} \\right) $ in GeV')
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




    Histogram_Deviation_perp_PV('recoilpatpfNoPUMET_PerpZ', 'No PU MET', 0, ScaleErr)
    Histogram_Deviation_perp_PV('recoilpatpfPUCorrectedMET_PerpZ', 'PU corrected MET', 7, ScaleErr)
    Histogram_Deviation_perp_PV( 'recoilpatpfPUMET_PerpZ', 'PU MET', 2, ScaleErr)
    Histogram_Deviation_perp_PV('recoilpatpfTrackMET_PerpZ', 'Track MET', 3, ScaleErr)
    Histogram_Deviation_perp_PV('recoilslimmedMETsPuppi_PerpZ', 'Puppi MET', 4, ScaleErr)
    #Histogram_Deviation_perp_PV('LongZCorrectedRecoil_PerpZ', 'GBRT', 5, ScaleErr)
    Histogram_Deviation_perp_PV('recoilslimmedMETs_PerpZ', 'PF', 1, ScaleErr)
    Histogram_Deviation_perp_PV('NN_PerpZ', 'NN', 6, ScaleErr)


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.ylabel('Counts')
    plt.xlabel('$ U_{\perp}  $ in GeV')
    plt.xlim(HistLimMin,HistLimMax)
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{|-\\vec{p}_T^Z|} \\right) $ in GeV')
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


    Hist_LongZ('recoilpatpfNoPUMET_LongZ', 'No PU MET', 0, ScaleErr)
    Hist_LongZ('recoilpatpfPUCorrectedMET_LongZ', 'PU corrected MET', 7, ScaleErr)
    Hist_LongZ( 'recoilpatpfPUMET_LongZ', 'PU MET', 2, ScaleErr)
    Hist_LongZ('recoilpatpfTrackMET_LongZ', 'Track MET', 3, ScaleErr)
    Hist_LongZ('recoilslimmedMETsPuppi_LongZ', 'Puppi MET',7, ScaleErr)
    #Hist_LongZ('LongZCorrectedRecoil_LongZ', 'GBRT MET', 5, ScaleErr)
    Hist_LongZ('recoilslimmedMETs_LongZ', 'PF MET', 1, ScaleErr)
    Hist_LongZ(Target_Pt, 'Target MET', 4, ScaleErr)
    Hist_LongZ('NN_LongZ', 'NN MET', 6, ScaleErr)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.ylabel('Counts')
    plt.xlabel('$ U_{\parallel}   $ in GeV')
    plt.xlim(HistLimMin,HistLimMax)
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{|-\\vec{p}_T^Z|} \\right) $ in GeV')
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


    Hist_PerpZ('recoilpatpfNoPUMET_PerpZ', 'No PU MET', 0, ScaleErr)
    Hist_PerpZ('recoilpatpfPUCorrectedMET_PerpZ', 'PU corrected MET', 7, ScaleErr)
    Hist_PerpZ( 'recoilpatpfPUMET_PerpZ', 'PU MET', 2, ScaleErr)
    Hist_PerpZ('recoilpatpfTrackMET_PerpZ', 'Track MET', 3, ScaleErr)
    Hist_PerpZ('recoilslimmedMETsPuppi_PerpZ', 'Puppi MET', 4, ScaleErr)
    #Hist_PerpZ('LongZCorrectedRecoil_PerpZ', 'GBRT MET', 5, ScaleErr)
    Hist_PerpZ('recoilslimmedMETs_PerpZ', 'PF MET', 1, ScaleErr)
    Hist_PerpZ('NN_PerpZ', 'NN MET', 6, ScaleErr)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.ylabel('Counts')
    plt.xlabel('$ U_{\perp}  $ in GeV')
    plt.xlim(HistLimMin,HistLimMax)
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{|-\\vec{p}_T^Z|} \\right) $ in GeV')
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
    plt.xlabel('$ U_{\parallel} - |-\\vec{p}_T^Z|  $ in GeV')
    plt.xlim(HistLimMin,HistLimMax)
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{|-\\vec{p}_T^Z|} \\right) $ in GeV')
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





    Histogram_Deviation_para_PV('recoilpatpfNoPUMET_LongZ', 'No PU MET', 0, ScaleErr)
    Histogram_Deviation_para_PV('recoilpatpfPUCorrectedMET_LongZ', 'PU corrected MET', 7, ScaleErr)
    Histogram_Deviation_para_PV( 'recoilpatpfPUMET_LongZ', 'PU MET', 2, ScaleErr)
    Histogram_Deviation_para_PV('recoilpatpfTrackMET_LongZ', 'Track MET', 3, ScaleErr)
    Histogram_Deviation_para_PV('recoilslimmedMETsPuppi_LongZ', 'Puppi MET', 4, ScaleErr)
    #Histogram_Deviation_para_PV('LongZCorrectedRecoil_LongZ', 'GBRT MET', 5, ScaleErr)
    Histogram_Deviation_para_PV('recoilslimmedMETs_LongZ', 'PF MET', 1, ScaleErr)
    Histogram_Deviation_para_PV('NN_LongZ', 'NN MET', 6, ScaleErr)


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    plt.ylabel('Counts')
    plt.xlabel('$ U_{\parallel} - p_T^Z  $ in GeV')
    plt.xlim(HistLimMin,HistLimMax)
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{|-\\vec{p}_T^Z|} \\right) $ in GeV')
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



    Histogram_Angle_Dev('recoilpatpfNoPUMET_Phi', 'No PU MET', 0, ScaleErr)
    Histogram_Angle_Dev('recoilpatpfPUCorrectedMET_Phi', 'PU corrected MET', 7, ScaleErr)
    Histogram_Angle_Dev( 'recoilpatpfPUMET_Phi', 'PU MET', 2, ScaleErr)
    Histogram_Angle_Dev('recoilpatpfTrackMET_Phi',  'Track MET', 3, ScaleErr)
    Histogram_Angle_Dev('recoilslimmedMETsPuppi_Phi',  'Puppi MET', 4, ScaleErr)
    #Histogram_Angle_Dev('LongZCorrectedRecoil_Phi',  'GBRT MET', 5, ScaleErr)
    Histogram_Angle_Dev('recoilslimmedMETs_Phi',  'PF MET', 1, ScaleErr)
    Histogram_Angle_Dev('NN_Phi', 'NN MET', 6, ScaleErr)


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    plt.ylabel('Counts')
    plt.xlabel('$ \\Delta \\alpha $ in rad')
    plt.xlim(-np.pi,np.pi)
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{|-\\vec{p}_T^Z|} \\right) $ in GeV')
    #plt.title('Deviation Histogram perp')
    #plt.text('$p_T$ and $\# pT$ range restriction')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    ##plt.ylim(ylimResMVAMin, ylimResMax)
    plt.savefig("%sHist_Delta_Alpha.png"%(plotsD), bbox_inches="tight")
    plt.close()
    print('getAngle(NN_Phi)[getResponseIdx(NN_LongZ)].shape', getAngle('NN_Phi')[getResponseIdx('NN_LongZ')].shape)
    print('getResponse(NN_LongZ)', getResponse('NN_LongZ').shape)
    print('getResponse(recoilslimmedMETs_LongZ)', getResponse('recoilslimmedMETs_LongZ').shape)


    '''
    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)





    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    #Mean_Std_Deviation_pTZ_para('LongZCorrectedRecoil_LongZ', 'GBRT', 5, ScaleErr)
    Mean_Std_Deviation_pTZ_para('NN_LongZ', 'NN', 6, ScaleErr)
    Mean_Std_Deviation_pTZ_para('recoilslimmedMETs_LongZ', 'PF', 1, ScaleErr)


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    plt.ylabel('$\\langle  U_{\parallel} - p_T^Z \\rangle$')
    plt.xlabel('$p_T^Z $ in GeV')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{|-\\vec{p}_T^Z|} \\right) $ in GeV')
    #plt.title('parallel deviation over $|-\\vec{p}_T^Z|$')
    #plt.text('$p_T$ and $\# PV$ range restriction')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    ##plt.ylim(HistLimMin, HistLimMax)
    plt.savefig("%sDelta_para_Std_para_pT.png"%(plotsD), bbox_inches="tight")
    plt.close()
    '''



    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    Histogram_Response('recoilslimmedMETsPuppi_LongZ', 'Puppi', 4, ScaleErr)
    Histogram_Response('NN_LongZ', 'NN', 6, ScaleErr)
    Histogram_Response('recoilslimmedMETs_LongZ', 'PF', 1, ScaleErr)


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    plt.xlabel('$  \\frac{U_{\parallel}}{|-\\vec{p}_T^Z|}} $')
    plt.ylabel('Counts')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{|-\\vec{p}_T^Z|} \\right) $ in GeV')
    #plt.title('parallel deviation over $|-\\vec{p}_T^Z|$')
    #plt.text('$p_T$ and $\# PV$ range restriction')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    plt.xlim(ResponseMin, ResponseMax)
    plt.savefig("%sHist_Response.png"%(plotsD), bbox_inches="tight")
    plt.close()




    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    #Mean_Std_Deviation_pTZ_perp('LongZCorrectedRecoil_PerpZ', 'GBRT', 5, ScaleErr)
    Mean_Std_Deviation_pTZ_perp('NN_PerpZ', 'NN', 6, ScaleErr)
    Mean_Std_Deviation_pTZ_perp('recoilslimmedMETs_PerpZ', 'PF', 1, ScaleErr)


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    plt.ylabel('$\\langle  U_{\perp} \\rangle$')
    plt.xlabel('$ p_T^Z $ in GeV')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{|-\\vec{p}_T^Z|} \\right) $ in GeV')
    #plt.title('perpendicular deviation over $|-\\vec{p}_T^Z|$')
    #plt.text('$p_T$ and $\# PV$ range restriction')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    ##plt.ylim(HistLimMin, HistLimMax)
    plt.savefig("%sDelta_perp_Std_perp_pT.png"%(plotsD), bbox_inches="tight")
    plt.close()


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    #Mean_Std_Deviation_PV_para('LongZCorrectedRecoil_LongZ', 'GBRT', 5, ScaleErr)
    Mean_Std_Deviation_PV_para('NN_LongZ', 'NN', 6, ScaleErr)
    Mean_Std_Deviation_PV_para('recoilslimmedMETs_LongZ', 'PF', 1, ScaleErr)


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    plt.ylabel('$\\langle  U_{\parallel} - p_T^Z \\rangle$')
    plt.xlabel('\#PV')
    #plt.title('parallel deviation over \#PV')
    #plt.text('$p_T$ and $\# PV$ range restriction')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    ##plt.ylim(HistLimMin, HistLimMax)
    plt.savefig("%sDelta_para_Std_para_PV.png"%(plotsD), bbox_inches="tight")
    plt.close()


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    #Mean_Std_Deviation_PV_perp('LongZCorrectedRecoil_PerpZ', 'GBRT', 5, ScaleErr)
    Mean_Std_Deviation_PV_perp('NN_PerpZ', 'NN', 6, ScaleErr)
    Mean_Std_Deviation_PV_perp('recoilslimmedMETs_PerpZ', 'PF', 1, ScaleErr)


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    plt.ylabel('$\\langle  U_{\perp} \\rangle$')
    plt.xlabel('#PV')
    #plt.title('parallel deviation over \#PV')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    ##plt.ylim(HistLimMin, HistLimMax)
    plt.savefig("%sDelta_perp_Std_perp_PV.png"%(plotsD), bbox_inches="tight")

    plt.clf()
    plt.figure()
    #plt.suptitle('x-Korrektur Prediction-Target ')
    plt.ylabel("$ \Delta \\alpha $")
    plt.xlabel("Response")
    heatmap, xedges, yedges = np.histogram2d(   getResponse('NN_LongZ'), getAngle('NN_Phi')[getResponseIdx('NN_LongZ')], bins=50,
                                             range=[[-1,3],
                                                    [-np.pi,np.pi]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #plt.clf()
    HM =plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm(), aspect='auto')
    plt.colorbar(HM)
    plt.legend()
    plt.savefig("%sHM_Response_NN_Delta_Alpha.png"%(plotsD), bbox_inches="tight")
    plt.close()
    '''
    plt.clf()
    plt.figure()
    plt.suptitle(pTRangeString_low)
    plt.xlabel("$ \Delta \\alpha $")
    plt.ylabel("Response")
    Angle, Response = getResponse_pTRange('NN_LongZ', ptMin_low, ptMax_low)
    print("length   IdxAngle", len(Angle))
    print("length   IdxResponse", len(Response))
    heatmap, xedges, yedges = np.histogram2d(   Angle, Response, bins=50,
                                             range=[[-np.pi,np.pi],
                                                    [-10,10]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #plt.clf()
    HM =plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm(), aspect='auto')
    plt.colorbar(HM)
    plt.legend()
    plt.savefig("%sHM_Response_NN_Delta_Alpha_low.png"%(plotsD), bbox_inches="tight")
    plt.close()

    plt.clf()
    plt.figure()
    plt.suptitle(pTRangeString_mid)
    plt.xlabel("$ \Delta \\alpha $")
    plt.ylabel("Response")
    Angle, Response = getResponse_pTRange('NN_LongZ', ptMin_mid, ptMax_mid)
    heatmap, xedges, yedges = np.histogram2d(   Angle, Response, bins=50,
                                             range=[[-np.pi,np.pi],
                                                    [-1.5,1.5]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #plt.clf()
    HM =plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm(), aspect='auto')
    plt.colorbar(HM)
    plt.legend()
    plt.savefig("%sHM_Response_NN_Delta_Alpha_mid.png"%(plotsD), bbox_inches="tight")
    plt.close()


    plt.clf()
    plt.figure()
    plt.suptitle(pTRangeString_high)
    plt.xlabel("$ \Delta \\alpha $")
    plt.ylabel("Response")
    Angle, Response = getResponse_pTRange('NN_LongZ', ptMin_high, ptMax_high)
    heatmap, xedges, yedges = np.histogram2d(   Angle, Response, bins=50,
                                             range=[[-np.pi,np.pi],
                                                    [-1.5,1.5]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #plt.clf()
    HM =plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm(), aspect='auto')
    plt.colorbar(HM)
    plt.legend()
    plt.savefig("%sHM_Response_NN_Delta_Alpha_high.png"%(plotsD), bbox_inches="tight")
    plt.close()


    plt.clf()
    plt.figure()
    plt.suptitle(pTRangeString_low)
    plt.xlabel("$ \Delta \\alpha $")
    plt.ylabel("Response")
    Angle, Response = getResponse_pTRange('recoilslimmedMETs_LongZ', ptMin_low, ptMax_low)
    heatmap, xedges, yedges = np.histogram2d(   Angle, Response, bins=50,
                                             range=[[-np.pi,np.pi],
                                                    [-10,10]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #plt.clf()
    HM =plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm(), aspect='auto')
    plt.colorbar(HM)
    plt.legend()
    plt.savefig("%sHM_Response_PF_Delta_Alpha_low.png"%(plotsD), bbox_inches="tight")
    plt.close()

    plt.clf()
    plt.figure()
    plt.suptitle(pTRangeString_mid)
    plt.xlabel("$ \Delta \\alpha $")
    plt.ylabel("Response")
    Angle, Response = getResponse_pTRange('recoilslimmedMETs_LongZ', ptMin_mid, ptMax_mid)
    heatmap, xedges, yedges = np.histogram2d(   Angle, Response, bins=50,
                                             range=[[-np.pi,np.pi],
                                                    [-1.5,1.5]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #plt.clf()
    HM =plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm(), aspect='auto')
    plt.colorbar(HM)
    plt.legend()
    plt.savefig("%sHM_Response_PF_Delta_Alpha_mid.png"%(plotsD), bbox_inches="tight")
    plt.close()


    plt.clf()
    plt.figure()
    plt.suptitle(pTRangeString_high)
    plt.xlabel("$ \Delta \\alpha $")
    plt.ylabel("Response")
    Angle, Response = getResponse_pTRange('recoilslimmedMETs_LongZ', ptMin_high, ptMax_high)
    heatmap, xedges, yedges = np.histogram2d(   Angle, Response, bins=50,
                                             range=[[-np.pi,np.pi],
                                                    [-1.5,1.5]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #plt.clf()
    HM =plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm(), aspect='auto')
    plt.colorbar(HM)
    plt.legend()
    plt.savefig("%sHM_Response_PF_Delta_Alpha_high.png"%(plotsD), bbox_inches="tight")
    plt.close()
    '''

    print("NN: Korrellationskoeffizient zwischen Response und Resolution para",np.corrcoef(-DFName['NN_LongZ']/DFName[Target_Pt], -DFName['NN_LongZ']-DFName[Target_Pt]))
    print("PF: Korrellationskoeffizient zwischen Response und Resolution para",np.corrcoef(-DFName['recoilslimmedMETs_LongZ']/DFName[Target_Pt], -DFName['recoilslimmedMETs_LongZ']-DFName[Target_Pt]))
    #print("GBRT: Korrellationskoeffizient zwischen Response und Resolution para",np.corrcoef(-DFName['LongZCorrectedRecoil_LongZ']/DFName[Target_Pt], -DFName['LongZCorrectedRecoil_LongZ']-DFName[Target_Pt]))

    print("NN: Korrellationskoeffizient zwischen Response und Resolution perp",np.corrcoef(DFName['NN_PerpZ'], -DFName['NN_LongZ']-DFName[Target_Pt]))
    print("PF: Korrellationskoeffizient zwischen Response und Resolution perp",np.corrcoef(DFName['recoilslimmedMETs_PerpZ'], -DFName['recoilslimmedMETs_LongZ']-DFName[Target_Pt]))
    #print("GBRT: Korrellationskoeffizient zwischen Response und Resolution perp",np.corrcoef(DFName['LongZCorrectedRecoil_PerpZ'], -DFName['LongZCorrectedRecoil_LongZ']-DFName[Target_Pt]))

    print("NN: Korrellationskoeffizient zwischen Resolution para und Resolution perp",np.corrcoef(DFName['NN_PerpZ'], -DFName['NN_LongZ']-DFName[Target_Pt]))
    print("PF: Korrellationskoeffizient zwischen Resolution para und Resolution perp",np.corrcoef(DFName['recoilslimmedMETs_PerpZ']/DFName[Target_Pt], -DFName['recoilslimmedMETs_LongZ']-DFName[Target_Pt]))
    #print("GBRT: Korrellationskoeffizient zwischen Resolution para und Resolution perp",np.corrcoef(DFName['LongZCorrectedRecoil_PerpZ'], -DFName['LongZCorrectedRecoil_LongZ']-DFName[Target_Pt]))



    plt.clf()
    plt.figure()
    #plt.suptitle('x-Korrektur Prediction-Target ')
    plt.xlabel("$ \Delta \\alpha $")
    plt.ylabel("Response")
    dPhi = np.linspace(-np.pi, np.pi, 200)
    Response_pTperfect = np.cos(dPhi)
    mean_Response('NN_LongZ', 'NN_Phi', 'NN', 6, ScaleErr)
    mean_Response('recoilslimmedMETsPuppi_LongZ', 'recoilslimmedMETsPuppi_Phi', 'Puppi', 5, ScaleErr)
    mean_Response('recoilslimmedMETs_LongZ', 'recoilslimmedMETs_Phi', 'PF', 1, ScaleErr)
    plt.plot(np.linspace(-np.pi, np.pi, 2000), np.cos(np.linspace(-np.pi, np.pi, 2000)), linewidth=2, markersize=12, label='$\cos \Delta \\alpha$')
    plt.legend()
    plt.ylim(-7,5)
    plt.grid()
    plt.rcParams['agg.path.chunksize'] = 10000
    plt.savefig("%sNN_Delta_Alpha_perfect_Guess.png"%(plotsD), bbox_inches="tight")
    plt.close()

    plt.clf()
    plt.figure()
    #plt.suptitle('x-Korrektur Prediction-Target ')
    plt.xlabel("$ \Delta \\alpha $")
    plt.ylabel("Response")
    dPhi = np.linspace(-np.pi, np.pi, 200)
    Response_pTperfect = np.cos(dPhi)
    ScaleErrResponse = 1000
    mean_Response_wErr('NN_LongZ', 'NN_Phi', 'NN', 6, ScaleErrResponse, 0, 200, 20)
    mean_Response_wErr('recoilslimmedMETsPuppi_LongZ', 'recoilslimmedMETsPuppi_Phi', 'Puppi', 5, ScaleErrResponse, 0, 200, 20)
    mean_Response_wErr('recoilslimmedMETs_LongZ', 'recoilslimmedMETs_Phi', 'PF', 1, ScaleErrResponse, 0, 200, 20)
    plt.plot(np.linspace(-np.pi, np.pi, 2000), np.cos(np.linspace(-np.pi, np.pi, 2000)), linewidth=2, markersize=12, label='$\cos \Delta \\alpha$')
    plt.legend()
    plt.ylim(-8,5)
    plt.grid()
    plt.rcParams['agg.path.chunksize'] = 10000
    plt.savefig("%sNN_Delta_Alpha_perfect_Guess_wErr.png"%(plotsD), bbox_inches="tight")
    plt.close()
    '''
    plt.clf()
    plt.figure()
    #plt.suptitle('x-Korrektur Prediction-Target ')
    plt.xlabel("$ \Delta \\alpha $")
    plt.ylabel("Response")
    dPhi = np.linspace(-np.pi, np.pi, 200)
    Response_pTperfect = np.cos(dPhi)
    mean_Response_wErr('NN_LongZ', 'NN_Phi', 'NN', 6, ScaleErrResponse, ptMin_low, ptMax_low, 20)
    mean_Response_wErr('recoilslimmedMETsPuppi_LongZ', 'recoilslimmedMETsPuppi_Phi', 'Puppi', 5, ScaleErrResponse, ptMin_low, ptMax_low, 20)
    mean_Response_wErr('recoilslimmedMETs_LongZ', 'recoilslimmedMETs_Phi', 'PF', 1, ScaleErrResponse, ptMin_low, ptMax_low, 20)
    plt.plot(np.linspace(-np.pi, np.pi, 2000), np.cos(np.linspace(-np.pi, np.pi, 2000)), linewidth=2, markersize=12, label='$\cos \Delta \\alpha$')
    plt.legend()
    #plt.ylim(-8,5)
    plt.grid()
    plt.rcParams['agg.path.chunksize'] = 10000
    plt.savefig("%sNN_Delta_Alpha_perfect_Guess_wErr_low.png"%(plotsD), bbox_inches="tight")
    plt.close()

    plt.clf()
    plt.figure()
    #plt.suptitle('x-Korrektur Prediction-Target ')
    plt.xlabel("$ \Delta \\alpha $")
    plt.ylabel("Response")
    dPhi = np.linspace(-np.pi, np.pi, 200)
    Response_pTperfect = np.cos(dPhi)
    mean_Response_wErr('NN_LongZ', 'NN_Phi', 'NN', 6, ScaleErrResponse, ptMin_mid, ptMax_mid, 10)
    mean_Response_wErr('recoilslimmedMETsPuppi_LongZ', 'recoilslimmedMETsPuppi_Phi', 'Puppi', 5, ScaleErrResponse, ptMin_mid, ptMax_mid, 10)
    mean_Response_wErr('recoilslimmedMETs_LongZ', 'recoilslimmedMETs_Phi', 'PF', 1, ScaleErrResponse, ptMin_mid, ptMax_mid, 10)
    plt.plot(np.linspace(-np.pi, np.pi, 2000), np.cos(np.linspace(-np.pi, np.pi, 2000)), linewidth=2, markersize=12, label='$\cos \Delta \\alpha$')
    plt.legend()
    #plt.ylim(-8,5)
    plt.grid()
    plt.rcParams['agg.path.chunksize'] = 10000
    plt.savefig("%sNN_Delta_Alpha_perfect_Guess_wErr_mid.png"%(plotsD), bbox_inches="tight")
    plt.close()
    '''

    plt.clf()
    plt.figure()
    #plt.suptitle('x-Korrektur Prediction-Target ')
    plt.xlabel("$ \Delta \\alpha $")
    plt.ylabel("Response")
    dPhi = np.linspace(-np.pi, np.pi, 200)
    Response_pTperfect = np.cos(dPhi)
    mean_Response_wErr('NN_LongZ', 'NN_Phi', 'NN', 6, ScaleErrResponse, ptMin_high, ptMax_high, 10)
    mean_Response_wErr('recoilslimmedMETsPuppi_LongZ', 'recoilslimmedMETsPuppi_Phi', 'Puppi', 5, ScaleErrResponse, ptMin_high, ptMax_high, 10)
    mean_Response_wErr('recoilslimmedMETs_LongZ', 'recoilslimmedMETs_Phi', 'PF', 1, ScaleErrResponse, ptMin_high, ptMax_high, 10)
    plt.plot(np.linspace(-np.pi, np.pi, 2000), np.cos(np.linspace(-np.pi, np.pi, 2000)), linewidth=2, markersize=12, label='$\cos \Delta \\alpha$')
    plt.legend()
    #plt.ylim(-8,5)
    plt.grid()
    plt.rcParams['agg.path.chunksize'] = 10000
    plt.savefig("%sNN_Delta_Alpha_perfect_Guess_wErr_high.png"%(plotsD), bbox_inches="tight")
    plt.close()



    plt.clf()
    plt.figure()
    #plt.suptitle('x-Korrektur Prediction-Target ')
    plt.xlabel("$ \Delta \\alpha $")
    plt.ylabel("Response")
    dPhi = np.linspace(-np.pi/180*10, np.pi/180*10, 200)
    Response_pTperfect = np.cos(dPhi)
    mean_Response_CR('NN_LongZ', 'NN_Phi', 'NN', 6, ScaleErr)
    mean_Response_CR('recoilslimmedMETs_LongZ', 'recoilslimmedMETs_Phi', 'PF', 1, ScaleErr)
    plt.plot(np.linspace(-np.pi/180*10, np.pi/180*10, 2000), np.cos(np.linspace(-np.pi/180*10, np.pi/180*10, 2000)), linewidth=2, markersize=12, label='perfect guess $p_T$')
    plt.ylim(0,5)
    plt.grid()
    plt.legend()
    plt.savefig("%sResponse_NN_Delta_Alpha_CR.png"%(plotsD), bbox_inches="tight")
    plt.close()


    '''
    plt.clf()
    plotHM(getAngle('recoilpatpfPUMET_Phi')[getResponseIdx('recoilpatpfPUMET_LongZ')],getResponse('recoilpatpfPUMET_LongZ'), "$ \Delta \\alpha $", "Response", -np.pi, np.pi ,-1, 3)
    plt.savefig("%sHM_CC_Response_PF_Delta_Alpha.png"%(plotsD), bbox_inches="tight")
    plt.close()

    plt.clf()
    plotHM(getAngle('NN_Phi')[getResponseIdx('NN_LongZ')],getResponse('NN_LongZ'), "$ \Delta \\alpha $", "Response", -np.pi, np.pi ,-1, 3)
    plt.savefig("%sHM_CC_Response_NN_Delta_Alpha.png"%(plotsD), bbox_inches="tight")
    plt.close()
    '''

    plt.clf()
    plt.figure()
    #plt.suptitle('x-Korrektur Prediction-Target ')
    plt.xlabel("$ \Delta \\alpha $")
    plt.ylabel("Response")
    heatmap, xedges, yedges = np.histogram2d(  getAngle('recoilpatpfPUMET_Phi')[getResponseIdx('recoilpatpfPUMET_LongZ')], getResponse('recoilpatpfPUMET_LongZ'),  bins=50,
                                             range=[[-np.pi,np.pi],
                                                    [0,2]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.clf()
    HM =plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm(), aspect='auto')
    plt.colorbar(HM)
    plt.legend()
    plt.savefig("%sHM_Response_PF_Delta_Alpha.png"%(plotsD), bbox_inches="tight")
    plt.close()




    '''
    plt.clf()
    plt.figure()
    plt.xlabel("$ \Delta \\alpha $")
    plt.ylabel("$|U|-|p_T^Z|$")
    PF_r, PF_phi = kar2pol(-DFName['recoilpatpfPUMET_LongZ'], DFName['recoilpatpfPUMET_PerpZ'])
    Response = np.divide(-DFName['recoilpatpfPUMET_LongZ'], DFName[Target_Pt] )
    x=PF_phi
    y=DFName['recoilpatpfPUMET_Pt']-DFName[Target_Pt]
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
    plt.ylabel("$|U|-|p_T^Z|$")
    PF_r, PF_phi = kar2pol(-DFName['NN_LongZ'], DFName['NN_PerpZ'])
    Response = np.divide(-DFName['NN_LongZ'], DFName[Target_Pt] )
    x=PF_phi
    y=np.sqrt(np.multiply(DFName['NN_LongZ'], DFName['NN_LongZ'])+ np.multiply(DFName['NN_PerpZ'], DFName['NN_PerpZ']))-DFName[Target_Pt]
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
    plt.ylabel("$|U|-|p_T^Z|$")
    y=np.sqrt(np.multiply(DFName['recoilslimmedMETs_LongZ'], DFName['recoilslimmedMETs_LongZ'])+ np.multiply(DFName['recoilslimmedMETs_PerpZ'], DFName['recoilslimmedMETs_PerpZ']))-DFName[Target_Pt]
    delta_alpha=np.linspace(-np.pi,np.pi,201)
    delta_pT=np.linspace(y.min(),y.max(),201)
    PF_r, PF_phi = kar2pol(-DFName['recoilslimmedMETs_LongZ'], DFName['recoilslimmedMETs_PerpZ'])
    Response = np.divide(-DFName['recoilslimmedMETs_LongZ'], DFName[Target_Pt] )
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
    plt.ylabel("$|U|-|p_T^Z|$")
    delta_alpha=np.linspace(-np.pi,np.pi,201)
    y=np.sqrt(np.multiply(DFName['NN_LongZ'], DFName['NN_LongZ'])+ np.multiply(DFName['NN_PerpZ'], DFName['NN_PerpZ']))-DFName[Target_Pt]
    delta_pT=np.linspace(y.min(),y.max(),201)
    PF_r, PF_phi = kar2pol(-DFName['NN_LongZ'], DFName['NN_PerpZ'])
    Response = np.divide(-DFName['NN_LongZ'], DFName[Target_Pt] )
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
    '''
    plt.clf()
    plt.figure()
    plt.xlabel("$ \Delta \\alpha $")
    plt.ylabel("$|U|-|p_T^Z|$")
    DeltapT = DFName['recoilpatpfPUMET_Pt'] - DFName[Target_Pt]
    heatmap, xedges, yedges = np.histogram2d( getAngle('recoilpatpfPUMET_Phi'), DeltapT,  bins=50,
                                             range=[[-np.pi,np.pi],
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
    plt.ylabel("$|U|-|p_T^Z|$")
    heatmap, xedges, yedges = np.histogram2d(  angularrange(DFName['NN_Phi']+np.pi-DFName['NN_Phi']), DFName['NN_Pt'] - DFName[Target_Pt],  bins=50,
                                             range=[[-np.pi,np.pi],
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
    Response = np.divide(-DFName['recoilpatpfPUMET_LongZ'], DFName[Target_Pt] )
    Response = Response[~np.isnan(Response)]
    heatmap, xedges, yedges = np.histogram2d(  angularrange(DFName['recoilpatpfPUMET_Phi']+np.pi-DFName[Target_Phi]), Response,  bins=50,
                                             range=[[-np.pi,np.pi],
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
    heatmap, xedges, yedges = np.histogram2d(  getAngle('recoilpatpfPUMET_Phi')[getResponseIdx('recoilpatpfPUMET_LongZ')], getResponse('recoilpatpfPUMET_LongZ'),  bins=[25,25],
                                             range=[[-np.pi,np.pi],
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
    heatmap, xedges, yedges = np.histogram2d(  getAngle('NN_Phi')[getResponseIdx('NN_LongZ')], getResponse('NN_LongZ'),  bins=[25,25],
                                             range=[[-np.pi,np.pi],
                                                    [0.75,1.25]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #plt.clf()
    HM =plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm(), aspect='auto')
    plt.colorbar(HM)
    plt.legend()
    plt.savefig("%sHM_Response_NN_Delta_Alpha_CR.png"%(plotsD), bbox_inches="tight")
    '''

    plt.clf()
    plt.figure()
    #plt.suptitle('x-Korrektur Prediction-Target ')
    plt.ylabel("$ \Delta \\alpha $")
    plt.xlabel("$p_T$")
    heatmap, xedges, yedges = np.histogram2d(  DFName['NN_Pt'], getAngle('NN_Phi'),  bins=50,
                                             range=[[0,200], [-np.pi,np.pi]])
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
    heatmap, xedges, yedges = np.histogram2d(  DFName['recoilpatpfPUMET_Pt'], getAngle('recoilpatpfPUMET_Phi'),  bins=50,
                                             range=[[0,200], [-np.pi,np.pi]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #plt.clf()
    HM =plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm(), aspect='auto')
    plt.colorbar(HM)
    plt.legend()
    plt.savefig("%sHM_PF_Delta_Alpha_pT.png"%(plotsD), bbox_inches="tight")

    '''
    plt.clf()
    plt.figure()
    #plt.suptitle('x-Korrektur Prediction-Target ')
    plt.ylabel("$ \Delta \\alpha $")
    plt.xlabel("$p_T$")
    heatmap, xedges, yedges = np.histogram2d(  DFName['recoilpatpfPUMET_Pt'], getAngle('recoilpatpfPUMET_Phi'),  bins=50,
                                             range=[[0,30], [-np.pi,np.pi]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #plt.clf()
    HM =plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm(), aspect='auto')
    plt.colorbar(HM)
    plt.legend()
    plt.savefig("%sHM_PF_Delta_Alpha_pT_CR.png"%(plotsD), bbox_inches="tight")


    plt.clf()
    plt.figure()
    #plt.suptitle('x-Korrektur Prediction-Target ')
    plt.ylabel("$ \Delta \\alpha $")
    plt.xlabel("$p_T$")
    heatmap, xedges, yedges = np.histogram2d(  DFName['NN_Pt'], getAngle('NN_Phi'),  bins=50,
                                             range=[[0,30], [-np.pi,np.pi]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #plt.clf()
    HM =plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm(), aspect='auto')
    plt.colorbar(HM)
    plt.legend()
    plt.savefig("%sHM_NN_Delta_Alpha_pT_CR.png"%(plotsD), bbox_inches="tight")

    plt.clf()
    plt.figure()
    #plt.suptitle('x-Korrektur Prediction-Target ')
    plt.xlabel("$ |-\\vec{p}_T^Z| $")
    plt.ylabel("Response")
    heatmap, xedges, yedges = np.histogram2d(  DFName[Target_Pt][getResponseIdx('NN_LongZ')], getResponse('NN_LongZ'),  bins=[25,25],
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
    plt.xlabel("$ |-\\vec{p}_T^Z| $")
    plt.ylabel("Response")
    heatmap, xedges, yedges = np.histogram2d(  DFName[Target_Pt][getResponseIdx('NN_LongZ')], getResponse('NN_LongZ'),  bins=[25,25],
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
    heatmap, xedges, yedges = np.histogram2d(  DFName[Target_Pt], getResponse('NN_LongZ'),  bins=[25,25],
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
    plt.xlabel("$ |-\\vec{p}_T^Z| $")
    plt.ylabel("Response")
    heatmap, xedges, yedges = np.histogram2d(  DFName[Target_Pt][getResponseIdx('recoilpatpfPUMET_LongZ')], getResponse('recoilpatpfPUMET_LongZ'),  bins=[25,25],
                                             range=[[0,10],
                                                    [0,2]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #plt.clf()
    HM =plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm(), aspect='auto')
    plt.colorbar(HM)
    plt.legend()
    plt.savefig("%sHM_Response_PF_Delta_pT_10.png"%(plotsD), bbox_inches="tight")
    '''
    plt.clf()
    plt.figure()
    #plt.suptitle('x-Korrektur Prediction-Target ')
    plt.xlabel("$ |-\\vec{p}_T^Z| $")
    plt.ylabel("Response")
    heatmap, xedges, yedges = np.histogram2d(  DFName[Target_Pt][getResponseIdx('recoilpatpfPUMET_LongZ')], getResponse('recoilpatpfPUMET_LongZ'),  bins=[25,25],
                                             range=[[0,200],
                                                    [0,2]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #plt.clf()
    HM =plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm(), aspect='auto')
    plt.colorbar(HM)
    plt.legend()
    plt.savefig("%sHM_Response_PF_Delta_pT_0_200.png"%(plotsD), bbox_inches="tight")
    '''
    plt.clf()
    plt.figure()
    #plt.suptitle('x-Korrektur Prediction-Target ')
    plt.xlabel("$ |-\\vec{p}_T^Z| $")
    plt.ylabel("Response")
    heatmap, xedges, yedges = np.histogram2d(  DFName[Target_Pt][getResponseIdx('NN_LongZ')], getResponse('NN_LongZ'),  bins=[25,25],
                                             range=[[150,200],
                                                    [0,2]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #plt.clf()
    HM =plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm(), aspect='auto')
    plt.colorbar(HM)
    plt.legend()
    plt.savefig("%sHM_Response_NN_Delta_pT_150.png"%(plotsD), bbox_inches="tight")
    '''

    print("low pt range 33 percent of data", np.percentile(DFName[Target_Pt],0.3333*100))
    print("mid pt range 33 percent of data", np.percentile(DFName[Target_Pt],0.6666*100))
    print("high pt range 33 percent of data", np.percentile(DFName[Target_Pt],100))




    print('PF: Wie viele negative Responses', np.sum(DFName['recoilpatpfPUMET_LongZ']<0))
    #print('GBRT: Wie viele negative Responses', np.sum(DFName['LongZCorrectedRecoil_LongZ']<0))
    print('NN: Wie viele negative Responses', np.sum(DFName['NN_LongZ']<0))
    print('BosonPt: Wie viele negative Responses', np.sum(DFName[Target_Pt]<0))

if __name__ == "__main__":
    inputDir = sys.argv[1]
    filesDir =  sys.argv[2]
    plotDir = sys.argv[3]
    PhysicsProcess = sys.argv[4]
    NN_mode = sys.argv[5]
    if PhysicsProcess == 'Tau':
        Target_Pt = 'Boson_Pt'
        Target_Phi = 'Boson_Phi'
        DFName_plain = loadData_woutGBRT(filesDir, inputDir, Target_Pt, Target_Phi, NN_mode, PhysicsProcess)
    else:
        Target_Pt = 'Boson_Pt'
        Target_Phi = 'Boson_Phi'
        DFName_plain = loadData_woutGBRT(filesDir, inputDir, Target_Pt, Target_Phi, NN_mode, PhysicsProcess)
    DFName=DFName_plain[DFName_plain[Target_Pt]<=200]
    DFName=DFName[DFName[Target_Pt]>20]
    DFName=DFName[DFName['NVertex']<=50]
    DFName=DFName[DFName['NVertex']>=0]



    DFName_nVertex = DFName
    getPlotsOutput(inputDir, filesDir, plotDir, DFName, DFName_nVertex, Target_Pt, Target_Phi)
