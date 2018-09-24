
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


pTMin, pTMax = 20,200


nbins = int((pTMax-pTMin)/20)
binsAngle = 7
nbinsVertex = 5
nbinsHist = 40
nbinsHistBin = 40
nbins_relR = 10
colors = cm.brg(np.linspace(0, 1, 8))

colors_InOut = cm.brg(np.linspace(0, 1, 8))
colors2 = colors
HistLimMin, HistLimMax = -50, 50
ResponseMin, ResponseMax = -1,3
ResolutionParaMin, ResolutionParaMax = -60, 60
ResolutionPerpMin, ResolutionPerpMax = -60, 60
ResponseMinErr, ResponseMaxErr = 0, 1.05
ylimResMVAMin, ylimResMVAMax = 0, 35
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

    n, _ = np.histogram(DFName[Target_Pt], bins=nbins)
    sy, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=(getResponse(branchString)))
    sy2, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=(DFName[branchString]/DFName[Target_Pt])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    meanc = np.mean((getResponse(branchString)))
    stdc = np.std((getResponse(branchString)))
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName+': %8.2f $\pm$ %8.2f'%(meanc, stdc), linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAmedianResponseOverpTZ_woutError(branchString, labelName, errbars_shift, ScaleErr):

    n, _ = np.histogram(DFName[Target_Pt], bins=nbins)
    sy, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=(getResponse(branchString)))
    sy2, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=(DFName[branchString]/DFName[Target_Pt])**2)
    end, start = _[1:], _[:-1]
    median = [np.median(getResponse(branchString)[(DFName[Target_Pt]>=start[i]) & (DFName[Target_Pt]<end[i])]) for i in range(0,len(start))]
    if branchString=="NN_LongZ":
        print("median NN", median)
        print("median NN, len(getResponse(branchString)))", len(getResponse(branchString)))
        print("median NN, [(DFName[Target_Pt]>=start[0]) & (DFName[Target_Pt]<end[0])]", [(DFName[Target_Pt]>=start[0]) & (DFName[Target_Pt]<end[0])])
        print("any nans", np.any(np.isnan(getResponse(branchString)[(DFName[Target_Pt]>=start[0]) & (DFName[Target_Pt]<end[0])])))
    #mean = sy / n
    #std = np.sqrt(sy2/n - mean*mean)
    meanc = np.median((getResponse(branchString)))
    stdc = np.std((getResponse(branchString)))
    plt.errorbar((_[1:] + _[:-1])/2, median, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName+': %8.2f $\pm$ %8.2f'%(meanc, stdc), linestyle="None", capsize=0,  color=colors[errbars_shift])


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
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName+': %8.2f $\pm$ %8.2f'%(meanc, stdc), linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAAngularOverpTZ(branchStringPhi, labelName, errbars_shift, ScaleErr):
    binwidth = (DFName[Target_Pt].values.max() - DFName[Target_Pt].values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName[Target_Pt], bins=nbins)
    sy, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=getAngle(branchStringPhi))
    sy2, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=(getAngle(branchStringPhi))**2)
    meanc = np.mean(getAngle(branchStringPhi))
    stdc = np.std(getAngle(branchStringPhi))
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName+': %8.2f $\pm$ %8.2f'%(meanc, stdc), linestyle="None", capsize=0,  color=colors[errbars_shift])



def plotMVANormOverpTZ_wErr(branchString, labelName, errbars_shift, ScaleErr):
    binwidth = (DFName[Target_Pt].values.max() - DFName[Target_Pt].values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName[Target_Pt][getResponseIdx(branchString)], bins=nbins)
    sy, _ = np.histogram(DFName[Target_Pt][getResponseIdx(branchString)], bins=nbins, weights=getResponse(branchString))
    sy2, _ = np.histogram(DFName[Target_Pt][getResponseIdx(branchString)], bins=nbins, weights=getResponse(branchString)**2)
    meanc = np.mean(getResponse(branchString))
    stdc = np.std(getResponse(branchString))
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName+': %8.2f $\pm$ %8.2f'%(meanc, stdc), linestyle="None", capsize=0,  color=colors[errbars_shift])
    if errbars_shift==5: errbars_shift2 = 0
    elif errbars_shift==6: errbars_shift2 = 2
    else: errbars_shift2 = errbars_shift
    plt.errorbar((_[:-1]+(_[1:]-_[:-1])/3*errbars_shift2), mean, yerr=std*ScaleErr, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAAngularOverpTZ_wErr(branchStringPhi, labelName, errbars_shift, ScaleErr):
    nbins=1000
    ScaleErr=1
    #binwidth = (DFName[Target_Pt].values.max() - DFName[Target_Pt].values.min())/(nbins) #5MET-Definitionen
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
    #print('pT bin End with std under 45 degrees', _[indSmaller45Degrees[0:10]]+binwidth)
    print('crosscheck std with std under 45 degrees', std[indSmaller45Degrees[0:10]])
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName+': %8.2f $\pm$ %8.2f'%(meanc, stdc), linestyle="None", capsize=0,  color=colors[errbars_shift])
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
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName+': %8.2f $\pm$ %8.2f'%(mean, std), linestyle="None", capsize=0,  color=colors[errbars_shift])


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
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName+': %8.2f $\pm$ %8.2f'%(meanc, stdc), linestyle="None", capsize=0,  color=colors_InOut[errbars_shift])

def plotMVAResponseOverNVertex_woutError_Tresh(branchString, labelName, errbars_shift, ScaleErr):
    DFName_nVertex_Tresh = DFName_nVertex[DFName_nVertex[Target_Pt]>pTTresh]
    binwidth = (DFName_nVertex_Tresh.NVertex.values.max() - DFName_nVertex_Tresh.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName_nVertex_Tresh.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName_nVertex_Tresh.NVertex, bins=nbinsVertex, weights=-(DFName_nVertex_Tresh[branchString])/DFName_nVertex_Tresh[Target_Pt])
    sy2, _ = np.histogram(DFName_nVertex_Tresh.NVertex, bins=nbinsVertex, weights=(DFName_nVertex_Tresh[branchString]/DFName_nVertex_Tresh[Target_Pt])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName+': %8.2f $\pm$ %8.2f'%(mean, std), linestyle="None", capsize=0,  color=colors[errbars_shift])

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
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName+': %8.2f $\pm$ %8.2f'%(mean, std), linestyle="None", capsize=0,  color=colors[errbars_shift])



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
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName+': %8.2f $\pm$ %8.2f'%(mean, std), linestyle="None", capsize=0,  color=colors[errbars_shift])


def plotMVAResolutionOverpTZ_woutError_para(branchString, labelName, errbars_shift, ScaleErr):
    binwidth = (DFName[Target_Pt].values.max() - DFName[Target_Pt].values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName[Target_Pt], bins=nbins)
    sy, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=-(DFName[branchString]+DFName[Target_Pt]))
    sy2, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=(DFName[branchString]+DFName[Target_Pt])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    #rangemin, rangemax = 20, 200
    stdc = np.std((-(DFName[branchString])-DFName[Target_Pt]))
    #stdc = np.std(-(DFName[branchString]))
    plt.errorbar((_[1:] + _[:-1])/2, std, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName+': %8.2f'%(stdc), linestyle="None", capsize=0,  color=colors[errbars_shift])


def mean_Response(branchString_Long, branchString_Phi, labelName, errbars_shift, ScaleErr):
    binwidth = (-np.pi - np.pi)/binsAngle
    n, _ = np.histogram(DFName[Target_Pt], bins=nbins)
    NN_phi = getAngle(branchString_Phi)[getResponseIdx(branchString_Long)]

    n, _ = np.histogram(NN_phi, bins=20)
    sy, _ = np.histogram(NN_phi, bins=20, weights=getResponse(branchString_Long))
    sy2, _ = np.histogram(NN_phi, bins=20, weights=(getResponse(branchString_Long))**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName+': %8.2f $\pm$ %8.2f'%(mean, std), linestyle="None", capsize=0,  color=colors[errbars_shift])
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
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName+': %8.2f $\pm$ %8.2f'%(mean, std), linestyle="None", capsize=0,  color=colors[errbars_shift])
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
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName+': %8.2f $\pm$ %8.2f'%(mean, std), linestyle="None", capsize=0,  color=colors[errbars_shift])
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
    plt.errorbar((_[1:] + _[:-1])/2, std, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName+': %8.2f $\pm$ %8.2f'%(mean, std), linestyle="None", capsize=0,  color=colors_InOut[errbars_shift])

def MeanDeviation_Pt(branchString, labelName, errbars_shift, ScaleErr):
    binwidth = (DFName[Target_Pt].values.max() - DFName[Target_Pt].values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName[Target_Pt], bins=nbins)
    sy, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=-(DFName[branchString]+DFName[Target_Pt]))
    sy2, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=(DFName[branchString]+DFName[Target_Pt])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName+': %8.2f $\pm$ %8.2f'%(mean, std), linestyle="None", capsize=0,  color=colors[errbars_shift])


def MeanDeviation_PV(branchString, labelName, errbars_shift, ScaleErr):
    binwidth = (DFName_nVertex.NVertex.values.max() - DFName_nVertex.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=-(DFName_nVertex[branchString]+DFName_nVertex[Target_Pt]))
    sy2, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(DFName_nVertex[branchString]+DFName_nVertex[Target_Pt])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName+': %8.2f $\pm$ %8.2f'%(mean, std), linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAResolutionOverpTZ_woutError_perp(branchString, labelName, errbars_shift, ScaleErr):
    binwidth = (DFName[Target_Pt].values.max() - DFName[Target_Pt].values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName[Target_Pt], bins=nbins)
    sy, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=-(DFName[branchString]))
    sy2, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=(DFName[branchString])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    meanc=np.mean(-(DFName[branchString]))
    stdc=np.std(-(DFName[branchString]))
    plt.errorbar((_[1:] + _[:-1])/2, std, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName+': %8.2f '%(stdc), linestyle="None", capsize=0,  color=colors_InOut[errbars_shift])




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

def getstdcpara(branchString, rangemin, rangemax):
    return np.std((-(DFName[branchString][(DFName[Target_Pt]>rangemin) & (DFName[Target_Pt]<rangemax)])-DFName[Target_Pt][(DFName[Target_Pt]>rangemin) & (DFName[Target_Pt]<rangemax)]))

def Hist_Resolution_para_RC(branchString, labelName, errbars_shift, ScaleErr, rangemin, rangemax):
    u_para = -(DFName[branchString][(DFName[Target_Pt]>rangemin) & (DFName[Target_Pt]<rangemax)])
    pZ = DFName[Target_Pt][(DFName[Target_Pt]>rangemin) & (DFName[Target_Pt]<rangemax)]
    RC = np.divide((u_para-pZ),np.divide(u_para,pZ))
    Mean = np.mean(RC)
    Std = np.std(RC)
    plt.hist(RC, bins=nbinsHist, range=[ResolutionParaMin, ResolutionParaMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift])

def Hist_Resolution_perp_RC(branchString, labelName, errbars_shift, ScaleErr, rangemin, rangemax):
    u_para = -(DFName[branchString][(DFName[Target_Pt]>rangemin) & (DFName[Target_Pt]<rangemax)])
    u_perp = DFName[branchString][(DFName[Target_Pt]>rangemin) & (DFName[Target_Pt]<rangemax)]
    pZ = DFName[Target_Pt][(DFName[Target_Pt]>rangemin) & (DFName[Target_Pt]<rangemax)]
    RC = np.divide((u_perp),np.divide(u_para,pZ))
    Mean = np.mean(RC)
    Std = np.std(RC)
    plt.hist(RC, bins=nbinsHist, range=[ResolutionParaMin, ResolutionParaMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift])


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
    plt.hist(Reso[~np.isnan(Reso)], bins=nbinsHistBin, range=[ResponseMin, ResponseMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[bin])

def Histogram_Response_bin(branchString, labelName, bin, ScaleErr, rangemin, rangemax):
    Mean = np.mean(np.divide(-(DFName[branchString][(DFName[Target_Pt]>rangemin) & (DFName[Target_Pt]<rangemax)]),DFName[Target_Pt][(DFName[Target_Pt]>rangemin) & (DFName[Target_Pt]<rangemax)].values))
    Std = np.std(np.divide(-(DFName[branchString][(DFName[Target_Pt]>rangemin) & (DFName[Target_Pt]<rangemax)]),DFName[Target_Pt][(DFName[Target_Pt]>rangemin) & (DFName[Target_Pt]<rangemax)].values))
    Reso = np.divide(-(DFName[branchString][(DFName[Target_Pt]>rangemin) & (DFName[Target_Pt]<rangemax)]),DFName[Target_Pt][(DFName[Target_Pt]>rangemin) & (DFName[Target_Pt]<rangemax)].values)
    #n, _ = np.histogram(-(DFName[branchString])-DFName[Target_Pt].values, bins=nbinsHistBin)
    plt.hist(Reso[~np.isnan(Reso)], bins=nbinsHistBin, range=[ResponseMin, ResponseMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[bin])



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
        plt.hist(((DFName_nVertex[branchString])), bins=nbinsHist, range=[0, 75], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift], linewidth=1.5)

    else:
        if branchString in ['NN_LongZ', 'recoilslimmedMETs_LongZ']:
            Mean = np.mean(-(DFName_nVertex[branchString]))
            Std = np.std(-(DFName_nVertex[branchString]))
            plt.hist((-(DFName_nVertex[branchString])), bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift], linewidth=1.5)
        else:
            Mean = np.mean(-(DFName_nVertex[branchString]))
            Std = np.std(-(DFName_nVertex[branchString]))
            plt.hist((-(DFName_nVertex[branchString])), bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift])



def Hist_LongZ_bin(branchString, labelName, errbars_shift, ScaleErr, rangemin, rangemax):
    if branchString == Target_Pt:
        Mean = np.mean((DFName_nVertex[branchString][(DFName_nVertex[Target_Pt]>rangemin) & (DFName_nVertex[Target_Pt]<rangemax)]))
        Std = np.std((DFName_nVertex[branchString][(DFName_nVertex[Target_Pt]>rangemin) & (DFName_nVertex[Target_Pt]<rangemax)]))
        plt.hist(((DFName_nVertex[branchString][(DFName_nVertex[Target_Pt]>rangemin) & (DFName_nVertex[Target_Pt]<rangemax)])), bins=nbinsHist, range=[0, 75], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift], linewidth=1.5)

    else:
        if branchString in ['NN_LongZ', 'recoilslimmedMETs_LongZ']:
            Mean = np.mean(-(DFName_nVertex[branchString][(DFName_nVertex[Target_Pt]>rangemin) & (DFName_nVertex[Target_Pt]<rangemax)]))
            Std = np.std(-(DFName_nVertex[branchString][(DFName_nVertex[Target_Pt]>rangemin) & (DFName_nVertex[Target_Pt]<rangemax)]))
            plt.hist((-(DFName_nVertex[branchString][(DFName_nVertex[Target_Pt]>rangemin) & (DFName_nVertex[Target_Pt]<rangemax)])), bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift], linewidth=1.5)
        else:
            Mean = np.mean(-(DFName_nVertex[branchString][(DFName_nVertex[Target_Pt]>rangemin) & (DFName_nVertex[Target_Pt]<rangemax)]))
            Std = np.std(-(DFName_nVertex[branchString][(DFName_nVertex[Target_Pt]>rangemin) & (DFName_nVertex[Target_Pt]<rangemax)]))
            plt.hist((-(DFName_nVertex[branchString][(DFName_nVertex[Target_Pt]>rangemin) & (DFName_nVertex[Target_Pt]<rangemax)])), bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift])



def Hist_PerpZ(branchString, labelName, errbars_shift, ScaleErr):
    Mean = np.mean((DFName_nVertex[branchString]))
    Std = np.std((DFName_nVertex[branchString]))
    if branchString in ['NN_PerpZ', 'recoilslimmedMETs_PerpZ']:
        plt.hist(((DFName_nVertex[branchString])), bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift], linewidth=1.5)
    else:
        plt.hist(((DFName_nVertex[branchString])), bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift])

def Hist_PerpZ_bin(branchString, labelName, errbars_shift, ScaleErr, rangemin, rangemax):
    Mean = np.mean((DFName_nVertex[branchString][(DFName_nVertex[Target_Pt]>rangemin) & (DFName_nVertex[Target_Pt]<rangemax)]))
    Std = np.std((DFName_nVertex[branchString][(DFName_nVertex[Target_Pt]>rangemin) & (DFName_nVertex[Target_Pt]<rangemax)]))
    if branchString in ['NN_PerpZ', 'recoilslimmedMETs_PerpZ']:
        plt.hist(((DFName_nVertex[branchString][(DFName_nVertex[Target_Pt]>rangemin) & (DFName_nVertex[Target_Pt]<rangemax)])), bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift], linewidth=1.5)
    else:
        plt.hist(((DFName_nVertex[branchString][(DFName_nVertex[Target_Pt]>rangemin) & (DFName_nVertex[Target_Pt]<rangemax)])), bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift])




def NN_Response_pT( labelName, errbars_shift, ScaleErr):
    binwidth = (DFName[Target_Pt].values.max() - DFName[Target_Pt].values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName[Target_Pt], bins=nbins)
    sy, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=getResponse('NN_LongZ'))
    sy2, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=getResponse('NN_LongZ')**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName+': %8.2f $\pm$ %8.2f'%(mean, std), linestyle="None", capsize=0,  color=colors[errbars_shift])

def NN_Response_PV(branchString, labelName, errbars_shift, ScaleErr):
    binwidth = (DFName_nVertex.NVertex.values.max() - DFName_nVertex.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=getResponse(branchString))
    sy2, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(getResponse(branchString))**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName+': %8.2f $\pm$ %8.2f'%(mean, std), linestyle="None", capsize=0,  color=colors[errbars_shift])

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
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName+': %8.2f $\pm$ %8.2f'%(mean, std), linestyle="None", capsize=0,  color=colors[errbars_shift])

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
    #plt.plot((_[1:] + _[:-1])/2, mean, marker='.', label=labelName+': %8.2f $\pm$ %8.2f'%(mean, std), linestyle="None", color=MVAcolors[errbars_shift])
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName+': %8.2f $\pm$ %8.2f'%(mean, std), linestyle="None", capsize=0,  color=colors[errbars_shift])




def plotMVAResolutionOverpTZ_woutError_para_RC(branchString, labelName, errbars_shift, ScaleErr):
    binwidth = (DFName[Target_Pt].values.max() - DFName[Target_Pt].values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName[Target_Pt], bins=nbins)
    sy, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=-(DFName[branchString]+DFName[Target_Pt]))
    sy2, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=(DFName[branchString]+DFName[Target_Pt])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    sy_resp, _resp = np.histogram(DFName[Target_Pt], bins=nbins, weights=-(DFName[branchString]+DFName[Target_Pt]))
    mean_resp = sy_resp / n
    plt.errorbar((_[1:] + _[:-1])/2, div0(std,mean_resp), marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName+': %8.2f $\pm$ %8.2f'%(mean, std), linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAResolutionOverNVertex_woutError_para_RC(branchString, labelName, errbars_shift, ScaleErr):
    binwidth = (DFName_nVertex.NVertex.values.max() - DFName_nVertex.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=-(DFName_nVertex[branchString]+DFName_nVertex[Target_Pt]))
    sy2, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(DFName_nVertex[branchString]+DFName_nVertex[Target_Pt])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    sy_resp, _resp = np.histogram(DFName_nVertex[Target_Pt], bins=nbinsVertex, weights=-(DFName_nVertex[branchString]+DFName_nVertex[Target_Pt]))
    mean_resp = sy_resp / n
    plt.errorbar((_[1:] + _[:-1])/2, div0(std,mean_resp), marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName+': %8.2f $\pm$ %8.2f'%(mean, std), linestyle="None", capsize=0,  color=colors[errbars_shift])


def plotMVAResolutionOverNVertex_woutError_perp(branchString, labelName, errbars_shift, ScaleErr, rangemin, rangemax):
    #IdxRange = (DFName_nVertex[Target_Pt]>=rangemin) & (DFName_nVertex[Target_Pt]<rangemin)
    DF_Resolution_pe_PV = DFName_nVertex[(DFName_nVertex[Target_Pt]>=rangemin) & (DFName_nVertex[Target_Pt]<rangemax)]
    binwidth = (50)/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DF_Resolution_pe_PV.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DF_Resolution_pe_PV.NVertex, bins=nbinsVertex, weights=-(DF_Resolution_pe_PV[branchString]))
    sy2, _ = np.histogram(DF_Resolution_pe_PV.NVertex, bins=nbinsVertex, weights=(DF_Resolution_pe_PV[branchString])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, std, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName+': %8.2f $\pm$ %8.2f'%(mean, std), linestyle="None", capsize=0,  color=colors_InOut[errbars_shift])

def plotMVAResolutionOverpTZ_woutError_perp_RC(branchString, labelName, errbars_shift, ScaleErr):
    binwidth = (DFName[Target_Pt].values.max() - DFName[Target_Pt].values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName[Target_Pt], bins=nbins)
    sy, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=-(DFName[branchString]))
    sy2, _ = np.histogram(DFName[Target_Pt], bins=nbins, weights=(DFName[branchString])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    sy_resp, _resp = np.histogram(DFName[Target_Pt], bins=nbins, weights=-(DFName[branchString])/DFName[Target_Pt])
    mean_resp = sy_resp / n
    plt.errorbar((_[1:] + _[:-1])/2, div0(std,mean_resp), marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName+': %8.2f $\pm$ %8.2f'%(mean, std), linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAResolutionOverNVertex_woutError_perp_RC(branchString, labelName, errbars_shift, ScaleErr):
    binwidth = (DFName_nVertex.NVertex.values.max() - DFName_nVertex.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(DFName_nVertex[branchString]))
    sy2, _ = np.histogram(DFName_nVertex.NVertex, bins=nbinsVertex, weights=(DFName_nVertex[branchString])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    sy_resp, _resp = np.histogram(DFName_nVertex[Target_Pt], bins=nbinsVertex, weights=-(DFName_nVertex[branchString]+DFName_nVertex[Target_Pt]))
    mean_resp = sy_resp / n
    plt.errorbar((_[1:] + _[:-1])/2, div0(std,mean_resp), marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName+': %8.2f $\pm$ %8.2f'%(mean, std), linestyle="None", capsize=0,  color=colors_InOut[errbars_shift])

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
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName+': %8.2f $\pm$ %8.2f'%(mean, std), linestyle="None", capsize=0,  color=colors[errbars_shift])
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
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName+': %8.2f $\pm$ %8.2f'%(mean, std), linestyle="None", capsize=0,  color=colors[errbars_shift])
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
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName+': %8.2f $\pm$ %8.2f'%(mean, std), linestyle="None", capsize=0,  color=colors[errbars_shift])
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
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName+': %8.2f $\pm$ %8.2f'%(mean, std), linestyle="None", capsize=0,  color=colors[errbars_shift])
    if errbars_shift==5: errbars_shift2 = 0
    elif errbars_shift==6: errbars_shift2 = 2
    else: errbars_shift2 = errbars_shift
    plt.errorbar((_[:-1]+(_[1:]-_[:-1])/3*errbars_shift2), mean, yerr=std*ScaleErr, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])





def getPlotsOutput(inputD, filesD, plotsD,DFName, DFName_nVertex, Target_Pt, Target_Phi):



    #Plot settings
    ScaleErr = 1
    NPlotsLines = 6
    MVA_NPlotsLines = 3
    pTRangeString_Err = '$%8.2f\ \mathrm{GeV} < |-\\vec{p}_T^Z| \leq %8.2f\ \mathrm{GeV}$ \n $\mathrm{\# Vertex} \leq 50$ \\ Error scaled: '%(pTMin,pTMax)+str(ScaleErr)
    pTRangeString= '$%8.2f\ \mathrm{GeV} < |-\\vec{p}_T^Z| \leq %8.2f\ \mathrm{GeV}$ \n $\mathrm{\# Vertex} \leq 50$'%(pTMin,pTMax)
    pTRangeString_low= pTRangeString_mid= pTRangeString_high= pTRangeString



    pTRangeString_Tresh = '$1\ \mathrm{GeV} < |-\\vec{p}_T^Z| \leq 200\ \mathrm{GeV}$ \n $\mathrm{\# Vertex} \leq 50$'
    pTRangeStringNVertex = pTRangeString
    if Target_Pt=='Boson_Pt':
        LegendTitle = 'MET definition: mean $\pm$ standard deviation'
    else:
        LegendTitle = '$\mathrm{Summer\ 17\ campaign}$' '\n'  '$\mathrm{Z \  \\rightarrow \ \\tau \\tau   \\rightarrow \ \mu \mu}$'
    NPlotsLines=7
    colors = cm.brg(np.linspace(0, 1, NPlotsLines))
    colors_InOut = cm.brg(np.linspace(0, 1, 8))
    colors = colors_InOut
    MVAcolors =  colors
    ylimResMin, ylimResMax = 7.5 , 50
    ylimResMVAMin_RC, ylimResMax_RC = 0 , 50

    PF_Delta_pT, PF_Delta_Phi = kar2pol(DFName['recoilslimmedMETs_LongZ'],DFName['recoilslimmedMETs_PerpZ'])





    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    plotMVAResponseOverpTZ_woutError('recoilslimmedMETsPuppi_LongZ', 'Puppi', 4, ScaleErr)
    plotMVAResponseOverpTZ_woutError('recoilslimmedMETs_LongZ', 'PF', 1, ScaleErr)
    plotMVAResponseOverpTZ_woutError('NN_LongZ', 'NN', 6, ScaleErr)
    plt.plot([pTMin, pTMax], [1, 1], color='k', linestyle='--', linewidth=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    #handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.xlabel('$|-\\vec{p}_T^Z| $ in GeV', fontsize=18)
    plt.ylabel('$\\langle \\frac{U_{\parallel}}{|-\\vec{p}_T^Z|} \\rangle$ ', fontsize=18)
    #plt.title('Response $U_{\parallel}$', fontsize=18)

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(0.95, 0.05), loc=4, borderaxespad=0., fontsize='large', title=LegendTitle, numpoints=1, framealpha=1.0	)
    plt.grid()
    plt.ylim(0.6, ResponseMaxErr)
    plt.xlim(pTMin, pTMax)
    plt.savefig("%sResponse_pT.png"%(plotsD), bbox_inches="tight")
    plt.close()






    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    plotMVAResolutionOverpTZ_woutError_para('recoilslimmedMETsPuppi_LongZ', 'Puppi', 4, ScaleErr)
    plotMVAResolutionOverpTZ_woutError_para('recoilslimmedMETs_LongZ', 'PF', 1, ScaleErr)
    plotMVAResolutionOverpTZ_woutError_para('NN_LongZ', 'NN', 6, ScaleErr)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    #handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.xlabel('$|-\\vec{p}_T^Z| $ in GeV', fontsize=18)
    plt.ylabel('$\sigma \\left( U_{\parallel} + p_T^Z \\right) $ in GeV', fontsize=18)
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{|-\\vec{p}_T^Z|} \\right) $ in GeV')
    #plt.title('Resolution $U_{\perp}$', fontsize=18)

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(0.95, 0.05), loc=4, borderaxespad=0., fontsize='large', title='MET definition: standard deviation', numpoints=1, framealpha=1.0	)
    plt.grid()
    plt.xlim(20, 200)
    plt.ylim(ylimResMVAMin, ylimResMVAMax)
    plt.savefig("%sResolution_para_pT.png"%(plotsD), bbox_inches="tight")
    plt.close()



    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    #Hist_Resolution_para('LongZCorrectedRecoil_LongZ', 'GBRT', 5, ScaleErr, 0, 200)
    Hist_Resolution_para('NN_LongZ', 'NN', 6, ScaleErr, 20, 40)
    Hist_Resolution_para('recoilslimmedMETsPuppi_LongZ', 'Puppi', 4, ScaleErr, 20, 40)
    Hist_Resolution_para('recoilslimmedMETs_LongZ', 'PF', 1, ScaleErr, 20, 40)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    #handles.insert(0,mpatches.Patch(color='none', label='$%8.2f\ \mathrm{GeV} < |-\\vec{p}_T^Z| \leq %8.2f\ \mathrm{GeV}$ \n $\mathrm{\# Vertex} \leq 50$'%(20,40)))

    plt.xlabel('$U_{\parallel} + p_T^Z$', fontsize=18)
    plt.ylabel('Counts', fontsize=18)
    #plt.ylabel('$\sigma \\left( \\frac{u_{\parallel}}{|-\\vec{p}_T^Z|} \\right) $ in GeV')
    #plt.title('Resolution $U_{\parallel}$', fontsize=18)

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1.00), loc=2, borderaxespad=0., fontsize='large', title=LegendTitle, numpoints=1, framealpha=1.0	)
    plt.grid()
    plt.xlim(ResolutionParaMin, ResolutionParaMax )
    plt.savefig("%sHist_Resolution_para_20_40.png"%(plotsD), bbox_inches="tight")
    plt.close()




    #######u perp ######
    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    plotMVAResolutionOverpTZ_woutError_perp('recoilslimmedMETsPuppi_PerpZ', 'Puppi', 4, ScaleErr)
    plotMVAResolutionOverpTZ_woutError_perp('recoilslimmedMETs_PerpZ', 'PF', 1, ScaleErr)
    plotMVAResolutionOverpTZ_woutError_perp('NN_PerpZ', 'NN', 6, ScaleErr)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    #handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.xlabel('$|-\\vec{p}_T^Z| $ in GeV', fontsize=18)
    plt.ylabel('$\sigma \\left( U_{\perp} \\right) $ in GeV', fontsize=18)
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{|-\\vec{p}_T^Z|} \\right) $ in GeV')
    #plt.title('Resolution $U_{\perp}$', fontsize=18)

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(0.95, 0.05), loc=4, borderaxespad=0., fontsize='large', title='MET definition: standard deviation', numpoints=1, framealpha=1.0	)
    plt.grid()
    plt.ylim(ylimResMVAMin, ylimResMVAMax)
    plt.xlim(20, 200)
    plt.savefig("%sResolution_perp_pT.png"%(plotsD), bbox_inches="tight")
    plt.close()






    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)


    Hist_LongZ_bin('recoilslimmedMETsPuppi_LongZ', 'Puppi MET',4, ScaleErr, 20, 40)
    Hist_LongZ_bin('recoilslimmedMETs_LongZ', 'PF MET', 1, ScaleErr, 20, 40)
    Hist_LongZ_bin('NN_LongZ', 'NN MET', 6, ScaleErr, 20, 40)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    #handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.ylabel('Counts', fontsize=18)
    plt.xlabel('$ U_{\parallel}   $ in GeV', fontsize=18)
    plt.xlim(HistLimMin,HistLimMax)

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1.00), loc=2, borderaxespad=0., fontsize='large', title=LegendTitle, numpoints=1, framealpha=1.0	)
    plt.grid()
    plt.savefig("%sHist_para_20_40.png"%(plotsD), bbox_inches="tight")
    plt.close()

    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)


    Hist_PerpZ('recoilslimmedMETsPuppi_PerpZ', 'Puppi MET', 4, ScaleErr)
    Hist_PerpZ('recoilslimmedMETs_PerpZ', 'PF MET', 1, ScaleErr)
    Hist_PerpZ('NN_PerpZ', 'NN MET', 6, ScaleErr)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    #handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.ylabel('Counts', fontsize=18)
    plt.xlabel('$ U_{\perp}  $ in GeV', fontsize=18)
    plt.xlim(HistLimMin,HistLimMax)


    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1.00), loc=2, borderaxespad=0., fontsize='large', title=LegendTitle, numpoints=1, framealpha=1.0	)
    plt.grid()
    plt.savefig("%sHist_perp.png"%(plotsD), bbox_inches="tight")
    plt.close()



    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)


    Hist_PerpZ_bin('recoilslimmedMETsPuppi_PerpZ', 'Puppi MET', 4, ScaleErr, 20, 40)
    Hist_PerpZ_bin('recoilslimmedMETs_PerpZ', 'PF MET', 1, ScaleErr, 20, 40)
    Hist_PerpZ_bin('NN_PerpZ', 'NN MET', 6, ScaleErr, 20, 40)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    #handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.ylabel('Counts', fontsize=18)
    plt.xlabel('$ U_{\perp}  $ in GeV', fontsize=18)
    plt.xlim(HistLimMin,HistLimMax)

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1.00), loc=2, borderaxespad=0., fontsize='large', title=LegendTitle, numpoints=1, framealpha=1.0	)
    plt.grid()
    plt.savefig("%sHist_perp_20_40.png"%(plotsD), bbox_inches="tight")
    plt.close()







    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    Histogram_Response('recoilslimmedMETsPuppi_LongZ', 'Puppi', 4, ScaleErr)
    Histogram_Response('NN_LongZ', 'NN', 6, ScaleErr)
    Histogram_Response('recoilslimmedMETs_LongZ', 'PF', 1, ScaleErr)


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    #handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    plt.xlabel('$  \\frac{U_{\parallel}}{|-\\vec{p}_T^Z|}} $', fontsize=18)
    plt.ylabel('Counts', fontsize=18)

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1.00), loc=2, borderaxespad=0., fontsize='large', title=LegendTitle, numpoints=1, framealpha=1.0	)
    plt.grid()
    plt.xlim(ResponseMin, ResponseMax)
    plt.savefig("%sHist_Response.png"%(plotsD), bbox_inches="tight")
    plt.close()

    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    Histogram_Response_bin('recoilslimmedMETsPuppi_LongZ', 'Puppi', 4, ScaleErr, 20, 40)
    Histogram_Response_bin('NN_LongZ', 'NN', 6, ScaleErr, 20, 40)
    Histogram_Response_bin('recoilslimmedMETs_LongZ', 'PF', 1, ScaleErr, 20, 40)


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    #handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    plt.xlabel('$  \\frac{U_{\parallel}}{|-\\vec{p}_T^Z|}} $', fontsize=18)
    plt.ylabel('Counts', fontsize=18)

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1.00), loc=2, borderaxespad=0., fontsize='large', title=LegendTitle, numpoints=1, framealpha=1.0	)
    plt.grid()
    plt.xlim(ResponseMin, ResponseMax)
    plt.savefig("%sHist_Response_20_40.png"%(plotsD), bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    inputDir = sys.argv[1]
    filesDir =  sys.argv[2]
    plotDir = sys.argv[3]
    plotDir = plotDir+"Hamburg/"
    PhysicsProcess = sys.argv[4]
    NN_mode = sys.argv[5]
    if PhysicsProcess == 'Tau':
        Target_Pt = 'Boson_Pt'
        Target_Phi = 'Boson_Phi'
        DFName_plain = loadData_woutGBRT(filesDir, inputDir, Target_Pt, Target_Phi, NN_mode, PhysicsProcess)
    else:
        Target_Pt = 'Boson_Pt'
        Target_Phi = 'Boson_Phi'
        DFName_plain2 = loadData_woutGBRT(filesDir, inputDir, Target_Pt, Target_Phi, NN_mode, PhysicsProcess)
        Test_Idx2 = h5py.File("%sTest_Idx_%s.h5" % (filesDir, NN_mode), "r")
        Test_Idx = Test_Idx2["Test_Idx"]
        #DFName_plain = DFName_plain2.iloc[Test_Idx]
        DFName_plain = DFName_plain2
    DFName=DFName_plain[DFName_plain[Target_Pt]<=pTMax]
    #DFName=DFName[DFName[Target_Pt]>pTMin]
    #DFName=DFName[DFName['NVertex']<=50]
    #DFName=DFName[DFName['NVertex']>=0]



    DFName_nVertex = DFName
    getPlotsOutput(inputDir, filesDir, plotDir, DFName, DFName_nVertex, Target_Pt, Target_Phi)
