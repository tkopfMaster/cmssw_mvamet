import numpy as np
import root_numpy as rnp
from scipy import optimize
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
import h5py
import sys
import time
from scipy.stats import chisquare

VertexReweight=False
pTMin, pTMax = 20, 200
nBinspT = int((pTMax-pTMin)*2)
VertexMax = 50
nBinsVertex = int((VertexMax)/10)

def div0( a, b ):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
    return c

def pol2kar_x(norm, phi):
    x = np.cos(phi)*norm
    return(x)
def pol2kar_y(norm, phi):
    y = np.sin(phi)*norm
    return(y)

def pol2kar(norm, phi):
    x = np.cos(phi)*norm
    y = np.sin(phi)*norm
    return(x, y)

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

def Spectrum(x, a, b):
    return a  * np.exp(x/(2*b))

def Spectrum2(x, a, b, c, n):
    return a * x * b**(b*n) * np.divide(np.exp(-n**2/2) , (b-n + np.divide(x,c))**(b*n) )

def getCurceParameters(x_data, y_data):
    params, params_covariance = optimize.curve_fit(Spectrum2, x_data, y_data, p0=[2, 2, 2, 2])
    return(params)

def loadData(fName, Target_Pt, Target_Phi, PhysicsProcess):
    tfile = ROOT.TFile(fName)

    for key in tfile.GetListOfKeys():
            print('key.GetName()', key.GetName())
            if key.GetName() == "MAPAnalyzer" and Target_Pt=='Boson_Pt':
                tree = key.ReadObj()
                print('tree', tree)
                treeName = 't'
                arrayName = rnp.root2array(fName,  branches=[Target_Pt, Target_Phi, 'NVertex' ,
                    'recoilslimmedMETsPuppi_Pt', 'recoilslimmedMETsPuppi_Phi', 'recoilslimmedMETsPuppi_sumEt',
                    'recoilslimmedMETs_Pt', 'recoilslimmedMETs_Phi', 'recoilslimmedMETs_sumEt',
                    'recoilpatpfNoPUMET_Pt','recoilpatpfNoPUMET_Phi', 'recoilpatpfNoPUMET_sumEt',
                    'recoilpatpfPUCorrectedMET_Pt', 'recoilpatpfPUCorrectedMET_Phi', 'recoilpatpfPUCorrectedMET_sumEt',
                    'recoilpatpfPUMET_Pt', 'recoilpatpfPUMET_Phi', 'recoilpatpfPUMET_sumEt',
                    'recoilpatpfTrackMET_Pt', 'recoilpatpfTrackMET_Phi', 'recoilpatpfTrackMET_sumEt' ,
                    'recoilslimmedMETs_LongZ', 'recoilslimmedMETs_PerpZ',
                    'recoilpatpfNoPUMET_LongZ','recoilpatpfNoPUMET_PerpZ',
                    'recoilpatpfPUCorrectedMET_LongZ', 'recoilpatpfPUCorrectedMET_PerpZ',
                    'recoilpatpfPUMET_LongZ', 'recoilpatpfPUMET_PerpZ'],)
                DFName = pd.DataFrame.from_records(arrayName.view(np.recarray))
            else:
                print('Target_Pt', Target_Pt)
                arrayName = rnp.root2array(fName, branches=[Target_Pt, Target_Phi, 'NVertex' ,
                    'recoilslimmedMETsPuppi_Pt', 'recoilslimmedMETsPuppi_Phi', 'recoilslimmedMETsPuppi_sumEt',
                    'recoilslimmedMETs_Pt', 'recoilslimmedMETs_Phi', 'recoilslimmedMETs_sumEt',
                    'recoilpatpfNoPUMET_Pt','recoilpatpfNoPUMET_Phi', 'recoilpatpfNoPUMET_sumEt',
                    'recoilpatpfPUCorrectedMET_Pt', 'recoilpatpfPUCorrectedMET_Phi', 'recoilpatpfPUCorrectedMET_sumEt',
                    'recoilpatpfPUMET_Pt', 'recoilpatpfPUMET_Phi', 'recoilpatpfPUMET_sumEt',
                    'recoilpatpfTrackMET_Pt', 'recoilpatpfTrackMET_Phi', 'recoilpatpfTrackMET_sumEt' ,
                    'recoilslimmedMETs_LongZ', 'recoilslimmedMETs_PerpZ',
                    'recoilpatpfNoPUMET_LongZ','recoilpatpfNoPUMET_PerpZ',
                    'recoilpatpfPUCorrectedMET_LongZ', 'recoilpatpfPUCorrectedMET_PerpZ',
                    'recoilpatpfPUMET_LongZ', 'recoilpatpfPUMET_PerpZ',
                    'recoilpatpfTrackMET_LongZ', 'recoilpatpfTrackMET_PerpZ',
                    'recoilslimmedMETsPuppi_LongZ', 'recoilslimmedMETsPuppi_PerpZ'],)
                DFName = pd.DataFrame.from_records(arrayName.view(np.recarray))
    if not tfile.GetListOfKeys():
        arrayName = rnp.root2array(fName, branches=[Target_Pt, Target_Phi, 'NVertex' ,
                    'recoilslimmedMETsPuppi_Pt', 'recoilslimmedMETsPuppi_Phi', 'recoilslimmedMETsPuppi_sumEt',
                    'recoilslimmedMETs_Pt', 'recoilslimmedMETs_Phi', 'recoilslimmedMETs_sumEt',
                    'recoilpatpfNoPUMET_Pt','recoilpatpfNoPUMET_Phi', 'recoilpatpfNoPUMET_sumEt',
                    'recoilpatpfPUCorrectedMET_Pt', 'recoilpatpfPUCorrectedMET_Phi', 'recoilpatpfPUCorrectedMET_sumEt',
                    'recoilpatpfPUMET_Pt', 'recoilpatpfPUMET_Phi', 'recoilpatpfPUMET_sumEt',
                    'recoilpatpfTrackMET_Pt', 'recoilpatpfTrackMET_Phi', 'recoilpatpfTrackMET_sumEt' ,
                    'recoilslimmedMETs_LongZ', 'recoilslimmedMETs_PerpZ',
                    'recoilpatpfNoPUMET_LongZ','recoilpatpfNoPUMET_PerpZ',
                    'recoilpatpfPUCorrectedMET_LongZ', 'recoilpatpfPUCorrectedMET_PerpZ',
                    'recoilpatpfPUMET_LongZ', 'recoilpatpfPUMET_PerpZ'
                    'recoilslimmedMETsPuppi_LongZ', 'recoilslimmedMETsPuppi_PerpZ'],)
        DFName = pd.DataFrame.from_records(arrayName.view(np.recarray))
    if Target_Pt=='Boson_Pt':
        arrayName = rnp.root2array(fName, branches=[Target_Pt, Target_Phi, 'NVertex' , 'genMet_Pt', 'genMet_Phi',
                    'recoilslimmedMETsPuppi_Pt', 'recoilslimmedMETsPuppi_Phi', 'recoilslimmedMETsPuppi_sumEt',
                    'recoilslimmedMETs_Pt', 'recoilslimmedMETs_Phi', 'recoilslimmedMETs_sumEt',
                    'recoilpatpfNoPUMET_Pt','recoilpatpfNoPUMET_Phi', 'recoilpatpfNoPUMET_sumEt',
                    'recoilpatpfPUCorrectedMET_Pt', 'recoilpatpfPUCorrectedMET_Phi', 'recoilpatpfPUCorrectedMET_sumEt',
                    'recoilpatpfPUMET_Pt', 'recoilpatpfPUMET_Phi', 'recoilpatpfPUMET_sumEt',
                    'recoilpatpfTrackMET_Pt', 'recoilpatpfTrackMET_Phi', 'recoilpatpfTrackMET_sumEt' ,
                    'recoilslimmedMETs_LongZ', 'recoilslimmedMETs_PerpZ',
                    'recoilpatpfTrackMET_LongZ', 'recoilpatpfTrackMET_PerpZ',
                    'recoilpatpfNoPUMET_LongZ','recoilpatpfNoPUMET_PerpZ',
                    'recoilpatpfPUCorrectedMET_LongZ', 'recoilpatpfPUCorrectedMET_PerpZ',
                    'recoilpatpfPUMET_LongZ', 'recoilpatpfPUMET_PerpZ',
                    'recoilslimmedMETsPuppi_LongZ', 'recoilslimmedMETsPuppi_PerpZ'],)
        DFName = pd.DataFrame.from_records(arrayName.view(np.recarray))
    '''
    if (PhysicsProcess=='Tau'):
        DFName['Boson_Pt'], DFName['Boson_Phi']=kar2pol(pol2kar_x(DFName[Target_Pt], DFName[Target_Phi])+
                                                        pol2kar_x(DFName['genMet_Pt'], DFName['genMet_Phi']),
                                                        pol2kar_y(DFName[Target_Pt], DFName[Target_Phi])+
                                                        pol2kar_y(DFName['genMet_Pt'], DFName['genMet_Phi']))
        print('isnan Boson pt prepare Input', sum(np.isnan(DFName['Boson_Pt'])))
        print('isnan Boson_Phi pt prepare Input', sum(np.isnan(DFName['Boson_Phi'])))
    '''
    DFName = DFName[DFName[Target_Pt]>pTMin]
    DFName = DFName[DFName[Target_Pt]<=pTMax]
    DFName = DFName[DFName['NVertex']<=VertexMax]

    return(DFName)


def loadData_training(fName, Target_Pt, Target_Phi, PhysicsProcess):
    tfile = ROOT.TFile(fName)

    for key in tfile.GetListOfKeys():
            print('key.GetName()', key.GetName())
            if key.GetName() == "MAPAnalyzer" and Target_Pt=='Boson_Pt':
                tree = key.ReadObj()
                print('tree', tree)
                treeName = 't'
                arrayName = rnp.root2array(fName,  branches=[Target_Pt, Target_Phi, 'NVertex' ,
                    'recoilslimmedMETsPuppi_Pt', 'recoilslimmedMETsPuppi_Phi', 'recoilslimmedMETsPuppi_sumEt',
                    'recoilslimmedMETs_Pt', 'recoilslimmedMETs_Phi', 'recoilslimmedMETs_sumEt',
                    'recoilpatpfNoPUMET_Pt','recoilpatpfNoPUMET_Phi', 'recoilpatpfNoPUMET_sumEt',
                    'recoilpatpfPUCorrectedMET_Pt', 'recoilpatpfPUCorrectedMET_Phi', 'recoilpatpfPUCorrectedMET_sumEt',
                    'recoilpatpfPUMET_Pt', 'recoilpatpfPUMET_Phi', 'recoilpatpfPUMET_sumEt',
                    'recoilpatpfTrackMET_Pt', 'recoilpatpfTrackMET_Phi', 'recoilpatpfTrackMET_sumEt' ,
                    'recoilslimmedMETs_LongZ', 'recoilslimmedMETs_PerpZ',
                    'recoilpatpfNoPUMET_LongZ','recoilpatpfNoPUMET_PerpZ',
                    'recoilpatpfPUCorrectedMET_LongZ', 'recoilpatpfPUCorrectedMET_PerpZ',
                    'recoilpatpfPUMET_LongZ', 'recoilpatpfPUMET_PerpZ'],)
                DFName = pd.DataFrame.from_records(arrayName.view(np.recarray))
            else:
                print('Target_Pt', Target_Pt)
                arrayName = rnp.root2array(fName, branches=[Target_Pt, Target_Phi, 'NVertex' ,
                    'recoilslimmedMETsPuppi_Pt', 'recoilslimmedMETsPuppi_Phi', 'recoilslimmedMETsPuppi_sumEt',
                    'recoilslimmedMETs_Pt', 'recoilslimmedMETs_Phi', 'recoilslimmedMETs_sumEt',
                    'recoilpatpfNoPUMET_Pt','recoilpatpfNoPUMET_Phi', 'recoilpatpfNoPUMET_sumEt',
                    'recoilpatpfPUCorrectedMET_Pt', 'recoilpatpfPUCorrectedMET_Phi', 'recoilpatpfPUCorrectedMET_sumEt',
                    'recoilpatpfPUMET_Pt', 'recoilpatpfPUMET_Phi', 'recoilpatpfPUMET_sumEt',
                    'recoilpatpfTrackMET_Pt', 'recoilpatpfTrackMET_Phi', 'recoilpatpfTrackMET_sumEt' ,
                    'recoilslimmedMETs_LongZ', 'recoilslimmedMETs_PerpZ',
                    'recoilpatpfNoPUMET_LongZ','recoilpatpfNoPUMET_PerpZ',
                    'recoilpatpfPUCorrectedMET_LongZ', 'recoilpatpfPUCorrectedMET_PerpZ',
                    'recoilpatpfPUMET_LongZ', 'recoilpatpfPUMET_PerpZ',
                    'recoilpatpfTrackMET_LongZ', 'recoilpatpfTrackMET_PerpZ',
                    'recoilslimmedMETsPuppi_LongZ', 'recoilslimmedMETsPuppi_PerpZ'],)
                DFName = pd.DataFrame.from_records(arrayName.view(np.recarray))
    if not tfile.GetListOfKeys():
        arrayName = rnp.root2array(fName, branches=[Target_Pt, Target_Phi, 'NVertex' ,
                    'recoilslimmedMETsPuppi_Pt', 'recoilslimmedMETsPuppi_Phi', 'recoilslimmedMETsPuppi_sumEt',
                    'recoilslimmedMETs_Pt', 'recoilslimmedMETs_Phi', 'recoilslimmedMETs_sumEt',
                    'recoilpatpfNoPUMET_Pt','recoilpatpfNoPUMET_Phi', 'recoilpatpfNoPUMET_sumEt',
                    'recoilpatpfPUCorrectedMET_Pt', 'recoilpatpfPUCorrectedMET_Phi', 'recoilpatpfPUCorrectedMET_sumEt',
                    'recoilpatpfPUMET_Pt', 'recoilpatpfPUMET_Phi', 'recoilpatpfPUMET_sumEt',
                    'recoilpatpfTrackMET_Pt', 'recoilpatpfTrackMET_Phi', 'recoilpatpfTrackMET_sumEt' ,
                    'recoilslimmedMETs_LongZ', 'recoilslimmedMETs_PerpZ',
                    'recoilpatpfNoPUMET_LongZ','recoilpatpfNoPUMET_PerpZ',
                    'recoilpatpfPUCorrectedMET_LongZ', 'recoilpatpfPUCorrectedMET_PerpZ',
                    'recoilpatpfPUMET_LongZ', 'recoilpatpfPUMET_PerpZ'
                    'recoilslimmedMETsPuppi_LongZ', 'recoilslimmedMETsPuppi_PerpZ'],)
        DFName = pd.DataFrame.from_records(arrayName.view(np.recarray))
    if Target_Pt=='Boson_Pt':
        arrayName = rnp.root2array(fName, branches=[Target_Pt, Target_Phi, 'NVertex' , 'genMet_Pt', 'genMet_Phi',
                    'recoilslimmedMETsPuppi_Pt', 'recoilslimmedMETsPuppi_Phi', 'recoilslimmedMETsPuppi_sumEt',
                    'recoilslimmedMETs_Pt', 'recoilslimmedMETs_Phi', 'recoilslimmedMETs_sumEt',
                    'recoilpatpfNoPUMET_Pt','recoilpatpfNoPUMET_Phi', 'recoilpatpfNoPUMET_sumEt',
                    'recoilpatpfPUCorrectedMET_Pt', 'recoilpatpfPUCorrectedMET_Phi', 'recoilpatpfPUCorrectedMET_sumEt',
                    'recoilpatpfPUMET_Pt', 'recoilpatpfPUMET_Phi', 'recoilpatpfPUMET_sumEt',
                    'recoilpatpfTrackMET_Pt', 'recoilpatpfTrackMET_Phi', 'recoilpatpfTrackMET_sumEt' ,
                    'recoilslimmedMETs_LongZ', 'recoilslimmedMETs_PerpZ',
                    'recoilpatpfTrackMET_LongZ', 'recoilpatpfTrackMET_PerpZ',
                    'recoilpatpfNoPUMET_LongZ','recoilpatpfNoPUMET_PerpZ',
                    'recoilpatpfPUCorrectedMET_LongZ', 'recoilpatpfPUCorrectedMET_PerpZ',
                    'recoilpatpfPUMET_LongZ', 'recoilpatpfPUMET_PerpZ',
                    'recoilslimmedMETsPuppi_LongZ', 'recoilslimmedMETsPuppi_PerpZ'],)
        DFName = pd.DataFrame.from_records(arrayName.view(np.recarray))
        print('isnan Boson pt prepare Input', sum(np.isnan(DFName['Boson_Pt'])))
        print('isnan Boson_Phi pt prepare Input', sum(np.isnan(DFName['Boson_Phi'])))
    DFName = DFName[DFName[Target_Pt]>pTMin]
    DFName = DFName[DFName[Target_Pt]<=pTMax]
    DFName = DFName[DFName['NVertex']<=VertexMax]

    return(DFName)

def loadData_proj(fName, Target_Pt, Target_Phi):
    treeName = 't'
    arrayName = rnp.root2array(fName, treeName, branches=[Target_Pt, Target_Phi, 'NVertex' ,   'recoilslimmedMETsPuppi_PerpZ', 'recoilslimmedMETsPuppi_LongZ', 'recoilslimmedMETs_PerpZ', 'recoilslimmedMETs_LongZ', 'recoilpatpfNoPUMET_PerpZ','recoilpatpfNoPUMET_LongZ','recoilpatpfPUCorrectedMET_PerpZ', 'recoilpatpfPUCorrectedMET_LongZ', 'recoilpatpfPUMET_PerpZ', 'recoilpatpfPUMET_LongZ', 'recoilpatpfTrackMET_PerpZ', 'recoilpatpfTrackMET_LongZ' ],)
    DFName = pd.DataFrame.from_records(arrayName.view(np.recarray))
    return(DFName)

def find_interval(x, intervals):
    for i in range (0, len(intervals)):
        if x < intervals[i]:
            return i-1
    return -1

def WeightsOverPt(weights, BosonPt):
    binwidth = (BosonPt.max() - BosonPt.min())/(nBinspT) #5MET-Definitionen
    n, _ = np.histogram(BosonPt, bins=nBinspT)
    sy, _ = np.histogram(BosonPt, bins=nBinspT, weights=weights)
    sy2, _ = np.histogram(BosonPt, bins=nBinspT, weights=(weights)**2)
    mean = sy / n
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label='Weights', linestyle="None", capsize=0,  color="red")

def WeightsOverVertex(weights, BosonPt):
    binwidth = (BosonPt.max() - BosonPt.min())/(nBinsVertex) #5MET-Definitionen
    n, _ = np.histogram(BosonPt, bins=nBinsVertex)
    sy, _ = np.histogram(BosonPt, bins=nBinsVertex, weights=weights)
    sy2, _ = np.histogram(BosonPt, bins=nBinsVertex, weights=(weights)**2)
    mean = sy / n
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label='Weights', linestyle="None", capsize=0,  color="red")

def Vertex(NVertex):
    binwidth = (NVertex.max() - NVertex.min())/(nBinsVertex) #5MET-Definitionen
    n, _ = np.histogram(NVertex, bins=nBinsVertex)
    plt.errorbar((_[1:] + _[:-1])/2, n, marker='.', xerr=(_[1:]-_[:-1])/2, label='#Vertex', linestyle="None", capsize=0,  color="red")


def getweightBosonPt(BosonPt):
    if BosonPt.max()>50:
        n, interval = np.histogram(BosonPt, bins=nBinspT)
    else:
        n, interval = np.histogram(BosonPt, bins=nBinsVertex)
    nmean = np.mean(n)
    print("Mean Boson Pt bin population", nmean)
    intervalmean = np.divide(interval[:-1]+interval[1:],2)
    print("intervalmean 0:10", intervalmean[0:10])
    y = np.divide(1.0, n)
    #y = np.divide(1, np.multiply(n, np.square(intervalmean )))
    weight=np.repeat(np.nan, len(BosonPt))
    for i in range(0,len(BosonPt)):
        weight[i]= y[find_interval(BosonPt.iloc[i], interval)]
        if i%100000==0:
            print(i)
    if not (len(BosonPt)==len(weight)) or np.any(np.isnan(weight)):
        raise EXCEPTION("Weights not same length as Boson Pt")
    return weight

def getweightNVertex(NVertex):
    n, interval = np.histogram(NVertex, bins=nBinsVertex)
    nmean = np.mean(n)
    print("Mean NVertex bin population", nmean)
    y = np.divide(1.0, n)
    weight=np.repeat(np.nan, len(NVertex))
    for i in range(0,len(NVertex)):
        weight[i]= y[find_interval(NVertex.iloc[i], interval)]
        if i%100000==0:
            print(i)
    if not (len(NVertex)==len(weight)) or np.any(np.isnan(weight)):
        raise EXCEPTION("Weights not same length as Boson Pt")
    return weight

def getInputs_xy_pTCut(DataF, outputD, PhysicsProcess, Target_Pt, Target_Phi, dset):
    pTCut = 0
    IdxpTCut = (DataF['Boson_Pt']>pTMin) & (DataF['Boson_Pt']<=pTMax) & (DataF['NVertex']<=VertexMax)
    start_time = time.time()
    n, _ = np.histogram(DataF['Boson_Pt'][IdxpTCut], bins=10000)
    if VertexReweight:
          weightsPV = getweightNVertex(DataF['NVertex'][IdxpTCut])
          weightspT = getweightBosonPt(DataF['Boson_Pt'][IdxpTCut])
          weights = np.multiply(weightspT, weightsPV)

          fig=plt.figure(figsize=(10,6))
          fig.patch.set_facecolor('white')
          ax = plt.subplot(111)

          WeightsOverVertex(weightsPV, DataF['NVertex'])

          box = ax.get_position()
          ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
          handles, labels = ax.get_legend_handles_labels()
          handles.insert(0,mpatches.Patch(color='none', label="$weight = \\frac{1}{\mathrm{counts}} $ \n with #Vertex nbins = %8.2f"%(nBinsVertex)))

          plt.xlabel('#Vertex ')
          plt.ylabel(' Weight ')
          #plt.title('Response $U_{\parallel}$')
          LegendTitle = '$\mathrm{Summer\ 17\ campaign}$' '\n'  '$\mathrm{Z \  \\rightarrow \ \mu \mu}$'
          ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
          plt.grid()
          plt.xlim(0, VertexMax)
          plt.savefig("%sWeightPV_PV.png"%(plotsD), bbox_inches="tight")
          plt.close()
          plt.yscale('log')

          fig=plt.figure(figsize=(10,6))
          fig.patch.set_facecolor('white')
          ax = plt.subplot(111)

          WeightsOverVertex(weights, DataF['NVertex'])

          box = ax.get_position()
          ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
          handles, labels = ax.get_legend_handles_labels()
          handles.insert(0,mpatches.Patch(color='none', label="$weight = \\frac{1}{\mathrm{counts}} $ \n with #Vertex nbins = %8.2f"%(nBinsVertex)))

          plt.xlabel('#Vertex ')
          plt.ylabel(' Weight ')
          #plt.title('Response $U_{\parallel}$')
          LegendTitle = '$\mathrm{Summer\ 17\ campaign}$' '\n'  '$\mathrm{Z \  \\rightarrow \ \mu \mu}$'
          ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
          plt.grid()
          plt.xlim(0, VertexMax)
          plt.savefig("%sWeight_PV.png"%(plotsD), bbox_inches="tight")
          plt.close()

          fig=plt.figure(figsize=(10,6))
          fig.patch.set_facecolor('white')
          ax = plt.subplot(111)

          Vertex(DataF['NVertex'][IdxpTCut])

          box = ax.get_position()
          ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
          handles, labels = ax.get_legend_handles_labels()
          handles.insert(0,mpatches.Patch(color='none', label="$weight = \\frac{1}{\mathrm{counts}} $ \n with #Vertex nbins = %8.2f"%(nBinsVertex)))

          plt.xlabel('#Vertex ')
          plt.ylabel(' Weight ')
          #plt.title('Response $U_{\parallel}$')
          LegendTitle = '$\mathrm{Summer\ 17\ campaign}$' '\n'  '$\mathrm{Z \  \\rightarrow \ \mu \mu}$'
          ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
          plt.grid()
          plt.xlim(0, VertexMax)
          plt.savefig("%sPV.png"%(plotsD), bbox_inches="tight")
          plt.close()

          fig=plt.figure(figsize=(10,6))
          fig.patch.set_facecolor('white')
          ax = plt.subplot(111)

          WeightsOverPt(weightspT, DataF['Boson_Pt'])

          box = ax.get_position()
          ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
          handles, labels = ax.get_legend_handles_labels()
          handles.insert(0,mpatches.Patch(color='none', label="$weight = \\frac{1}{\mathrm{counts}} $ \n with $p_T^Z$ nbins = %8.2f"%(nBinspT)))

          plt.xlabel('$ p_T^Z$ ')
          plt.ylabel(' Weight ')
          #plt.title('Response $U_{\parallel}$')
          LegendTitle = '$\mathrm{Summer\ 17\ campaign}$' '\n'  '$\mathrm{Z \  \\rightarrow \ \mu \mu}$'
          ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
          plt.grid()
          plt.xlim(pTMin, pTMax)
          plt.savefig("%sWeightpT_pT.png"%(plotsD), bbox_inches="tight")
          plt.close()


    else:
        #weights = np.divide(1.0, np.multiply(Spectrum(DataF['Boson_Pt'][IdxpTCut], getCurceParameters((_[:-1]+_[1:])/2,n)[0], getCurceParameters((_[:-1]+_[1:])/2,n)[1]), np.square(DataF['Boson_Pt'][IdxpTCut])))
        InvWeights = Spectrum2(DataF['Boson_Pt'][IdxpTCut], getCurceParameters((_[:-1]+_[1:])/2,n)[0], getCurceParameters((_[:-1]+_[1:])/2,n)[1], getCurceParameters((_[:-1]+_[1:])/2,n)[2], getCurceParameters((_[:-1]+_[1:])/2,n)[3])
        InvWeights_n = np.divide(1.0, Spectrum2((_[:-1]+_[1:])/2, getCurceParameters((_[:-1]+_[1:])/2,n)[0], getCurceParameters((_[:-1]+_[1:])/2,n)[1], getCurceParameters((_[:-1]+_[1:])/2,n)[2],getCurceParameters((_[:-1]+_[1:])/2,n)[3]))
        weights = np.divide(1.0, InvWeights)
        ScaleSpectrum= np.divide(len(DataF['Boson_Pt'][IdxpTCut]), np.sum(Spectrum(DataF['Boson_Pt'][IdxpTCut], getCurceParameters((_[:-1]+_[1:])/2,n)[0], getCurceParameters((_[:-1]+_[1:])/2,n)[1])))
        #print("Scale Spectrum", ScaleSpectrum)
        #weights = Spectrum(DataF['Boson_Pt'][IdxpTCut], getCurceParameters((_[:-1]+_[1:])/2,n)[0], getCurceParameters((_[:-1]+_[1:])/2,n)[1])
        #weights = getweightBosonPt(DataF['Boson_Pt'][IdxpTCut])
        #print("chi square  summe (counts-fitValue(ptz))/sqrt(counts)", np.sum(div0((n-InvWeights_n)**2,n)))
        #print("Chi Square of Fit n observed", chisquare(np.divide(n, np.sum(n)), f_exp=np.divide(InvWeights_n, np.sum(InvWeights_n))))
        #print("Chi Square of Fit InvWeights_n observed", chisquare(np.divide(InvWeights_n, np.sum(InvWeights_n)), f_exp=np.divide(n, np.sum(n))))


    #uniform = n * Spectrum((_[:-1]+_[1:])/2, getCurceParameters((_[:-1]+_[1:])/2,n)[0], getCurceParameters((_[:-1]+_[1:])/2,n)[1])
    uniform = n * InvWeights_n
    plt.plot((_[:-1]+_[1:])/2, uniform, label="n*pf")
    #plt.plot((_[:-1]+_[1:])/2, n*getweightBosonPt((_[:-1]+_[1:])/2), label="binned weight")
    #plt.hist(weights, bins=nBinspT , lw=3, label="Training loss")
    plt.xlabel("Weights"), plt.ylabel("Counts")
    plt.legend()
    plt.savefig("%sWeights.png"%(plotsD))
    plt.close()

    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    plt.scatter((_[:-1]+_[1:])/2, div0(1,n), label='Data', alpha=0.2, s=1)
    plt.plot(np.arange(pTMin, pTMax, 0.01), np.divide(1.0, Spectrum2(np.arange(pTMin, pTMax, 0.01), getCurceParameters((_[:-1]+_[1:])/2,n)[0],
                                                                     getCurceParameters((_[:-1]+_[1:])/2,n)[1], getCurceParameters((_[:-1]+_[1:])/2,n)[2],getCurceParameters((_[:-1]+_[1:])/2,n)[3])),
             label='Fitted function', color = "r")


    plt.xlabel("$p_T^Z$"), plt.ylabel("1/Counts")
    plt.legend(loc='best')
    plt.xlim(pTMin, pTMax)
    plt.savefig("%sFittedFunctionWeight.png"%(plotsD), bbox_inches="tight")
    plt.close()

    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    #plt.scatter((_[:-1]+_[1:])/2, n, label='Data', alpha=0.2, s=1)
    plt.plot((_[:-1]+_[1:])/2, Spectrum2((_[:-1]+_[1:])/2, getCurceParameters((_[:-1]+_[1:])/2,n)[0], getCurceParameters((_[:-1]+_[1:])/2,n)[1], getCurceParameters((_[:-1]+_[1:])/2,n)[2],getCurceParameters((_[:-1]+_[1:])/2,n)[3]),
             label='Fitted function', color = "r")



    plt.xlabel("$p_T^Z$"), plt.ylabel("Counts")
    plt.legend(loc='best')
    plt.xlim(pTMin, pTMax)
    plt.savefig("%sFittedFunction.png"%(plotsD), bbox_inches="tight")
    plt.close()

    print("Values of pT fitted Function", np.divide(1.0, Spectrum2(DataF['Boson_Pt'][IdxpTCut], getCurceParameters((_[:-1]+_[1:])/2,n)[0], getCurceParameters((_[:-1]+_[1:])/2,n)[1], getCurceParameters((_[:-1]+_[1:])/2,n)[2],getCurceParameters((_[:-1]+_[1:])/2,n)[3])))
    print("mean input", np.mean(n))
    print("max fitted", np.mean(Spectrum2(np.arange(pTMin, pTMax, 0.01), getCurceParameters((_[:-1]+_[1:])/2,n)[0],
                                                                     getCurceParameters((_[:-1]+_[1:])/2,n)[1], getCurceParameters((_[:-1]+_[1:])/2,n)[2],getCurceParameters((_[:-1]+_[1:])/2,n)[3])))



    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    WeightsOverPt(weights, DataF['Boson_Pt'])

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0,mpatches.Patch(color='none', label="$weight = \\frac{1}{\mathrm{counts}} $ \n with $p_T^Z$ nbins = %8.2f"%(nBinspT)))

    plt.xlabel('$ p_T^Z$ ')
    plt.ylabel(' Weight ')
    #plt.title('Response $U_{\parallel}$')
    LegendTitle = '$\mathrm{Summer\ 17\ campaign}$' '\n'  '$\mathrm{Z \  \\rightarrow \ \mu \mu}$'
    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    plt.xlim(pTMin, pTMax)
    plt.savefig("%sWeight_pT.png"%(plotsD), bbox_inches="tight")
    plt.close()


    end_time = time.time()
    print("Gewichte bestimmen hat {0} Sekunden gedauert".format(end_time-start_time))
    print('DataF[recoilslimmedMETs_Pt]', DataF['recoilslimmedMETs_Pt'].shape)
    dset_NVertex = dset.create_dataset("NVertex",  dtype='f',
        data=[DataF['NVertex'][IdxpTCut]])

    dset_PF = dset.create_dataset("PF",  dtype='f',
        data=[pol2kar_x(DataF['recoilslimmedMETs_Pt'][IdxpTCut], DataF['recoilslimmedMETs_Phi'][IdxpTCut]),
        pol2kar_y(DataF['recoilslimmedMETs_Pt'][IdxpTCut], DataF['recoilslimmedMETs_Phi'][IdxpTCut]),
        DataF['recoilslimmedMETs_sumEt'][IdxpTCut]])

    dset_Track = dset.create_dataset("Track",  dtype='f',
        data=[ pol2kar_x(DataF['recoilpatpfTrackMET_Pt'][IdxpTCut], DataF['recoilpatpfTrackMET_Phi'][IdxpTCut]),
        pol2kar_y(DataF['recoilpatpfTrackMET_Pt'][IdxpTCut], DataF['recoilpatpfTrackMET_Phi'][IdxpTCut]),
        DataF['recoilpatpfTrackMET_sumEt'][IdxpTCut]])


    dset_NoPU = dset.create_dataset("NoPU",  dtype='f',
        data=[pol2kar_x(DataF['recoilpatpfNoPUMET_Pt'][IdxpTCut], DataF['recoilpatpfNoPUMET_Phi'][IdxpTCut]),
        pol2kar_y(DataF['recoilpatpfNoPUMET_Pt'][IdxpTCut], DataF['recoilpatpfNoPUMET_Phi'][IdxpTCut]),
        DataF['recoilpatpfNoPUMET_sumEt'][IdxpTCut]])


    dset_PUCorrected = dset.create_dataset("PUCorrected",  dtype='f',
        data=[pol2kar_x(DataF['recoilpatpfPUCorrectedMET_Pt'][IdxpTCut], DataF['recoilpatpfPUCorrectedMET_Phi'][IdxpTCut]),
        pol2kar_y(DataF['recoilpatpfPUCorrectedMET_Pt'][IdxpTCut], DataF['recoilpatpfPUCorrectedMET_Phi'][IdxpTCut]),
        DataF['recoilpatpfPUCorrectedMET_sumEt'][IdxpTCut]])


    dset_PU = dset.create_dataset("PU",  dtype='f',
        data=[pol2kar_x(DataF['recoilpatpfPUMET_Pt'][IdxpTCut], DataF['recoilpatpfPUMET_Phi'][IdxpTCut]),
        pol2kar_y(DataF['recoilpatpfPUMET_Pt'][IdxpTCut], DataF['recoilpatpfPUMET_Phi'][IdxpTCut]),
        DataF['recoilpatpfPUMET_sumEt'][IdxpTCut]])


    dset_Puppi = dset.create_dataset("Puppi",  dtype='f',
        data=[pol2kar_x(DataF['recoilslimmedMETsPuppi_Pt'][IdxpTCut], DataF['recoilslimmedMETsPuppi_Phi'][IdxpTCut]),
        pol2kar_y(DataF['recoilslimmedMETsPuppi_Pt'][IdxpTCut], DataF['recoilslimmedMETsPuppi_Phi'][IdxpTCut]),
        DataF['recoilslimmedMETsPuppi_sumEt'][IdxpTCut]])



    dset_Target = dset.create_dataset("Target",  dtype='f',
            data=[-pol2kar_x(DataF[Target_Pt][IdxpTCut], DataF[Target_Phi][IdxpTCut]),
            -pol2kar_y(DataF[Target_Pt][IdxpTCut], DataF[Target_Phi][IdxpTCut])])
    dset_Weight = dset.create_dataset("weights",  dtype='f',
            data=[weights])
    dset.close()



def getInputs_xy_Test(DataF, outputD, PhysicsProcess, Target_Pt, Target_Phi, dset):
    pTCut = 0
    IdxpTCut = (DataF['Boson_Pt']>pTMin) & (DataF['Boson_Pt']<=pTMax) & (DataF['NVertex']<=VertexMax)
    x_Test = np.repeat(0.4,6049481)
    y_Test = x_Test
    print('DataF[recoilslimmedMETs_Pt]', DataF['recoilslimmedMETs_Pt'].shape)
    dset_PF = dset.create_dataset("PF",  dtype='f',
        data=[x_Test, y_Test])
    ''' ,
        DataF['recoilslimmedMETs_sumEt'][IdxpTCut],
        DataF['NVertex'][IdxpTCut]])
    '''

    dset_Track = dset.create_dataset("Track",  dtype='f',
        data=[x_Test, y_Test])
    ''',
        DataF['recoilpatpfTrackMET_sumEt'][IdxpTCut]])
    '''

    dset_NoPU = dset.create_dataset("NoPU",  dtype='f',
        data=[x_Test, y_Test])
    ''',
        DataF['recoilpatpfNoPUMET_sumEt'][IdxpTCut]])'''

    dset_PUCorrected = dset.create_dataset("PUCorrected",  dtype='f',
        data=[x_Test, y_Test])
    ''',
        DataF['recoilpatpfPUCorrectedMET_sumEt'][IdxpTCut]])'''

    dset_PU = dset.create_dataset("PU",  dtype='f',
        data=[x_Test, y_Test])
    ''',
        DataF['recoilpatpfPUMET_sumEt'][IdxpTCut]])'''

    dset_Puppi = dset.create_dataset("Puppi",  dtype='f',
        data=[x_Test, y_Test])
    ''',
        DataF['recoilslimmedMETsPuppi_sumEt'][IdxpTCut]])'''
    dset_NoPV = dset.create_dataset("NVertex",  dtype='f',data=[DataF['NVertex'][IdxpTCut]] )


    dset_Target = dset.create_dataset("Target",  dtype='f',
            data=[np.repeat(1, 6049481), np.repeat(1, 6049481)])

    dset.close()



def getInputs_xy(DataF, outputD, PhysicsProcess, Target_Pt, Target_Phi, dset):
    pTCut = 0
    IdxpTCut = (DataF['Boson_Pt']>pTMin) & (DataF['Boson_Pt']<=pTMax) & (DataF['NVertex']<=VertexMax)
    print('sum(IdxpTCut)', IdxpTCut)
    print('DataF[recoilslimmedMETs_Pt]', DataF['recoilslimmedMETs_Pt'].shape)

    set_BosonPt = dset.create_dataset("Boson_Pt",  dtype='f',
        data=[ DataF['Boson_Pt'][IdxpTCut]])

    dset_PF = dset.create_dataset("PF",  dtype='f',
        data=[pol2kar_x(DataF['recoilslimmedMETs_Pt'][IdxpTCut], DataF['recoilslimmedMETs_Phi'][IdxpTCut]),
        pol2kar_y(DataF['recoilslimmedMETs_Pt'][IdxpTCut], DataF['recoilslimmedMETs_Phi'][IdxpTCut]),
        DataF['recoilslimmedMETs_sumEt'][IdxpTCut]])

    dset_Track = dset.create_dataset("Track",  dtype='f',
        data=[ pol2kar_x(DataF['recoilpatpfTrackMET_Pt'][IdxpTCut], DataF['recoilpatpfTrackMET_Phi'][IdxpTCut]),
        pol2kar_y(DataF['recoilpatpfTrackMET_Pt'][IdxpTCut], DataF['recoilpatpfTrackMET_Phi'][IdxpTCut]),
        DataF['recoilpatpfTrackMET_sumEt'][IdxpTCut]])

    dset_NoPU = dset.create_dataset("NoPU",  dtype='f',
        data=[pol2kar_x(DataF['recoilpatpfNoPUMET_Pt'][IdxpTCut], DataF['recoilpatpfNoPUMET_Phi'][IdxpTCut]),
        pol2kar_y(DataF['recoilpatpfNoPUMET_Pt'][IdxpTCut], DataF['recoilpatpfNoPUMET_Phi'][IdxpTCut]),
        DataF['recoilpatpfNoPUMET_sumEt'][IdxpTCut]])

    dset_PUCorrected = dset.create_dataset("PUCorrected",  dtype='f',
        data=[pol2kar_x(DataF['recoilpatpfPUCorrectedMET_Pt'][IdxpTCut], DataF['recoilpatpfPUCorrectedMET_Phi'][IdxpTCut]),
        pol2kar_y(DataF['recoilpatpfPUCorrectedMET_Pt'][IdxpTCut], DataF['recoilpatpfPUCorrectedMET_Phi'][IdxpTCut]),
        DataF['recoilpatpfPUCorrectedMET_sumEt'][IdxpTCut]])

    dset_PU = dset.create_dataset("PU",  dtype='f',
        data=[pol2kar_x(DataF['recoilpatpfPUMET_Pt'][IdxpTCut], DataF['recoilpatpfPUMET_Phi'][IdxpTCut]),
        pol2kar_y(DataF['recoilpatpfPUMET_Pt'][IdxpTCut], DataF['recoilpatpfPUMET_Phi'][IdxpTCut]),
        DataF['recoilpatpfPUMET_sumEt'][IdxpTCut]])

    dset_Puppi = dset.create_dataset("Puppi",  dtype='f',
        data=[pol2kar_x(DataF['recoilslimmedMETsPuppi_Pt'][IdxpTCut], DataF['recoilslimmedMETsPuppi_Phi'][IdxpTCut]),
        pol2kar_y(DataF['recoilslimmedMETsPuppi_Pt'][IdxpTCut], DataF['recoilslimmedMETsPuppi_Phi'][IdxpTCut]),
        DataF['recoilslimmedMETsPuppi_sumEt'][IdxpTCut]])

    dset_NoPV = dset.create_dataset("NVertex",  dtype='f',data=[DataF['NVertex'][IdxpTCut]] )


    dset_Target = dset.create_dataset("Target",  dtype='f',
            data=[-pol2kar_x(DataF[Target_Pt][IdxpTCut], DataF[Target_Phi][IdxpTCut]),
            -pol2kar_y(DataF[Target_Pt][IdxpTCut], DataF[Target_Phi][IdxpTCut])])

    dset.close()

def getInputs_xy_Test2(DataF, outputD, PhysicsProcess, Target_Pt, Target_Phi, dset):
    pTCut = 0
    IdxpTCut = (DataF['Boson_Pt']>pTMin) & (DataF['Boson_Pt']<=pTMax) & (DataF['NVertex']<=VertexMax)
    print('sum(IdxpTCut)', IdxpTCut)
    print('DataF[recoilslimmedMETs_Pt]', DataF['recoilslimmedMETs_Pt'].shape)
    Test_x, Test_y = np.repeat(0.4,6049481), np.repeat(0.4,6049481)
    set_BosonPt = dset.create_dataset("Boson_Pt",  dtype='f',
        data=[ np.repeat(np.sqrt(2),6049481)])

    dset_PF = dset.create_dataset("PF",  dtype='f',
        data=[Test_x, Test_y] )
    ''' ,
        DataF['recoilslimmedMETs_sumEt'][IdxpTCut],
        DataF['NVertex'][IdxpTCut]][IdxpTCut])
    '''

    dset_Track = dset.create_dataset("Track",  dtype='f',
        data=[Test_x, Test_y])
    ''',
        DataF['recoilpatpfTrackMET_sumEt'][IdxpTCut]][IdxpTCut])
    '''

    dset_NoPU = dset.create_dataset("NoPU",  dtype='f',
        data=[Test_x, Test_y])
    ''',
        DataF['recoilpatpfNoPUMET_sumEt'][IdxpTCut]][IdxpTCut])'''

    dset_PUCorrected = dset.create_dataset("PUCorrected",  dtype='f',
        data=[Test_x, Test_y])
    ''',
        DataF['recoilpatpfPUCorrectedMET_sumEt'][IdxpTCut]][IdxpTCut])'''

    dset_PU = dset.create_dataset("PU",  dtype='f',
        data=[Test_x, Test_y])
    ''',
        DataF['recoilpatpfPUMET_sumEt'][IdxpTCut]][IdxpTCut])'''

    dset_Puppi = dset.create_dataset("Puppi",  dtype='f',
        data=[Test_x, Test_y])
    ''',
        DataF['recoilslimmedMETsPuppi_sumEt'][IdxpTCut]][IdxpTCut])'''
    dset_NoPV = dset.create_dataset("NVertex",  dtype='f',data=[DataF['NVertex'][IdxpTCut]] )


    dset_Target = dset.create_dataset("Target",  dtype='f',
            data=[np.repeat(1, 6049481), np.repeat(1, 6049481)])

    dset.close()

def getInputs(fName, fileName_apply, NN_mode, outputD, PhysicsProcess, Target_Pt, Target_Phi):
    if NN_mode == 'xy':
            print("funktioniert 1")
            Data = loadData_training(fName,  Target_Pt, Target_Phi, PhysicsProcess)
            #Inputs = getInputs_xy_Test(Data, outputD, PhysicsProcess, Target_Pt, Target_Phi, writeInputs_training)
            print("funktioniert 2")
            Inputs = getInputs_xy_pTCut(Data, outputD, PhysicsProcess, Target_Pt, Target_Phi, writeInputs_training)
            print("funktioniert 3")
            Data_apply = loadData(fileName_apply,  Target_Pt, Target_Phi, PhysicsProcess)
            #Inputs_apply = getInputs_xy_Test2(Data_apply, outputD, PhysicsProcess, Target_Pt, Target_Phi, writeInputs_apply)
            print("funktioniert 4")
            Inputs_apply = getInputs_xy(Data_apply, outputD, PhysicsProcess, Target_Pt, Target_Phi, writeInputs_apply)





if __name__ == "__main__":
    fileName = sys.argv[1]
    outputDir = sys.argv[2]
    NN_mode = sys.argv[3]
    plotsD = sys.argv[4]
    PhysicsProcess = sys.argv[5]
    fileName_apply = sys.argv[6]
    Target_Pt = 'Boson_Pt'
    Target_Phi = 'Boson_Phi'
    print(fileName)
    writeInputs_training = h5py.File("%sNN_Input_training_%s.h5"%(outputDir,NN_mode), "w")
    writeInputs_apply = h5py.File("%sNN_Input_apply_%s.h5"%(outputDir,NN_mode), "w")
    getInputs(fileName, fileName_apply, NN_mode, plotsD, PhysicsProcess, Target_Pt, Target_Phi)
