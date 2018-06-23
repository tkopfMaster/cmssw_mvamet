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


NN_Mode='kart'


def loadData(fName):
    treeName = 't'
    arrayName = rnp.root2array(fName, branches=['Boson_Pt', 'Boson_Phi', 'NVertex' ,
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
#Data settings

def pol2kar_x(norm, phi):
    x = np.cos(phi[:])*norm[:]
    return(x)
def pol2kar_y(norm, phi):
    y = np.sin(phi[:])*norm[:]
    return(y)
def pol2kar(norm, phi):
    return (pol2kar_x(norm, phi), pol2kar_y(norm, phi))


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


def getInputs_kart(DataF):
    dset_PF1 = writeInputs.create_dataset("PF_1",  dtype='f',
        data=pol2kar_x(DataF['recoilslimmedMETs_Pt'], DataF['recoilslimmedMETs_Phi']))
    dset_PF2 = writeInputs.create_dataset("PF_2",  dtype='f',
        data=pol2kar_y(DataF['recoilslimmedMETs_Pt'], DataF['recoilslimmedMETs_Phi']))
    dset_PF3 = writeInputs.create_dataset("PF_3",  dtype='f',
        data=DataF['recoilslimmedMETs_sumEt'])
    dset_Track1 = writeInputs.create_dataset("Track_1",  dtype='f',
        data=pol2kar_x(DataF['recoilpatpfTrackMET_Pt'], DataF['recoilpatpfTrackMET_Phi']))
    dset_Track2 = writeInputs.create_dataset("Track_2",  dtype='f',
        data=pol2kar_y(DataF['recoilpatpfTrackMET_Pt'], DataF['recoilpatpfTrackMET_Phi']))
    dset_Track3 = writeInputs.create_dataset("Track_3",  dtype='f',
        data=DataF['recoilpatpfTrackMET_sumEt'])
    dset_NoPU1 = writeInputs.create_dataset("NoPU_1",  dtype='f',
        data=pol2kar_x(DataF['recoilpatpfNoPUMET_Pt'], DataF['recoilpatpfNoPUMET_Phi']))
    dset_NoPU2 = writeInputs.create_dataset("NoPU_2",  dtype='f',
        data=pol2kar_y(DataF['recoilpatpfNoPUMET_Pt'], DataF['recoilpatpfNoPUMET_Phi']))
    dset_NoPU3 = writeInputs.create_dataset("NoPU_3",  dtype='f',
        data=DataF['recoilpatpfNoPUMET_sumEt'])
    dset_PUCorrected1 = writeInputs.create_dataset("PUCorrected_1",  dtype='f',
        data=pol2kar_x(DataF['recoilpatpfPUCorrectedMET_Pt'], DataF['recoilpatpfPUCorrectedMET_Phi']))
    dset_PUCorrected2 = writeInputs.create_dataset("PUCorrected_2",  dtype='f',
        data=pol2kar_y(DataF['recoilpatpfPUCorrectedMET_Pt'], DataF['recoilpatpfPUCorrectedMET_Phi']))
    dset_PUCorrected3 = writeInputs.create_dataset("PUCorrected_3",  dtype='f',
        data=DataF['recoilpatpfPUCorrectedMET_sumEt'])
    dset_PU1 = writeInputs.create_dataset("PU_1",  dtype='f',
        data=pol2kar_x(DataF['recoilpatpfPUMET_Pt'], DataF['recoilpatpfPUMET_Phi']))
    dset_PU2 = writeInputs.create_dataset("PU_2",  dtype='f',
        data=pol2kar_y(DataF['recoilpatpfPUMET_Pt'], DataF['recoilpatpfPUMET_Phi']))
    dset_PU3 = writeInputs.create_dataset("PU_3",  dtype='f',
        data=DataF['recoilpatpfPUMET_sumEt'])
    dset_Puppi1 = writeInputs.create_dataset("Puppi_1",  dtype='f',
        data=pol2kar_x(DataF['recoilslimmedMETsPuppi_Pt'], DataF['recoilslimmedMETsPuppi_Phi']))
    dset_Puppi2 = writeInputs.create_dataset("Puppi_2",  dtype='f',
        data=pol2kar_y(DataF['recoilslimmedMETsPuppi_Pt'], DataF['recoilslimmedMETsPuppi_Phi']))
    dset_Puppi3 = writeInputs.create_dataset("Puppi_3",  dtype='f',
        data=DataF['recoilslimmedMETsPuppi_sumEt'])


    dset_Target1 = writeInputs.create_dataset("Target_1",  dtype='f',
        data=pol2kar_x(DataF['Boson_Pt'], DataF['Boson_Phi']))
    dset_Target2 =   writeInputs.create_dataset("Target_2",  dtype='f',
        data=pol2kar_y(DataF['Boson_Pt'], DataF['Boson_Phi']))
    writeInputs.close()

def getInputs_absCorr(DataF):
    return()

def getInputs_proj(DataF):
    return()


def getInputs(fName):

    if NN_Mode == 'kart':
        Data = loadData(fName)
        Inputs = getInputs_kart(Data)
    elif NN_Mode == 'absCorr':
        Data = loadData(fName)
        Inputs = getInputs_absCorr(Data)
    else:
        Data = loadData_proj(fName)
        Inputs =  getInputs_proj(Data)

if __name__ == "__main__":
	fileName = sys.argv[1]
	outputDir = sys.argv[2]
	print(fileName)
	writeInputs = h5py.File("%sNN_Input.h5"%outputDir, "w")
	getInputs(fileName)
    #getTarge
