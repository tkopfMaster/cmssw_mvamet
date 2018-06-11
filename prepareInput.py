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

NN_mode='xyr'

def div0( a, b ):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
    return c

def loadData(fName):
    tfile = ROOT.TFile(fName)
    for key in tfile.GetListOfKeys():
            if key.GetName() == "MAPAnalyzer/t":
                tree = key.ReadObj()
    arrayName = rnp.tree2array(tree, branches=['Boson_Pt', 'Boson_Phi', 'NVertex' ,
        'recoilslimmedMETsPuppi_Pt', 'recoilslimmedMETsPuppi_Phi', 'recoilslimmedMETsPuppi_sumEt',
        'recoilslimmedMETs_Pt', 'recoilslimmedMETs_Phi', 'recoilslimmedMETs_sumEt',
        'recoilpatpfNoPUMET_Pt','recoilpatpfNoPUMET_Phi', 'recoilpatpfNoPUMET_sumEt',
        'recoilpatpfPUCorrectedMET_Pt', 'recoilpatpfPUCorrectedMET_Phi', 'recoilpatpfPUCorrectedMET_sumEt',
        'recoilpatpfPUMET_Pt', 'recoilpatpfPUMET_Phi', 'recoilpatpfPUMET_sumEt',
        'recoilpatpfTrackMET_Pt', 'recoilpatpfTrackMET_Phi', 'recoilpatpfTrackMET_sumEt' ],)


    '''treeName = 't'
    arrayName = rnp.root2array(fName, branches=['Boson_Pt', 'Boson_Phi', 'NVertex' ,
        'recoilslimmedMETsPuppi_Pt', 'recoilslimmedMETsPuppi_Phi', 'recoilslimmedMETsPuppi_sumEt',
        'recoilslimmedMETs_Pt', 'recoilslimmedMETs_Phi', 'recoilslimmedMETs_sumEt',
        'recoilpatpfNoPUMET_Pt','recoilpatpfNoPUMET_Phi', 'recoilpatpfNoPUMET_sumEt',
        'recoilpatpfPUCorrectedMET_Pt', 'recoilpatpfPUCorrectedMET_Phi', 'recoilpatpfPUCorrectedMET_sumEt',
        'recoilpatpfPUMET_Pt', 'recoilpatpfPUMET_Phi', 'recoilpatpfPUMET_sumEt',
        'recoilpatpfTrackMET_Pt', 'recoilpatpfTrackMET_Phi', 'recoilpatpfTrackMET_sumEt' ],)
    '''
    DFName = pd.DataFrame.from_records(arrayName.view(np.recarray))
    return(DFName)

def loadData_proj(fName):
    treeName = 't'
    arrayName = rnp.root2array(fName, treeName, branches=['Boson_Pt', 'Boson_Phi', 'NVertex' ,   'recoilslimmedMETsPuppi_PerpZ', 'recoilslimmedMETsPuppi_LongZ', 'recoilslimmedMETs_PerpZ', 'recoilslimmedMETs_LongZ', 'recoilpatpfNoPUMET_PerpZ','recoilpatpfNoPUMET_LongZ','recoilpatpfPUCorrectedMET_PerpZ', 'recoilpatpfPUCorrectedMET_LongZ', 'recoilpatpfPUMET_PerpZ', 'recoilpatpfPUMET_LongZ', 'recoilpatpfTrackMET_PerpZ', 'recoilpatpfTrackMET_LongZ' ],)
    DFName = pd.DataFrame.from_records(arrayName.view(np.recarray))
    return(DFName)
#Data settings

def pol2kar_x(norm, phi):
    x = []
    x = np.sin(phi[:])*norm[:]
    return(x)
def pol2kar_y(norm, phi):
    y = []
    y = np.cos(phi[:])*norm[:]
    return(y)

def getInputs_xy(DataF):
    dset_PF = writeInputs.create_dataset("PF",  dtype='f',
        data=[pol2kar_x(DataF['recoilslimmedMETs_Pt'], DataF['recoilslimmedMETs_Phi']),
        pol2kar_y(DataF['recoilslimmedMETs_Pt'], DataF['recoilslimmedMETs_Phi']),
        DataF['recoilslimmedMETs_sumEt'],
        DataF['NVertex'] ])
    dset_Track = writeInputs.create_dataset("Track",  dtype='f',
        data=[ pol2kar_x(DataF['recoilpatpfTrackMET_Pt'], DataF['recoilpatpfTrackMET_Phi']),
        pol2kar_y(DataF['recoilpatpfTrackMET_Pt'], DataF['recoilpatpfTrackMET_Phi']),
        DataF['recoilpatpfTrackMET_sumEt']])
    dset_NoPU = writeInputs.create_dataset("NoPU",  dtype='f',
        data=[pol2kar_x(DataF['recoilpatpfNoPUMET_Pt'], DataF['recoilpatpfNoPUMET_Phi']),
        pol2kar_y(DataF['recoilpatpfNoPUMET_Pt'], DataF['recoilpatpfNoPUMET_Phi']),
        DataF['recoilpatpfNoPUMET_sumEt']])
    dset_PUCorrected = writeInputs.create_dataset("PUCorrected",  dtype='f',
        data=[pol2kar_x(DataF['recoilpatpfPUCorrectedMET_Pt'], DataF['recoilpatpfPUCorrectedMET_Phi']),
        pol2kar_y(DataF['recoilpatpfPUCorrectedMET_Pt'], DataF['recoilpatpfPUCorrectedMET_Phi']),
        DataF['recoilpatpfPUCorrectedMET_sumEt']])
    dset_PU = writeInputs.create_dataset("PU",  dtype='f',
        data=[pol2kar_x(DataF['recoilpatpfPUMET_Pt'], DataF['recoilpatpfPUMET_Phi']),
        pol2kar_y(DataF['recoilpatpfPUMET_Pt'], DataF['recoilpatpfPUMET_Phi']),
        DataF['recoilpatpfPUMET_sumEt']])
    dset_Puppi = writeInputs.create_dataset("Puppi",  dtype='f',
        data=[pol2kar_x(DataF['recoilslimmedMETsPuppi_Pt'], DataF['recoilslimmedMETsPuppi_Phi']),
        pol2kar_y(DataF['recoilslimmedMETsPuppi_Pt'], DataF['recoilslimmedMETsPuppi_Phi']),
        DataF['recoilslimmedMETsPuppi_sumEt']])


    dset_Target = writeInputs.create_dataset("Target",  dtype='f',
        data=[-pol2kar_x(DataF['Boson_Pt'], DataF['Boson_Phi']),
        -pol2kar_y(DataF['Boson_Pt'], DataF['Boson_Phi'])])
    writeInputs.close()

def getInputs_xyd(DataF):
    dset_PF = writeInputs.create_dataset("PF",  dtype='f',
        data=[pol2kar_x(DataF['recoilslimmedMETs_Pt'], DataF['recoilslimmedMETs_Phi']),
        pol2kar_y(DataF['recoilslimmedMETs_Pt'], DataF['recoilslimmedMETs_Phi']),
        DataF['recoilslimmedMETs_sumEt'],
        DataF['NVertex'] ])
    dset_Track = writeInputs.create_dataset("Track",  dtype='f',
        data=[ pol2kar_x(DataF['recoilpatpfTrackMET_Pt'], DataF['recoilpatpfTrackMET_Phi']),
        pol2kar_y(DataF['recoilpatpfTrackMET_Pt'], DataF['recoilpatpfTrackMET_Phi']),
        DataF['recoilpatpfTrackMET_sumEt']])
    dset_NoPU = writeInputs.create_dataset("NoPU",  dtype='f',
        data=[pol2kar_x(DataF['recoilpatpfNoPUMET_Pt'], DataF['recoilpatpfNoPUMET_Phi']),
        pol2kar_y(DataF['recoilpatpfNoPUMET_Pt'], DataF['recoilpatpfNoPUMET_Phi']),
        DataF['recoilpatpfNoPUMET_sumEt']])
    dset_PUCorrected = writeInputs.create_dataset("PUCorrected",  dtype='f',
        data=[pol2kar_x(DataF['recoilpatpfPUCorrectedMET_Pt'], DataF['recoilpatpfPUCorrectedMET_Phi']),
        pol2kar_y(DataF['recoilpatpfPUCorrectedMET_Pt'], DataF['recoilpatpfPUCorrectedMET_Phi']),
        DataF['recoilpatpfPUCorrectedMET_sumEt']])
    dset_PU = writeInputs.create_dataset("PU",  dtype='f',
        data=[pol2kar_x(DataF['recoilpatpfPUMET_Pt'], DataF['recoilpatpfPUMET_Phi']),
        pol2kar_y(DataF['recoilpatpfPUMET_Pt'], DataF['recoilpatpfPUMET_Phi']),
        DataF['recoilpatpfPUMET_sumEt']])
    dset_Puppi = writeInputs.create_dataset("Puppi",  dtype='f',
        data=[pol2kar_x(DataF['recoilslimmedMETsPuppi_Pt'], DataF['recoilslimmedMETsPuppi_Phi']),
        pol2kar_y(DataF['recoilslimmedMETsPuppi_Pt'], DataF['recoilslimmedMETsPuppi_Phi']),
        DataF['recoilslimmedMETsPuppi_sumEt']])


    dset_Target = writeInputs.create_dataset("Target",  dtype='f',
        data=[-pol2kar_x(DataF['Boson_Pt'], DataF['Boson_Phi']),
        -pol2kar_y(DataF['Boson_Pt'], DataF['Boson_Phi']),
        DataF['Boson_Pt']-DataF['recoilslimmedMETs_Pt'] ])
    writeInputs.close()

def getInputs_nr(DataF):
    dset_PF = writeInputs.create_dataset("PF",  dtype='f',
        data=[pol2kar_x(DataF['recoilslimmedMETs_Pt'], DataF['recoilslimmedMETs_Phi']),
        pol2kar_y(DataF['recoilslimmedMETs_Pt'], DataF['recoilslimmedMETs_Phi']),
        DataF['recoilslimmedMETs_sumEt'],
        DataF['NVertex'] ])
    dset_Track = writeInputs.create_dataset("Track",  dtype='f',
        data=[ pol2kar_x(DataF['recoilpatpfTrackMET_Pt'], DataF['recoilpatpfTrackMET_Phi']),
        pol2kar_y(DataF['recoilpatpfTrackMET_Pt'], DataF['recoilpatpfTrackMET_Phi']),
        DataF['recoilpatpfTrackMET_sumEt']])
    dset_NoPU = writeInputs.create_dataset("NoPU",  dtype='f',
        data=[pol2kar_x(DataF['recoilpatpfNoPUMET_Pt'], DataF['recoilpatpfNoPUMET_Phi']),
        pol2kar_y(DataF['recoilpatpfNoPUMET_Pt'], DataF['recoilpatpfNoPUMET_Phi']),
        DataF['recoilpatpfNoPUMET_sumEt']])
    dset_PUCorrected = writeInputs.create_dataset("PUCorrected",  dtype='f',
        data=[pol2kar_x(DataF['recoilpatpfPUCorrectedMET_Pt'], DataF['recoilpatpfPUCorrectedMET_Phi']),
        pol2kar_y(DataF['recoilpatpfPUCorrectedMET_Pt'], DataF['recoilpatpfPUCorrectedMET_Phi']),
        DataF['recoilpatpfPUCorrectedMET_sumEt']])
    dset_PU = writeInputs.create_dataset("PU",  dtype='f',
        data=[pol2kar_x(DataF['recoilpatpfPUMET_Pt'], DataF['recoilpatpfPUMET_Phi']),
        pol2kar_y(DataF['recoilpatpfPUMET_Pt'], DataF['recoilpatpfPUMET_Phi']),
        DataF['recoilpatpfPUMET_sumEt']])
    dset_Puppi = writeInputs.create_dataset("Puppi",  dtype='f',
        data=[pol2kar_x(DataF['recoilslimmedMETsPuppi_Pt'], DataF['recoilslimmedMETsPuppi_Phi']),
        pol2kar_y(DataF['recoilslimmedMETsPuppi_Pt'], DataF['recoilslimmedMETsPuppi_Phi']),
        DataF['recoilslimmedMETsPuppi_sumEt']])


    dset_Target = writeInputs.create_dataset("Target",  dtype='f',
        data=[-div0(pol2kar_x(DataF['Boson_Pt'], DataF['Boson_Phi']),DataF['Boson_Pt']),
        -div0(pol2kar_y(DataF['Boson_Pt'], DataF['Boson_Phi']),DataF['Boson_Pt']),
        DataF['Boson_Pt']
        ])
    writeInputs.close()

def getInputs_xyr(DataF):
    dset_PF = writeInputs.create_dataset("PF",  dtype='f',
        data=[pol2kar_x(DataF['recoilslimmedMETs_Pt'], DataF['recoilslimmedMETs_Phi']),
        pol2kar_y(DataF['recoilslimmedMETs_Pt'], DataF['recoilslimmedMETs_Phi']),
        DataF['recoilslimmedMETs_sumEt'],
        DataF['NVertex'] ])
    dset_Track = writeInputs.create_dataset("Track",  dtype='f',
        data=[ pol2kar_x(DataF['recoilpatpfTrackMET_Pt'], DataF['recoilpatpfTrackMET_Phi']),
        pol2kar_y(DataF['recoilpatpfTrackMET_Pt'], DataF['recoilpatpfTrackMET_Phi']),
        DataF['recoilpatpfTrackMET_sumEt']])
    dset_NoPU = writeInputs.create_dataset("NoPU",  dtype='f',
        data=[pol2kar_x(DataF['recoilpatpfNoPUMET_Pt'], DataF['recoilpatpfNoPUMET_Phi']),
        pol2kar_y(DataF['recoilpatpfNoPUMET_Pt'], DataF['recoilpatpfNoPUMET_Phi']),
        DataF['recoilpatpfNoPUMET_sumEt']])
    dset_PUCorrected = writeInputs.create_dataset("PUCorrected",  dtype='f',
        data=[pol2kar_x(DataF['recoilpatpfPUCorrectedMET_Pt'], DataF['recoilpatpfPUCorrectedMET_Phi']),
        pol2kar_y(DataF['recoilpatpfPUCorrectedMET_Pt'], DataF['recoilpatpfPUCorrectedMET_Phi']),
        DataF['recoilpatpfPUCorrectedMET_sumEt']])
    dset_PU = writeInputs.create_dataset("PU",  dtype='f',
        data=[pol2kar_x(DataF['recoilpatpfPUMET_Pt'], DataF['recoilpatpfPUMET_Phi']),
        pol2kar_y(DataF['recoilpatpfPUMET_Pt'], DataF['recoilpatpfPUMET_Phi']),
        DataF['recoilpatpfPUMET_sumEt']])
    dset_Puppi = writeInputs.create_dataset("Puppi",  dtype='f',
        data=[pol2kar_x(DataF['recoilslimmedMETsPuppi_Pt'], DataF['recoilslimmedMETsPuppi_Phi']),
        pol2kar_y(DataF['recoilslimmedMETsPuppi_Pt'], DataF['recoilslimmedMETsPuppi_Phi']),
        DataF['recoilslimmedMETsPuppi_sumEt']])


    dset_Target = writeInputs.create_dataset("Target",  dtype='f',
        data=[-pol2kar_x(DataF['Boson_Pt'], DataF['Boson_Phi']),
        -pol2kar_y(DataF['Boson_Pt'], DataF['Boson_Phi']),
        DataF['Boson_Pt']])
    writeInputs.close()

def getInputs_xyra(DataF):
    dset_PF = writeInputs.create_dataset("PF",  dtype='f',
        data=[pol2kar_x(DataF['recoilslimmedMETs_Pt'], DataF['recoilslimmedMETs_Phi']),
        pol2kar_y(DataF['recoilslimmedMETs_Pt'], DataF['recoilslimmedMETs_Phi']),
        DataF['recoilslimmedMETs_sumEt'],
        DataF['NVertex'] ])
    dset_Track = writeInputs.create_dataset("Track",  dtype='f',
        data=[ pol2kar_x(DataF['recoilpatpfTrackMET_Pt'], DataF['recoilpatpfTrackMET_Phi']),
        pol2kar_y(DataF['recoilpatpfTrackMET_Pt'], DataF['recoilpatpfTrackMET_Phi']),
        DataF['recoilpatpfTrackMET_sumEt']])
    dset_NoPU = writeInputs.create_dataset("NoPU",  dtype='f',
        data=[pol2kar_x(DataF['recoilpatpfNoPUMET_Pt'], DataF['recoilpatpfNoPUMET_Phi']),
        pol2kar_y(DataF['recoilpatpfNoPUMET_Pt'], DataF['recoilpatpfNoPUMET_Phi']),
        DataF['recoilpatpfNoPUMET_sumEt']])
    dset_PUCorrected = writeInputs.create_dataset("PUCorrected",  dtype='f',
        data=[pol2kar_x(DataF['recoilpatpfPUCorrectedMET_Pt'], DataF['recoilpatpfPUCorrectedMET_Phi']),
        pol2kar_y(DataF['recoilpatpfPUCorrectedMET_Pt'], DataF['recoilpatpfPUCorrectedMET_Phi']),
        DataF['recoilpatpfPUCorrectedMET_sumEt']])
    dset_PU = writeInputs.create_dataset("PU",  dtype='f',
        data=[pol2kar_x(DataF['recoilpatpfPUMET_Pt'], DataF['recoilpatpfPUMET_Phi']),
        pol2kar_y(DataF['recoilpatpfPUMET_Pt'], DataF['recoilpatpfPUMET_Phi']),
        DataF['recoilpatpfPUMET_sumEt']])
    dset_Puppi = writeInputs.create_dataset("Puppi",  dtype='f',
        data=[pol2kar_x(DataF['recoilslimmedMETsPuppi_Pt'], DataF['recoilslimmedMETsPuppi_Phi']),
        pol2kar_y(DataF['recoilslimmedMETsPuppi_Pt'], DataF['recoilslimmedMETsPuppi_Phi']),
        DataF['recoilslimmedMETsPuppi_sumEt']])


    dset_Target = writeInputs.create_dataset("Target",  dtype='f',
        data=[pol2kar_x(DataF['Boson_Pt'], DataF['Boson_Phi']),
        pol2kar_y(DataF['Boson_Pt'], DataF['Boson_Phi']),
        div0(DataF['Boson_Pt'],DataF['recoilslimmedMETs_Pt'])])
    writeInputs.close()

def getInputs_absCorr(DataF):
    return()

def getInputs_proj(DataF):
    dset_PF = writeInputs.create_dataset("PF",  dtype='f',
        data=[DataF['recoilslimmedMETs_LongZ'],
        DataF['recoilslimmedMETs_PerpZ'],
        DataF['recoilslimmedMETs_sumEt'] ])
    dset_Track = writeInputs.create_dataset("Track",  dtype='f',
        data=[ DataF['recoilpatpfTrackMET_LongZ'],
        DataF['recoilpatpfTrackMET_PerpZ'],
        DataF['recoilpatpfTrackMET_sumEt']])
    dset_NoPU = writeInputs.create_dataset("NoPU",  dtype='f',
        data=[DataF['recoilpatpfNoPUMET_LongZ'],
        DataF['recoilpatpfNoPUMET_PerpZ'],
        DataF['recoilpatpfNoPUMET_sumEt']])
    dset_PUCorrected = writeInputs.create_dataset("PUCorrected",  dtype='f',
        data=[DataF['recoilpatpfPUCorrectedMET_LongZ'],
        DataF['recoilpatpfPUCorrectedMET_PerpZ'],
        DataF['recoilpatpfPUCorrectedMET_sumEt']])
    dset_PU = writeInputs.create_dataset("PU",  dtype='f',
        data=[DataF['recoilpatpfPUMET_LongZ'],
        DataF['recoilpatpfPUMET_PerpZ'],
        DataF['recoilpatpfPUMET_sumEt']])
    dset_Puppi = writeInputs.create_dataset("Puppi",  dtype='f',
        data=[DataF['recoilslimmedMETsPuppi_LongZ'],
        DataF['recoilslimmedMETsPuppi_PerpZ'],
        DataF['recoilslimmedMETsPuppi_sumEt']])


    dset_Target = writeInputs.create_dataset("Target",  dtype='f',
        data=[-pol2kar_x(DataF['Boson_Pt'], DataF['Boson_Phi']),
        -pol2kar_y(DataF['Boson_Pt'], DataF['Boson_Phi']),
        DataF['Boson_Pt']
        ])
    writeInputs.close()


def getInputs(fName, NN_mode):

    if NN_mode == 'xy':
        Data = loadData(fName)
        Inputs = getInputs_xy(Data)
    elif NN_mode =='xyr':
        Data = loadData(fName)
        Inputs = getInputs_xyr(Data)
    elif NN_mode =='xyra':
        Data = loadData(fName)
        Inputs = getInputs_xyra(Data)
    elif NN_mode =='xyd':
        Data = loadData(fName)
        Inputs = getInputs_xyd(Data)        
    elif NN_mode =='nr':
        Data = loadData(fName)
        Inputs = getInputs_nr(Data)
    elif NN_mode == 'absCorr':
        Data = loadData(fName)
        Inputs = getInputs_absCorr(Data)
    else:
        Data = loadData_proj(fName)
        Inputs =  getInputs_proj(Data)

if __name__ == "__main__":
    fileName = sys.argv[1]
    outputDir = sys.argv[2]
    NN_mode = sys.argv[3]
    print(fileName)
    writeInputs = h5py.File("%sNN_Input_%s.h5"%(outputDir,NN_mode), "w")
    getInputs(fileName, NN_mode)
    #getTarge
