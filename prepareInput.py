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
import h5py
import sys



colors = cm.brg(np.linspace(0, 1, 7))

def div0( a, b , Target_Pt, Target_Phi):
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

def loadData(fName, Target_Pt, Target_Phi):
    tfile = ROOT.TFile(fName)
    for key in tfile.GetListOfKeys():
            print('key.GetName()', key.GetName())
            if key.GetName() == "MAPAnalyzer" and Target_Pt=='Boson_Pt':
                tree = key.ReadObj()
                print('tree', tree)
                treeName = 't'
                arrayName = rnp.tree2array(tree,  branches=[Target_Pt, Target_Phi, 'NVertex' ,
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
                    'LongZCorrectedRecoil_Pt', 'LongZCorrectedRecoil_Phi',
                    'LongZCorrectedRecoil_LongZ', 'LongZCorrectedRecoil_PerpZ',
                    'recoilslimmedMETsPuppi_LongZ', 'recoilslimmedMETsPuppi_PerpZ'],)
        DFName = pd.DataFrame.from_records(arrayName.view(np.recarray))
    return(DFName)



def loadData_proj(fName, Target_Pt, Target_Phi):
    treeName = 't'
    arrayName = rnp.root2array(fName, treeName, branches=[Target_Pt, Target_Phi, 'NVertex' ,   'recoilslimmedMETsPuppi_PerpZ', 'recoilslimmedMETsPuppi_LongZ', 'recoilslimmedMETs_PerpZ', 'recoilslimmedMETs_LongZ', 'recoilpatpfNoPUMET_PerpZ','recoilpatpfNoPUMET_LongZ','recoilpatpfPUCorrectedMET_PerpZ', 'recoilpatpfPUCorrectedMET_LongZ', 'recoilpatpfPUMET_PerpZ', 'recoilpatpfPUMET_LongZ', 'recoilpatpfTrackMET_PerpZ', 'recoilpatpfTrackMET_LongZ' ],)
    DFName = pd.DataFrame.from_records(arrayName.view(np.recarray))
    return(DFName)
#Data settings


def getInputs_xy(DataF, outputD, PhysicsProcess, Target_Pt, Target_Phi, dset):
    dset_PF = dset.create_dataset("PF",  dtype='f',
        data=[pol2kar_x(DataF['recoilslimmedMETs_Pt'], DataF['recoilslimmedMETs_Phi']),
        pol2kar_y(DataF['recoilslimmedMETs_Pt'], DataF['recoilslimmedMETs_Phi']) ])
    ''' ,
        DataF['recoilslimmedMETs_sumEt'],
        DataF['NVertex']])
    '''

    dset_Track = dset.create_dataset("Track",  dtype='f',
        data=[ pol2kar_x(DataF['recoilpatpfTrackMET_Pt'], DataF['recoilpatpfTrackMET_Phi']),
        pol2kar_y(DataF['recoilpatpfTrackMET_Pt'], DataF['recoilpatpfTrackMET_Phi'])])
    ''',
        DataF['recoilpatpfTrackMET_sumEt']])
    '''

    dset_NoPU = dset.create_dataset("NoPU",  dtype='f',
        data=[pol2kar_x(DataF['recoilpatpfNoPUMET_Pt'], DataF['recoilpatpfNoPUMET_Phi']),
        pol2kar_y(DataF['recoilpatpfNoPUMET_Pt'], DataF['recoilpatpfNoPUMET_Phi'])])
    ''',
        DataF['recoilpatpfNoPUMET_sumEt']])'''

    dset_PUCorrected = dset.create_dataset("PUCorrected",  dtype='f',
        data=[pol2kar_x(DataF['recoilpatpfPUCorrectedMET_Pt'], DataF['recoilpatpfPUCorrectedMET_Phi']),
        pol2kar_y(DataF['recoilpatpfPUCorrectedMET_Pt'], DataF['recoilpatpfPUCorrectedMET_Phi'])])
    ''',
        DataF['recoilpatpfPUCorrectedMET_sumEt']])'''

    dset_PU = dset.create_dataset("PU",  dtype='f',
        data=[pol2kar_x(DataF['recoilpatpfPUMET_Pt'], DataF['recoilpatpfPUMET_Phi']),
        pol2kar_y(DataF['recoilpatpfPUMET_Pt'], DataF['recoilpatpfPUMET_Phi'])])
    ''',
        DataF['recoilpatpfPUMET_sumEt']])'''

    dset_Puppi = dset.create_dataset("Puppi",  dtype='f',
        data=[pol2kar_x(DataF['recoilslimmedMETsPuppi_Pt'], DataF['recoilslimmedMETsPuppi_Phi']),
        pol2kar_y(DataF['recoilslimmedMETsPuppi_Pt'], DataF['recoilslimmedMETsPuppi_Phi'])])
    ''',
        DataF['recoilslimmedMETsPuppi_sumEt']])'''
    dset_NoPV = dset.create_dataset("NVertex",  dtype='f',data=[DataF['NVertex']] )


    dset_Target = dset.create_dataset("Target",  dtype='f',
            data=[-pol2kar_x(DataF[Target_Pt], DataF[Target_Phi]),
            -pol2kar_y(DataF[Target_Pt], DataF[Target_Phi])])

    dset.close()


def getInputs(fName, fileName_apply, NN_mode, outputD, PhysicsProcess, Target_Pt, Target_Phi):
    if PhysicsProcess=='Tau':
        Data = loadData(fName,  Target_Pt, Target_Phi)
        Inputs = getInputs_xy(Data, outputD, PhysicsProcess, Target_Pt, Target_Phi)
    else:
        if NN_mode == 'xy':
            Data = loadData(fName,  Target_Pt, Target_Phi)
            Inputs = getInputs_xy(Data, outputD, PhysicsProcess, Target_Pt, Target_Phi, writeInputs_training)
            Data_apply = loadData(fileName_apply,  Target_Pt, Target_Phi)
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
