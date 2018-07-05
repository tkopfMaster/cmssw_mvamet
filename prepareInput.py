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

def loadDataTau(fName):
    tfile = ROOT.TFile(fName)
    for key in tfile.GetListOfKeys():
            if key.GetName() == "MAPAnalyzer/t":
                tree = key.ReadObj()
    arrayName = rnp.tree2array(tree, branches=['genMET_Pt', 'genMET_Phi','Boson_Pt', 'Boson_Phi', 'NVertex' ,
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
    x = np.sin(phi[:])*norm[:]
    return(x)
def pol2kar_y(norm, phi):
    y = np.cos(phi[:])*norm[:]
    return(y)

def getInputs_xy(DataF, outputD):
    dset_PF = writeInputs.create_dataset("PF",  dtype='f',
        data=[pol2kar_x(DataF['recoilslimmedMETs_Pt'], DataF['recoilslimmedMETs_Phi']),
        pol2kar_y(DataF['recoilslimmedMETs_Pt'], DataF['recoilslimmedMETs_Phi']) ])
    ''' ,
        DataF['recoilslimmedMETs_sumEt'],
        DataF['NVertex']])
    '''

    dset_Track = writeInputs.create_dataset("Track",  dtype='f',
        data=[ pol2kar_x(DataF['recoilpatpfTrackMET_Pt'], DataF['recoilpatpfTrackMET_Phi']),
        pol2kar_y(DataF['recoilpatpfTrackMET_Pt'], DataF['recoilpatpfTrackMET_Phi'])])
    ''',
        DataF['recoilpatpfTrackMET_sumEt']])
    '''

    dset_NoPU = writeInputs.create_dataset("NoPU",  dtype='f',
        data=[pol2kar_x(DataF['recoilpatpfNoPUMET_Pt'], DataF['recoilpatpfNoPUMET_Phi']),
        pol2kar_y(DataF['recoilpatpfNoPUMET_Pt'], DataF['recoilpatpfNoPUMET_Phi'])])
    ''',
        DataF['recoilpatpfNoPUMET_sumEt']])'''

    dset_PUCorrected = writeInputs.create_dataset("PUCorrected",  dtype='f',
        data=[pol2kar_x(DataF['recoilpatpfPUCorrectedMET_Pt'], DataF['recoilpatpfPUCorrectedMET_Phi']),
        pol2kar_y(DataF['recoilpatpfPUCorrectedMET_Pt'], DataF['recoilpatpfPUCorrectedMET_Phi'])])
    ''',
        DataF['recoilpatpfPUCorrectedMET_sumEt']])'''

    dset_PU = writeInputs.create_dataset("PU",  dtype='f',
        data=[pol2kar_x(DataF['recoilpatpfPUMET_Pt'], DataF['recoilpatpfPUMET_Phi']),
        pol2kar_y(DataF['recoilpatpfPUMET_Pt'], DataF['recoilpatpfPUMET_Phi'])])
    ''',
        DataF['recoilpatpfPUMET_sumEt']])'''

    dset_Puppi = writeInputs.create_dataset("Puppi",  dtype='f',
        data=[pol2kar_x(DataF['recoilslimmedMETsPuppi_Pt'], DataF['recoilslimmedMETsPuppi_Phi']),
        pol2kar_y(DataF['recoilslimmedMETsPuppi_Pt'], DataF['recoilslimmedMETsPuppi_Phi'])])
    ''',
        DataF['recoilslimmedMETsPuppi_sumEt']])'''



    dset_Target = writeInputs.create_dataset("Target",  dtype='f',
        data=[-pol2kar_x(DataF['Boson_Pt'], DataF['Boson_Phi']),
        -pol2kar_y(DataF['Boson_Pt'], DataF['Boson_Phi'])])

    writeInputs.close()

def getInputs_xyd(DataF, outputD):
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

    LegendTitle = '$\mathrm{Summer\ 17\ campaign}$' '\n'  '$\mathrm{Z \  \\rightarrow \ \mu \mu}$'

    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)
    nbinsHist = 150
    plt.hist(dset_PF[0,:], bins=nbinsHist, range=[-200, 200], label='PF, mean=%.2f'%np.mean(dset_PF[0,:]), histtype='step', ec=colors[0])
    plt.hist(dset_Track[0,:], bins=nbinsHist, range=[-200, 200], label='Track, mean=%.2f'%np.mean(dset_Track[0,:]), histtype='step', ec=colors[0])
    plt.hist(dset_NoPU[0,:], bins=nbinsHist, range=[-200, 200], label='NoPU, mean=%.2f'%np.mean(dset_NoPU[0,:]), histtype='step', ec=colors[2])
    plt.hist(dset_PUCorrected[0,:], bins=nbinsHist, range=[-200, 200], label='PUCorrected, mean=%.2f'%np.mean(dset_PUCorrected[0,:]), histtype='step', ec=colors[3])
    plt.hist(dset_PU[0,:], bins=nbinsHist, range=[-200, 200], label='PU, mean=%.2f'%np.mean(dset_PU[0,:]), histtype='step', ec=colors[4])
    plt.hist(dset_Puppi[0,:], bins=nbinsHist, range=[-200, 200], label='Puppi, mean=%.2f'%np.mean(dset_Puppi[0,:]), histtype='step', ec=colors[5])


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    #handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.ylabel('Counts')
    plt.xlabel('$p_{T,x}$ in GeV')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_{T}^Z} \\right) $ in GeV')
    plt.title(' Histogram $p_{T,x}$')
    #plt.text('$p_T$ range restriction')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(ylimResMVAMin, ylimResMax)
    plt.savefig("%sInput1_Hist.png"%(outputD))

    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)
    nbinsHist = 150
    plt.hist(dset_PF[1,:], bins=nbinsHist, range=[-200, 200], label='PF, mean=%.2f'%np.mean(dset_PF[1,:]), histtype='step', ec=colors[0])
    plt.hist(dset_Track[1,:], bins=nbinsHist, range=[-200, 200], label='Track, mean=%.2f'%np.mean(dset_Track[1,:]), histtype='step', ec=colors[1])
    plt.hist(dset_NoPU[1,:], bins=nbinsHist, range=[-200, 200], label='NoPU, mean=%.2f'%np.mean(dset_NoPU[1,:]), histtype='step', ec=colors[2])
    plt.hist(dset_PUCorrected[1,:], bins=nbinsHist, range=[-200, 200], label='PUCorrected, mean=%.2f'%np.mean(dset_PUCorrected[1,:]), histtype='step', ec=colors[3])
    plt.hist(dset_PU[1,:], bins=nbinsHist, range=[-200, 200], label='PU, mean=%.2f'%np.mean(dset_PU[1,:]), histtype='step', ec=colors[4])
    plt.hist(dset_Puppi[1,:], bins=nbinsHist, range=[-200, 200], label='Puppi, mean=%.2f'%np.mean(dset_Puppi[1,:]), histtype='step', ec=colors[5])


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    #handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.ylabel('Counts')
    plt.xlabel('$p_{T,y}$ in GeV')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_{T}^Z} \\right) $ in GeV')
    plt.title(' Histogram $p_{T,y}$')
    #plt.text('$p_T$ range restriction')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(ylimResMVAMin, ylimResMax)
    plt.savefig("%sInput2_Hist.png"%(outputD))





    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    plt.hist(dset_PF[2,:], bins=nbinsHist, range=[0, 1500], label='PF', histtype='step', ec=colors[0])
    plt.hist(dset_Track[2,:], bins=nbinsHist, range=[0, 1500], label='Track', histtype='step', ec=colors[1])
    plt.hist(dset_NoPU[2,:], bins=nbinsHist, range=[0, 1500], label='NoPU', histtype='step', ec=colors[2])
    plt.hist(dset_PUCorrected[2,:], bins=nbinsHist, range=[0, 1500], label='PUCorrected', histtype='step', ec=colors[3])
    plt.hist(dset_PU[2,:], bins=nbinsHist, range=[0, 1500], label='PU', histtype='step', ec=colors[4])
    plt.hist(dset_Puppi[2,:], bins=nbinsHist, range=[0, 1500], label='Puppi', histtype='step', ec=colors[5])


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    #handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.ylabel('Counts')
    plt.xlabel('$\\Sigma E_{T}$ in GeV')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_{T}^Z} \\right) $ in GeV')
    plt.title(' Histogram $\\Sigma E_{T}$')
    #plt.text('$p_T$ range restriction')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(ylimResMVAMin, ylimResMax)
    plt.savefig("%sInput3_Hist.png"%(outputD))

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

    LegendTitle = '$\mathrm{Summer\ 17\ campaign}$' '\n'  '$\mathrm{Z \  \\rightarrow \ \mu \mu}$'

    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)
    nbinsHist = 150
    plt.hist(dset_PF[0,:], bins=nbinsHist, range=[-200, 200], label='PF, mean=%.2f'%np.mean(dset_PF[0,:]), histtype='step', ec=colors[0])
    plt.hist(dset_Track[0,:], bins=nbinsHist, range=[-200, 200], label='Track, mean=%.2f'%np.mean(dset_Track[0,:]), histtype='step', ec=colors[0])
    plt.hist(dset_NoPU[0,:], bins=nbinsHist, range=[-200, 200], label='NoPU, mean=%.2f'%np.mean(dset_NoPU[0,:]), histtype='step', ec=colors[2])
    plt.hist(dset_PUCorrected[0,:], bins=nbinsHist, range=[-200, 200], label='PUCorrected, mean=%.2f'%np.mean(dset_PUCorrected[0,:]), histtype='step', ec=colors[3])
    plt.hist(dset_PU[0,:], bins=nbinsHist, range=[-200, 200], label='PU, mean=%.2f'%np.mean(dset_PU[0,:]), histtype='step', ec=colors[4])
    plt.hist(dset_Puppi[0,:], bins=nbinsHist, range=[-200, 200], label='Puppi, mean=%.2f'%np.mean(dset_Puppi[0,:]), histtype='step', ec=colors[5])


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    #handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.ylabel('Counts')
    plt.xlabel('$p_{T,x}$ in GeV')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_{T}^Z} \\right) $ in GeV')
    plt.title(' Histogram $p_{T,x}$')
    #plt.text('$p_T$ range restriction')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(ylimResMVAMin, ylimResMax)
    plt.savefig("%sInput1_Hist.png"%(outputD))

    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    plt.hist(dset_PF[1,:], bins=nbinsHist, range=[-200, 200], label='PF, mean=%.2f'%np.mean(dset_PF[1,:]), histtype='step', ec=colors[0])
    plt.hist(dset_Track[1,:], bins=nbinsHist, range=[-200, 200], label='Track, mean=%.2f'%np.mean(dset_Track[1,:]), histtype='step', ec=colors[1])
    plt.hist(dset_NoPU[1,:], bins=nbinsHist, range=[-200, 200], label='NoPU, mean=%.2f'%np.mean(dset_NoPU[1,:]), histtype='step', ec=colors[2])
    plt.hist(dset_PUCorrected[1,:], bins=nbinsHist, range=[-200, 200], label='PUCorrected, mean=%.2f'%np.mean(dset_PUCorrected[1,:]), histtype='step', ec=colors[3])
    plt.hist(dset_PU[1,:], bins=nbinsHist, range=[-200, 200], label='PU, mean=%.2f'%np.mean(dset_PU[1,:]), histtype='step', ec=colors[4])
    plt.hist(dset_Puppi[1,:], bins=nbinsHist, range=[-200, 200], label='Puppi, mean=%.2f'%np.mean(dset_Puppi[1,:]), histtype='step', ec=colors[5])


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    #handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.ylabel('Counts')
    plt.xlabel('$p_{T,y}$ in GeV')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_{T}^Z} \\right) $ in GeV')
    plt.title(' Histogram $p_{T,y}$')
    #plt.text('$p_T$ range restriction')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(ylimResMVAMin, ylimResMax)
    plt.savefig("%sInput2_Hist.png"%(outputD))





    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    plt.hist(dset_PF[2,:], bins=nbinsHist, range=[0, 1500], label='PF', histtype='step', ec=colors[0])
    plt.hist(dset_Track[2,:], bins=nbinsHist, range=[0, 1500], label='Track', histtype='step', ec=colors[1])
    plt.hist(dset_NoPU[2,:], bins=nbinsHist, range=[0, 1500], label='NoPU', histtype='step', ec=colors[2])
    plt.hist(dset_PUCorrected[2,:], bins=nbinsHist, range=[0, 1500], label='PUCorrected', histtype='step', ec=colors[3])
    plt.hist(dset_PU[2,:], bins=nbinsHist, range=[0, 1500], label='PU', histtype='step', ec=colors[4])
    plt.hist(dset_Puppi[2,:], bins=nbinsHist, range=[0, 1500], label='Puppi', histtype='step', ec=colors[5])


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    #handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.ylabel('Counts')
    plt.xlabel('$\\Sigma E_{T}$ in GeV')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_{T}^Z} \\right) $ in GeV')
    plt.title(' Histogram $\\Sigma E_{T}$')
    #plt.text('$p_T$ range restriction')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(ylimResMVAMin, ylimResMax)
    plt.savefig("%sInput3_Hist.png"%(outputD))

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

def getInputs_rphi(DataF, outputD):
    dset_PF = writeInputs.create_dataset("PF",  dtype='f',
        data=[DataF['recoilslimmedMETs_Pt'],
              DataF['recoilslimmedMETs_Phi']])
    dset_Track = writeInputs.create_dataset("Track",  dtype='f',
        data=[ DataF['recoilpatpfTrackMET_Pt'],
              DataF['recoilpatpfTrackMET_Phi']])
    dset_NoPU = writeInputs.create_dataset("NoPU",  dtype='f',
        data=[DataF['recoilpatpfNoPUMET_Pt'],
              DataF['recoilpatpfNoPUMET_Phi']])
    dset_PUCorrected = writeInputs.create_dataset("PUCorrected",  dtype='f',
        data=[DataF['recoilpatpfPUCorrectedMET_Pt'],
              DataF['recoilpatpfPUCorrectedMET_Phi']])
    dset_PU = writeInputs.create_dataset("PU",  dtype='f',
        data=[DataF['recoilpatpfPUMET_Pt'],
              DataF['recoilpatpfPUMET_Phi']])
    dset_Puppi = writeInputs.create_dataset("Puppi",  dtype='f',
        data=[DataF['recoilslimmedMETsPuppi_Pt'],
              DataF['recoilslimmedMETsPuppi_Phi']])


    dset_Target = writeInputs.create_dataset("Target",  dtype='f',
        data=[DataF['Boson_Pt'],
              angularrange(DataF['Boson_Phi']+np.pi)])


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)
    nbinsHist = 150
    LegendTitle = '$\mathrm{Summer\ 17\ campaign}$' '\n'  '$\mathrm{Z \  \\rightarrow \ \mu \mu}$'
    plt.hist(angularrange(dset_PF[1,:]+np.pi), bins=nbinsHist, range=[-np.pi*1.1, np.pi*1.1], label='PF, mean=%.2f'%np.mean(angularrange(dset_PF[1,:]+np.pi)), histtype='step', ec=colors[0])
    plt.hist(angularrange(dset_Track[1,:]+np.pi), bins=nbinsHist, range=[-np.pi*1.1, np.pi*1.1], label='Track, mean=%.2f'%np.mean(angularrange(dset_Track[1,:]+np.pi)), histtype='step', ec=colors[1])
    plt.hist(angularrange(dset_NoPU[1,:]+np.pi), bins=nbinsHist, range=[-np.pi*1.1, np.pi*1.1], label='NoPU, mean=%.2f'%np.mean(angularrange(dset_NoPU[1,:]+np.pi)), histtype='step', ec=colors[2])
    plt.hist(angularrange(dset_PUCorrected[1,:]+np.pi), bins=nbinsHist, range=[-np.pi*1.1, np.pi*1.1], label='PUCorrected, mean=%.2f'%np.mean(angularrange(dset_PUCorrected[1,:]+np.pi)), histtype='step', ec=colors[3])
    plt.hist(angularrange(dset_PU[1,:]+np.pi), bins=nbinsHist, range=[-np.pi*1.1, np.pi*1.1], label='PU, mean=%.2f'%np.mean(angularrange(dset_PU[1,:]+np.pi)), histtype='step', ec=colors[4])
    plt.hist(angularrange(dset_Puppi[1,:]+np.pi), bins=nbinsHist, range=[-np.pi*1.1, np.pi*1.1], label='Puppi, mean=%.2f'%np.mean(angularrange(dset_Puppi[1,:]+np.pi)), histtype='step', ec=colors[5])
    plt.hist(dset_Target[1,:], bins=nbinsHist, range=[-np.pi*1.1, np.pi*1.1], label='Target, mean=%.2f'%np.mean(dset_Target[1,:]), histtype='step', ec=colors[6])


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    #handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.ylabel('Counts')
    plt.xlabel('$\\alpha$ in rad')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_{T}^Z} \\right) $ in GeV')
    plt.title('  Input vs. Target: angular ')
    #plt.text('$p_T$ range restriction')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(ylimResMVAMin, ylimResMax)
    plt.savefig("%sInputs_vs_Target_angular.png"%(outputD))

    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    plt.hist(dset_PF[0,:], bins=nbinsHist, range=[0, 100], label='PF, mean=%.2f'%np.mean(dset_PF[0,:]), histtype='step', ec=colors[0])
    plt.hist(dset_Track[0,:], bins=nbinsHist, range=[0, 100], label='Track, mean=%.2f'%np.mean(dset_Track[0,:]), histtype='step', ec=colors[1])
    plt.hist(dset_NoPU[0,:], bins=nbinsHist, range=[0, 100], label='NoPU, mean=%.2f'%np.mean(dset_NoPU[0,:]), histtype='step', ec=colors[2])
    plt.hist(dset_PUCorrected[0,:], bins=nbinsHist, range=[0, 100], label='PUCorrected, mean=%.2f'%np.mean(dset_PUCorrected[0,:]), histtype='step', ec=colors[3])
    plt.hist(dset_PU[0,:], bins=nbinsHist, range=[0, 100], label='PU, mean=%.2f'%np.mean(dset_PU[0,:]), histtype='step', ec=colors[4])
    plt.hist(dset_Puppi[0,:], bins=nbinsHist, range=[0, 100], label='Puppi, mean=%.2f'%np.mean(dset_Puppi[0,:]), histtype='step', ec=colors[5])
    plt.hist(dset_Target[0,:], bins=nbinsHist, range=[0, 100], label='Target, mean=%.2f'%np.mean(dset_Target[0,:]), histtype='step', ec=colors[6])


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    #handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.ylabel('Counts')
    plt.xlabel('$p_T$ in GeV')
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_{T}^Z} \\right) $ in GeV')
    plt.title('  Input vs. Target: norm')
    #plt.text('$p_T$ range restriction')

    ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
    plt.grid()
    #plt.ylim(ylimResMVAMin, ylimResMax)
    plt.savefig("%sInputs_vs_Target_norm.png"%(outputD))

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


def getInputs(fName, NN_mode, outputD, PhysicsProcess):
    if PhysicsProcess=='Tau':
        Data = loadDataTau(fName)
        Inputs = getInputs_Tau_xy(Data, outputD)
    else:
        if NN_mode == 'xy':
            Data = loadData(fName)
            Inputs = getInputs_xy(Data, outputD)
        elif NN_mode =='xyr':
            Data = loadData(fName)
            Inputs = getInputs_xyr(Data)
        elif NN_mode =='xyra':
            Data = loadData(fName)
            Inputs = getInputs_xyra(Data)
        elif NN_mode =='xyd':
            Data = loadData(fName)
            Inputs = getInputs_xyd(Data, outputD)
        elif NN_mode =='nr':
            Data = loadData(fName)
            Inputs = getInputs_nr(Data)
        elif NN_mode == 'absCorr':
            Data = loadData(fName)
            Inputs = getInputs_absCorr(Data)
        elif NN_mode == 'rphi':
            Data = loadData(fName)
            Inputs = getInputs_rphi(Data, outputD)
        else:
            Data = loadData_proj(fName)
            Inputs =  getInputs_proj(Data)





if __name__ == "__main__":
    fileName = sys.argv[1]
    outputDir = sys.argv[2]
    NN_mode = sys.argv[3]
    plotsD = sys.argv[4]
    PhysicsProcess = plotsD = sys.argv[5]
    print(fileName)
    writeInputs = h5py.File("%sNN_Input_%s.h5"%(outputDir,NN_mode), "w")
    getInputs(fileName, NN_mode, plotsD, PhysicsProcess)
    #getTarge
