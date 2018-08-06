import h5py
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import pandas as pd
from matplotlib.colors import LogNorm
import matplotlib.patches as mpatches
from getPlotsOutputclean import loadData, loadData_woutGBRT
from getResponse import getResponse, getResponseIdx
from prepareInput import pol2kar_x, pol2kar_y, kar2pol, pol2kar, angularrange




nbinsHist = 400
colors_InOut = cm.brg(np.linspace(0, 1, 8))
diff_p_min, diff_p_max = 0, 200


def Histogram_Angle(phi_, labelName, errbars_shift):
    if labelName=='Targets':
        phi_ = angularrange(phi_+np.pi)
    Mean = np.mean(phi_)
    Std = np.std(phi_)
    if labelName in ['NN MET', 'PF MET', 'Targets']:
        plt.hist(phi_, bins=nbinsHist, range=[-np.pi, np.pi], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift], linewidth=1.5)
    else:
        plt.hist(phi_, bins=nbinsHist, range=[-np.pi, np.pi], label=labelName+', %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', ec=colors_InOut[errbars_shift])


def Hist_Diff_norm(r,phi,pr,pphi, labelName, col, rangemin, rangemax):
    x = pol2kar_x(r[(pr > rangemin) & (pr < rangemax)],phi[(pr > rangemin) & (pr < rangemax)])
    y = pol2kar_y(r[(pr > rangemin) & (pr < rangemax)],phi[(pr > rangemin) & (pr < rangemax)])
    px = pol2kar_x(pr[(pr > rangemin) & (pr < rangemax)],pphi[(pr > rangemin) & (pr < rangemax)])
    py = pol2kar_y(pr[(pr > rangemin) & (pr < rangemax)],pphi[(pr > rangemin) & (pr < rangemax)])
    delta_x_sq = np.square(np.subtract(x,px))
    delta_y_sq = np.square(np.subtract(y,py))
    norm = np.sqrt(delta_x_sq+delta_y_sq)
    if rangemin==100:
        nbinsHist = 15
        diff_p_min_f, diff_p_max_f = 0, 400
    else:
        nbinsHist = 40
        diff_p_min_f, diff_p_max_f = diff_p_min, diff_p_max
    if labelName in ['NN', 'PF'] :
        plt.hist(norm, bins=nbinsHist, range=[diff_p_min_f, diff_p_max_f], label=labelName+', mean=%.2f$\pm$%.2f'%(np.mean(norm), np.std(norm)), histtype='step', ec=colors_InOut[col], linewidth=1.5, normed=False)
    else:
        plt.hist(norm, bins=nbinsHist, range=[diff_p_min_f, diff_p_max_f], label=labelName+', mean=%.2f$\pm$%.2f'%(np.mean(norm), np.std(norm)), histtype='step', ec=colors_InOut[col], normed=False)

def Diff_norm_PV(r, phi,pr,pphi, labelName, errbars_shift, rangemin, rangemax):
    IdxPVbins = (Outputs[Target_Pt]>=rangemin) & (Outputs[Target_Pt]<rangemax)
    DV_PVbins = Outputs[IdxPVbins]
    x = pol2kar_x(r[(pr >= rangemin) & (pr < rangemax)],phi[(pr >= rangemin) & (pr < rangemax)])
    y = pol2kar_y(r[(pr >= rangemin) & (pr < rangemax)],phi[(pr >= rangemin) & (pr < rangemax)])
    px = pol2kar_x(pr[(pr >= rangemin) & (pr < rangemax)],pphi[(pr >= rangemin) & (pr < rangemax)])
    py = pol2kar_y(pr[(pr >= rangemin) & (pr < rangemax)],pphi[(pr >= rangemin) & (pr < rangemax)])
    delta_x_sq = np.square(np.subtract(x,px))
    delta_y_sq = np.square(np.subtract(y,py))
    norm = np.sqrt(delta_x_sq+delta_y_sq)
    nbinsVertex=5
    binwidth = 50/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DV_PVbins.NVertex, bins=nbinsVertex, range=[0,50])
    sy, _ = np.histogram(DV_PVbins.NVertex, bins=nbinsVertex, weights=norm, range=[0,50])
    sy2, _ = np.histogram(DV_PVbins.NVertex, bins=nbinsVertex, weights=(norm)**2, range=[0,50])
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='.', xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors_InOut[errbars_shift])
    #plt.errorbar((_[:-1]+(_[1:]-_[:-1])/2), mean, yerr=std*ScaleErr, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])





def HM_Diff_norm(r,phi,pr,pphi, labelName, col):
    if labelName=='NN':
        x = pol2kar_x(r,phi)
        y = pol2kar_y(r,phi)
    else:
        x = pol2kar_x(r,phi)
        y = pol2kar_y(r,phi)
    px = -pol2kar_x(pr,pphi)
    py = -pol2kar_y(pr,pphi)
    delta_x_sq = np.square(np.subtract(x,px))
    delta_y_sq = np.square(np.subtract(y,py))
    norm = np.sqrt(delta_x_sq+delta_y_sq)
    return norm


def plotTraining(outputD, optim, loss_fct, NN_mode, plotsD, rootOutput, PhysicsProcess, Target_Pt, Target_Phi):
    if Target_Pt=='Boson_Pt':
        LegendTitle = '$\mathrm{Summer\ 17\ campaign}$' '\n'  '$\mathrm{Z \  \\rightarrow \ \mu \mu}$'
    else:
        LegendTitle = '$\mathrm{Summer\ 17\ campaign}$' '\n'  '$\mathrm{Z \  \\rightarrow \ \\tau \\tau   \\rightarrow \ \mu \mu}$'
    pTRangeString_Err = '$0\ \mathrm{GeV} < |-\\vec{p}_T^Z| \leq 200\ \mathrm{GeV}$ \n $\mathrm{\# Vertex} \leq 50$'
    pTRangeString= '$20\ \mathrm{GeV} < |-\\vec{p}_T^Z| \leq 200\ \mathrm{GeV}$ \n $\mathrm{\# Vertex} \leq 50$'
    pTRangeString_low= '$0\ \mathrm{GeV} < |-\\vec{p}_T^Z| \leq %8.2f \ \mathrm{GeV}$ \n $\mathrm{\# Vertex} \leq 50$'%(np.percentile(Outputs[Target_Pt],0.3333*100))
    pTRangeString_mid= '$%8.2f\ \mathrm{GeV} < |-\\vec{p}_T^Z| \leq %8.2f\ \mathrm{GeV}$ \n $\mathrm{\# Vertex} \leq 50$'%(np.percentile(Outputs[Target_Pt],0.3333*100), np.percentile(DFName[Target_Pt],0.6666*100))
    pTRangeString_high= '$20\ \mathrm{GeV} < |-\\vec{p}_T^Z| \leq 200\ \mathrm{GeV}$ \n $\mathrm{\# Vertex} \leq 50$'%(np.percentile(Outputs[Target_Pt],0.6666*100))
    pTRangeString_very_high= '$ 100 \mathrm{GeV} < |-\\vec{p}_T^Z| \leq 200\ \mathrm{GeV}$ \n $\mathrm{\# Vertex} \leq 50$'
    ptMin_low, ptMax_low = 0, np.percentile(Outputs[Target_Pt], 33.3333)
    ptMin_mid, ptMax_mid = np.percentile(Outputs[Target_Pt], 33.3333), np.percentile(Outputs[Target_Pt], 66.6666)
    ptMin_high, ptMax_high = 20, 200


    NN_Output_applied = h5py.File("%sNN_Output_applied_%s.h5"%(outputD,NN_mode), "r")
    #predictions=Outputs['NN_Pt'], Outputs['NN_Phi']
    predictions = NN_Output_applied["MET_Predictions"]
    Targets = NN_Output_applied["MET_GroundTruth"]
    #print("Shape predictions", predictions)
    #print("Shape Targets", Targets)

    #print('np.subtract(Outputs[NN_Pt], Outputs[Target_Pt])', np.subtract(Outputs['NN_Pt'], Outputs[Target_Pt]))
    print(Outputs[Target_Pt][:].shape)
    print(NN_Output_applied["MET_GroundTruth"].shape)
    print(Outputs[Target_Pt][:])
    print('np.mean(np.subtract(Outputs[NN_Pt], Outputs[Target_Pt]))', np.mean(np.subtract(Outputs['NN_Pt'], Outputs[Target_Pt])))
    #print('np.subtract(Outputs[NN_Phi], Outputs[Target_Phi])', np.subtract(Outputs['NN_Phi'], Outputs[Target_Phi]))
    print('np.mean(np.subtract(Outputs[NN_Pt], Outputs[Target_Pt]))', np.mean(np.subtract(Outputs['NN_Phi'], Outputs[Target_Phi])))

    #NN_Diff_x =  np.subtract(pol2kar_x(Outputs['NN_Pt'], Outputs['NN_Phi']), pol2kar_x(Outputs[Target_Pt], Outputs[Target_Phi]))
    #NN_Diff_y = np.subtract(pol2kar_y(Outputs['NN_Pt'], Outputs['NN_Phi']), pol2kar_y(Outputs[Target_Pt], Outputs[Target_Phi]))
    NN_Diff_x = np.subtract(Outputs['NN_x'], Outputs['Boson_x'])
    NN_Diff_y = np.subtract(Outputs['NN_y'], Outputs['Boson_y'])


    NN_Output = h5py.File("%sNN_Output_%s.h5"%(outputD,NN_mode), "r")
    loss = NN_Output["loss"]
    val_loss = NN_Output["val_loss"]



    if NN_mode == 'xy':
        #Load Inputs and Targets with Name
        InputsTargets_hdf5 = h5py.File("%sNN_Input_apply_%s.h5" % (outputD,NN_mode), "r")
        print("Keys: %s" % InputsTargets_hdf5.keys())
        keys = InputsTargets_hdf5.keys()
        values = [InputsTargets_hdf5[k] for k in keys]
        print(values)
        Norm_pT = np.sqrt(np.multiply(InputsTargets_hdf5['Target'][0,:], InputsTargets_hdf5['Target'][0,:]) + np.multiply(InputsTargets_hdf5['Target'][1,:], InputsTargets_hdf5['Target'][1,:]))
        Ind = (InputsTargets_hdf5['NVertex'].value<=50) &  (Norm_pT>0) & (Norm_pT<200)
        Ind = Ind.flatten()

        InputsTargets=InputsTargets_hdf5
        #print('InputsTargets', InputsTargets.shape())
        #print('Outputs', Outputs.shape())





        fig=plt.figure(figsize=(10,6))
        fig.patch.set_facecolor('white')
        ax = plt.subplot(111)
        nbinsHist = 40

        Hist_Diff_norm(Outputs['recoilpatpfTrackMET_Pt'], Outputs['recoilpatpfTrackMET_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'Track', 0, 0, 200)
        Hist_Diff_norm(Outputs['recoilpatpfNoPUMET_Pt'], Outputs['recoilpatpfNoPUMET_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'NoPU', 2, 0, 200)
        Hist_Diff_norm(Outputs['recoilpatpfPUCorrectedMET_Pt'], Outputs['recoilpatpfPUCorrectedMET_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'PUCorrected', 3, 0, 200)
        Hist_Diff_norm(Outputs['recoilpatpfPUMET_Pt'], Outputs['recoilpatpfPUMET_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'PU', 4, 0, 200)
        Hist_Diff_norm(Outputs['recoilslimmedMETsPuppi_Pt'], Outputs['recoilslimmedMETsPuppi_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'Puppi', 5, 0, 200)
        Hist_Diff_norm(Outputs['recoilslimmedMETs_Pt'], Outputs['recoilslimmedMETs_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'PF', 1, 0, 200)
        Hist_Diff_norm(Outputs['NN_Pt'], Outputs['NN_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'NN', 6, 0, 200)


        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        handles, labels = ax.get_legend_handles_labels()

        plt.ylabel('Counts')
        plt.xlabel('$|\\vec{U}+\\vec{p}_T^Z|$ in GeV')
        plt.xlim(diff_p_min,200)

        ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
        plt.grid()
        #plt.ylim(ylimResMVAMin, ylimResMax)
        plt.savefig("%sHist_Diff_norm.png"%(plotsD), bbox_inches="tight")

        '''
        fig=plt.figure(figsize=(10,6))
        fig.patch.set_facecolor('white')
        ax = plt.subplot(111)
        nbinsHist = 40

        Hist_Diff_norm(Outputs['recoilpatpfTrackMET_Pt'], Outputs['recoilpatpfTrackMET_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'Track', 0, ptMin_low, ptMax_low)
        Hist_Diff_norm(Outputs['recoilpatpfNoPUMET_Pt'], Outputs['recoilpatpfNoPUMET_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'NoPU', 2, ptMin_low, ptMax_low)
        Hist_Diff_norm(Outputs['recoilpatpfPUCorrectedMET_Pt'], Outputs['recoilpatpfPUCorrectedMET_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'PUCorrected', 3, ptMin_low, ptMax_low)
        Hist_Diff_norm(Outputs['recoilpatpfPUMET_Pt'], Outputs['recoilpatpfPUMET_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'PU', 4, ptMin_low, ptMax_low)
        Hist_Diff_norm(Outputs['recoilslimmedMETsPuppi_Pt'], Outputs['recoilslimmedMETsPuppi_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'Puppi', 5, ptMin_low, ptMax_low)
        Hist_Diff_norm(Outputs['recoilslimmedMETs_Pt'], Outputs['recoilslimmedMETs_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'PF', 1, ptMin_low, ptMax_low)
        Hist_Diff_norm(Outputs['NN_Pt'], Outputs['NN_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'NN', 6, ptMin_low, ptMax_low)


        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        handles, labels = ax.get_legend_handles_labels()
        handles.insert(0,mpatches.Patch(color='none', label=pTRangeString_low))

        plt.ylabel('Counts')
        plt.xlabel('$|\\vec{U}+\\vec{p}_T^Z|$ in GeV')
        plt.xlim(diff_p_min,diff_p_max)

        ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
        plt.grid()
        #plt.ylim(ylimResMVAMin, ylimResMax)
        plt.savefig("%sHist_Diff_norm_low.png"%(plotsD), bbox_inches="tight")

        fig=plt.figure(figsize=(10,6))
        fig.patch.set_facecolor('white')
        ax = plt.subplot(111)
        nbinsHist = 40

        Hist_Diff_norm(Outputs['recoilpatpfTrackMET_Pt'], Outputs['recoilpatpfTrackMET_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'Track', 0, ptMin_mid, ptMax_mid)
        Hist_Diff_norm(Outputs['recoilpatpfNoPUMET_Pt'], Outputs['recoilpatpfNoPUMET_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'NoPU', 2, ptMin_mid, ptMax_mid)
        Hist_Diff_norm(Outputs['recoilpatpfPUCorrectedMET_Pt'], Outputs['recoilpatpfPUCorrectedMET_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'PUCorrected', 3, ptMin_mid, ptMax_mid)
        Hist_Diff_norm(Outputs['recoilpatpfPUMET_Pt'], Outputs['recoilpatpfPUMET_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'PU', 4, ptMin_mid, ptMax_mid)
        Hist_Diff_norm(Outputs['recoilslimmedMETsPuppi_Pt'], Outputs['recoilslimmedMETsPuppi_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'Puppi', 5, ptMin_mid, ptMax_mid)
        Hist_Diff_norm(Outputs['recoilslimmedMETs_Pt'], Outputs['recoilslimmedMETs_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'PF', 1, ptMin_mid, ptMax_mid)
        Hist_Diff_norm(Outputs['NN_Pt'], Outputs['NN_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'NN', 6, ptMin_mid, ptMax_mid)


        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        handles, labels = ax.get_legend_handles_labels()
        handles.insert(0,mpatches.Patch(color='none', label=pTRangeString_mid))

        plt.ylabel('Counts')
        plt.xlabel('$|\\vec{U}+\\vec{p}_T^Z|$ in GeV')
        plt.xlim(diff_p_min,diff_p_max)

        ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
        plt.grid()
        #plt.ylim(ylimResMVAMin, ylimResMax)
        plt.savefig("%sHist_Diff_norm_mid.png"%(plotsD), bbox_inches="tight")



        fig=plt.figure(figsize=(10,6))
        fig.patch.set_facecolor('white')
        ax = plt.subplot(111)
        nbinsHist = 40

        Hist_Diff_norm(Outputs['recoilpatpfTrackMET_Pt'], Outputs['recoilpatpfTrackMET_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'Track', 0, ptMin_high, ptMax_high)
        Hist_Diff_norm(Outputs['recoilpatpfNoPUMET_Pt'], Outputs['recoilpatpfNoPUMET_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'NoPU', 2, ptMin_high, ptMax_high)
        Hist_Diff_norm(Outputs['recoilpatpfPUCorrectedMET_Pt'], Outputs['recoilpatpfPUCorrectedMET_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'PUCorrected', 3, ptMin_high, ptMax_high)
        Hist_Diff_norm(Outputs['recoilpatpfPUMET_Pt'], Outputs['recoilpatpfPUMET_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'PU', 4, ptMin_high, ptMax_high)
        Hist_Diff_norm(Outputs['recoilslimmedMETsPuppi_Pt'], Outputs['recoilslimmedMETsPuppi_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'Puppi', 5, ptMin_high, ptMax_high)
        Hist_Diff_norm(Outputs['recoilslimmedMETs_Pt'], Outputs['recoilslimmedMETs_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'PF', 1, ptMin_high, ptMax_high)
        Hist_Diff_norm(Outputs['NN_Pt'], Outputs['NN_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'NN', 6, ptMin_high, ptMax_high)


        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        handles, labels = ax.get_legend_handles_labels()
        handles.insert(0,mpatches.Patch(color='none', label=pTRangeString_high))

        plt.ylabel('Counts')
        plt.xlabel('$|\\vec{U}+\\vec{p}_T^Z|$ in GeV')
        plt.xlim(diff_p_min,diff_p_max)

        ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
        plt.grid()
        #plt.ylim(ylimResMVAMin, ylimResMax)
        plt.savefig("%sHist_Diff_norm_high.png"%(plotsD), bbox_inches="tight")
        '''

        fig=plt.figure(figsize=(10,6))
        fig.patch.set_facecolor('white')
        ax = plt.subplot(111)
        nbinsHist = 40

        Diff_norm_PV(Outputs['recoilpatpfTrackMET_Pt'], Outputs['recoilpatpfTrackMET_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'Track', 0, 0, 200)
        Diff_norm_PV(Outputs['recoilpatpfNoPUMET_Pt'], Outputs['recoilpatpfNoPUMET_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'NoPU', 2, 0, 200)
        Diff_norm_PV(Outputs['recoilpatpfPUCorrectedMET_Pt'], Outputs['recoilpatpfPUCorrectedMET_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'PUCorrected', 3, 0, 200)
        Diff_norm_PV(Outputs['recoilpatpfPUMET_Pt'], Outputs['recoilpatpfPUMET_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'PU', 4, 0, 200)
        Diff_norm_PV(Outputs['recoilslimmedMETsPuppi_Pt'], Outputs['recoilslimmedMETsPuppi_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'Puppi', 5, 0, 200)
        Diff_norm_PV(Outputs['recoilslimmedMETs_Pt'], Outputs['recoilslimmedMETs_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'PF', 1, 0, 200)
        Diff_norm_PV(Outputs['NN_Pt'], Outputs['NN_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'NN', 6, 0, 200)


        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        handles, labels = ax.get_legend_handles_labels()

        plt.xlabel('#PV')
        plt.ylabel('$|\\vec{U}+\\vec{p}_T^Z|$ in GeV')
        plt.xlim(0,50)

        ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
        plt.grid()
        #plt.ylim(ylimResMVAMin, ylimResMax)
        plt.savefig("%sDiff_norm_PV.png"%(plotsD), bbox_inches="tight")

        '''
        fig=plt.figure(figsize=(10,6))
        fig.patch.set_facecolor('white')
        ax = plt.subplot(111)
        nbinsHist = 40

        Diff_norm_PV(Outputs['recoilpatpfTrackMET_Pt'], Outputs['recoilpatpfTrackMET_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'Track', 0, ptMin_low, ptMax_low)
        Diff_norm_PV(Outputs['recoilpatpfNoPUMET_Pt'], Outputs['recoilpatpfNoPUMET_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'NoPU', 2, ptMin_low, ptMax_low)
        Diff_norm_PV(Outputs['recoilpatpfPUCorrectedMET_Pt'], Outputs['recoilpatpfPUCorrectedMET_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'PUCorrected', 3, ptMin_low, ptMax_low)
        Diff_norm_PV(Outputs['recoilpatpfPUMET_Pt'], Outputs['recoilpatpfPUMET_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'PU', 4, ptMin_low, ptMax_low)
        Diff_norm_PV(Outputs['recoilslimmedMETsPuppi_Pt'], Outputs['recoilslimmedMETsPuppi_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'Puppi', 5, ptMin_low, ptMax_low)
        Diff_norm_PV(Outputs['recoilslimmedMETs_Pt'], Outputs['recoilslimmedMETs_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'PF', 1, ptMin_low, ptMax_low)
        Diff_norm_PV(Outputs['NN_Pt'], Outputs['NN_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'NN', 6, ptMin_low, ptMax_low)


        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        handles, labels = ax.get_legend_handles_labels()
        handles.insert(0,mpatches.Patch(color='none', label=pTRangeString_low))

        plt.xlabel('#PV')
        plt.ylabel('$|\\vec{U}+\\vec{p}_T^Z|$ in GeV')
        plt.xlim(diff_p_min,diff_p_max)

        ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
        plt.grid()
        #plt.ylim(ylimResMVAMin, ylimResMax)
        plt.savefig("%sDiff_norm_PV_low.png"%(plotsD), bbox_inches="tight")

        fig=plt.figure(figsize=(10,6))
        fig.patch.set_facecolor('white')
        ax = plt.subplot(111)
        nbinsHist = 40

        Diff_norm_PV(Outputs['recoilpatpfTrackMET_Pt'], Outputs['recoilpatpfTrackMET_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'Track', 0, ptMin_mid, ptMax_mid)
        Diff_norm_PV(Outputs['recoilpatpfNoPUMET_Pt'], Outputs['recoilpatpfNoPUMET_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'NoPU', 2, ptMin_mid, ptMax_mid)
        Diff_norm_PV(Outputs['recoilpatpfPUCorrectedMET_Pt'], Outputs['recoilpatpfPUCorrectedMET_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'PUCorrected', 3, ptMin_mid, ptMax_mid)
        Diff_norm_PV(Outputs['recoilpatpfPUMET_Pt'], Outputs['recoilpatpfPUMET_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'PU', 4, ptMin_mid, ptMax_mid)
        Diff_norm_PV(Outputs['recoilslimmedMETsPuppi_Pt'], Outputs['recoilslimmedMETsPuppi_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'Puppi', 5, ptMin_mid, ptMax_mid)
        Diff_norm_PV(Outputs['recoilslimmedMETs_Pt'], Outputs['recoilslimmedMETs_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'PF', 1, ptMin_mid, ptMax_mid)
        Diff_norm_PV(Outputs['NN_Pt'], Outputs['NN_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'NN', 6, ptMin_mid, ptMax_mid)


        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        handles, labels = ax.get_legend_handles_labels()
        handles.insert(0,mpatches.Patch(color='none', label=pTRangeString_mid))

        plt.xlabel('#PV')
        plt.ylabel('$|\\vec{U}+\\vec{p}_T^Z|$ in GeV')
        plt.xlim(diff_p_min,diff_p_max)

        ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
        plt.grid()
        #plt.ylim(ylimResMVAMin, ylimResMax)
        plt.savefig("%sDiff_norm_PV_mid.png"%(plotsD), bbox_inches="tight")


        fig=plt.figure(figsize=(10,6))
        fig.patch.set_facecolor('white')
        ax = plt.subplot(111)
        nbinsHist = 40

        Diff_norm_PV(Outputs['recoilpatpfTrackMET_Pt'], Outputs['recoilpatpfTrackMET_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'Track', 0, ptMin_high, ptMax_high)
        Diff_norm_PV(Outputs['recoilpatpfNoPUMET_Pt'], Outputs['recoilpatpfNoPUMET_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'NoPU', 2, ptMin_high, ptMax_high)
        Diff_norm_PV(Outputs['recoilpatpfPUCorrectedMET_Pt'], Outputs['recoilpatpfPUCorrectedMET_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'PUCorrected', 3, ptMin_high, ptMax_high)
        Diff_norm_PV(Outputs['recoilpatpfPUMET_Pt'], Outputs['recoilpatpfPUMET_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'PU', 4, ptMin_high, ptMax_high)
        Diff_norm_PV(Outputs['recoilslimmedMETsPuppi_Pt'], Outputs['recoilslimmedMETsPuppi_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'Puppi', 5, ptMin_high, ptMax_high)
        Diff_norm_PV(Outputs['recoilslimmedMETs_Pt'], Outputs['recoilslimmedMETs_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'PF', 1, ptMin_high, ptMax_high)
        Diff_norm_PV(Outputs['NN_Pt'], Outputs['NN_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'NN', 6, ptMin_high, ptMax_high)


        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        handles, labels = ax.get_legend_handles_labels()
        handles.insert(0,mpatches.Patch(color='none', label=pTRangeString_high))

        plt.xlabel('#PV')
        plt.ylabel('$|\\vec{U}+\\vec{p}_T^Z|$ in GeV')
        plt.xlim(diff_p_min,diff_p_max)

        ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
        plt.grid()
        #plt.ylim(ylimResMVAMin, ylimResMax)
        plt.savefig("%sDiff_norm_PV_high.png"%(plotsD), bbox_inches="tight")
        '''


        fig=plt.figure(figsize=(10,6))
        fig.patch.set_facecolor('white')
        ax = plt.subplot(111)
        nbinsHist = 40

        Hist_Diff_norm(Outputs['recoilpatpfTrackMET_Pt'], Outputs['recoilpatpfTrackMET_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'Track', 0, 100, 200)
        Hist_Diff_norm(Outputs['recoilpatpfNoPUMET_Pt'], Outputs['recoilpatpfNoPUMET_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'NoPU', 2, 100, 200)
        Hist_Diff_norm(Outputs['recoilpatpfPUCorrectedMET_Pt'], Outputs['recoilpatpfPUCorrectedMET_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'PUCorrected', 3, 100, 200)
        Hist_Diff_norm(Outputs['recoilpatpfPUMET_Pt'], Outputs['recoilpatpfPUMET_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'PU', 4, 100, 200)
        Hist_Diff_norm(Outputs['recoilslimmedMETsPuppi_Pt'], Outputs['recoilslimmedMETsPuppi_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'Puppi', 5, 100, 200)
        Hist_Diff_norm(Outputs['recoilslimmedMETs_Pt'], Outputs['recoilslimmedMETs_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'PF', 1, 100, 200)
        Hist_Diff_norm(Outputs['NN_Pt'], Outputs['NN_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'NN', 6, 100, 200)


        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        handles, labels = ax.get_legend_handles_labels()
        handles.insert(0,mpatches.Patch(color='none', label=pTRangeString_very_high))

        plt.ylabel('Counts')
        plt.xlabel('$|\\vec{U}+\\vec{p}_T^Z|$ in GeV')
        plt.xlim(diff_p_min,400)

        ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
        plt.grid()
        #plt.ylim(ylimResMVAMin, ylimResMax)
        plt.savefig("%sHist_Diff_norm_very_high.png"%(plotsD), bbox_inches="tight")


        Norm_Diff = HM_Diff_norm(Outputs['recoilslimmedMETs_Pt'], Outputs['recoilslimmedMETs_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'PF', 1)
        '''
        plt.clf()
        plt.figure()
        #plt.suptitle('x-Korrektur Prediction-Target ')
        plt.ylabel("$ |\\vec{U}+\\vec{p}_T^Z| $")
        plt.xlabel("Response")

        heatmap, xedges, yedges = np.histogram2d(  Norm_Diff[getResponseIdx('NN_LongZ')], getResponse('NN_LongZ'),  bins=50,
                                                 range=[[0,20],
                                                        [0.75,1.25]])
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        #plt.clf()
        HM =plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm(), aspect='auto')
        plt.colorbar(HM)
        plt.legend()
        plt.savefig("%sHM_NN_Response_Normdiff.png"%(plotsD), bbox_inches="tight")

        plt.clf()
        plt.figure()
        #plt.suptitle('x-Korrektur Prediction-Target ')
        plt.ylabel("$ \\frac{|\\vec{U}+\\vec{p}_T^Z|}{|-\\vec{p}_T^Z|} $")
        plt.xlabel("Response")
        rel_Norm_Diff = np.divide(HM_Diff_norm(Outputs['NN_Pt'], Outputs['NN_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'NN', 6), Outputs[Target_Pt])
        heatmap, xedges, yedges = np.histogram2d(  getResponse('NN_LongZ')[getResponseIdx('NN_LongZ')== ~np.isnan(rel_Norm_Diff)], rel_Norm_Diff[getResponseIdx('NN_LongZ')== ~np.isnan(rel_Norm_Diff)],  bins=50,
                                                 range=[[0,2],
                                                        [0,2]])
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        #plt.clf()
        HM =plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm(), aspect='auto')
        plt.colorbar(HM)
        plt.legend()
        plt.savefig("%sHM_NN_Response_relDiffnorm.png"%(plotsD), bbox_inches="tight")


        plt.clf()
        plt.figure()
        #plt.suptitle('x-Korrektur Prediction-Target ')
        plt.ylabel("$ \\frac{|\\vec{U}+\\vec{p}_T^Z|}{|-\\vec{p}_T^Z|} $")
        plt.xlabel("Response")
        rel_Norm_Diff = np.divide(HM_Diff_norm(Outputs['recoilslimmedMETs_Pt'], Outputs['recoilslimmedMETs_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'PF', 1), Outputs[Target_Pt])
        heatmap, xedges, yedges = np.histogram2d(  getResponse('recoilslimmedMETs_LongZ')[getResponseIdx('recoilslimmedMETs_Pt')== ~np.isnan(rel_Norm_Diff)], rel_Norm_Diff[getResponseIdx('recoilslimmedMETs_Pt')== ~np.isnan(rel_Norm_Diff)],  bins=50,
                                                 range=[[0,2],
                                                        [0,2]])
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        #plt.clf()
        HM =plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm(), aspect='auto')
        plt.colorbar(HM)
        plt.legend()
        plt.savefig("%sHM_PF_Response_relDiffnorm.png"%(plotsD), bbox_inches="tight")


        plt.clf()
        plt.figure()
        #plt.suptitle('x-Korrektur Prediction-Target ')
        plt.ylabel("$ \\frac{|\\vec{U}+\\vec{p}_T^Z|}{|-\\vec{p}_T^Z|} $")
        plt.xlabel("$U_{\parallel}+\\vec{p}_T^Z$")
        rel_Norm_Diff = np.divide(HM_Diff_norm(Outputs['NN_Pt'], Outputs['NN_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'NN', 6), Outputs[Target_Pt])
        Resolution_para = -Outputs['NN_LongZ']-Outputs[Target_Pt]
        heatmap, xedges, yedges = np.histogram2d(  Resolution_para[~np.isnan(rel_Norm_Diff)], rel_Norm_Diff[~np.isnan(rel_Norm_Diff)],  bins=50,
                                                 range=[[-3,3],
                                                        [-4,4]])
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        #plt.clf()
        HM =plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm(), aspect='auto')
        plt.colorbar(HM)
        plt.legend()
        plt.savefig("%sHM_NN_Resolution_para_relDiffnorm.png"%(plotsD), bbox_inches="tight")


        plt.clf()
        plt.figure()
        #plt.suptitle('x-Korrektur Prediction-Target ')
        plt.ylabel("$ \\frac{|\\vec{U}+\\vec{p}_T^Z|}{|-\\vec{p}_T^Z|} $")
        plt.xlabel("$U_{\parallel}+\\vec{p}_T^Z$")
        rel_Norm_Diff = np.divide(HM_Diff_norm(Outputs['recoilslimmedMETs_Pt'], Outputs['recoilslimmedMETs_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'PF', 1), Outputs[Target_Pt])
        Resolution_para = -Outputs['recoilslimmedMETs_LongZ']-Outputs[Target_Pt]
        heatmap, xedges, yedges = np.histogram2d(  Resolution_para[~np.isnan(rel_Norm_Diff)], rel_Norm_Diff[~np.isnan(rel_Norm_Diff)],  bins=50,
                                                 range=[[-3,3],
                                                        [-4,4]])
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        #plt.clf()
        HM =plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm(), aspect='auto')
        plt.colorbar(HM)
        plt.legend()
        plt.savefig("%sHM_PF_Resolution_para_relDiffnorm.png"%(plotsD), bbox_inches="tight")

        plt.clf()
        plt.figure()
        #plt.suptitle('x-Korrektur Prediction-Target ')
        plt.ylabel("$ \\frac{|\\vec{U}+\\vec{p}_T^Z|}{|-\\vec{p}_T^Z|} $")
        plt.xlabel("$U_{\perp}$")
        rel_Norm_Diff = np.divide(HM_Diff_norm(Outputs['NN_Pt'], Outputs['NN_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'NN', 6), Outputs[Target_Pt])
        Resolution_perp = Outputs['NN_PerpZ']
        heatmap, xedges, yedges = np.histogram2d(  Resolution_perp[~np.isnan(rel_Norm_Diff)], rel_Norm_Diff[~np.isnan(rel_Norm_Diff)],  bins=50,
                                                 range=[[0,3],
                                                        [0,4]])
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        #plt.clf()
        HM =plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm(), aspect='auto')
        plt.colorbar(HM)
        plt.legend()
        plt.savefig("%sHM_NN_Resolution_perp_relDiffnorm.png"%(plotsD), bbox_inches="tight")


        plt.clf()
        plt.figure()
        #plt.suptitle('x-Korrektur Prediction-Target ')
        plt.ylabel("$ \\frac{|\\vec{U}+\\vec{p}_T^Z|}{|-\\vec{p}_T^Z|} $")
        plt.xlabel("$U_{\perp}$")
        rel_Norm_Diff = np.divide(HM_Diff_norm(Outputs['recoilslimmedMETs_Pt'], Outputs['recoilslimmedMETs_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'PF', 1), Outputs[Target_Pt])
        Resolution_perp = Outputs['recoilslimmedMETs_PerpZ']
        heatmap, xedges, yedges = np.histogram2d(  Resolution_perp[~np.isnan(rel_Norm_Diff)], rel_Norm_Diff[~np.isnan(rel_Norm_Diff)],  bins=50,
                                                 range=[[0,3],
                                                        [0,4]])
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        #plt.clf()
        HM =plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm(), aspect='auto')
        plt.colorbar(HM)
        plt.legend()
        plt.savefig("%sHM_PF_Resolution_perp_relDiffnorm.png"%(plotsD), bbox_inches="tight")


        plt.clf()
        plt.figure()
        #plt.suptitle('x-Korrektur Prediction-Target ')
        plt.xlabel("$ |\\vec{U}+\\vec{p}_T^Z| $")
        plt.ylabel("Response")
        #Norm_Diff = HM_Diff_norm(Outputs['recoilslimmedMETs_Pt'], Outputs['recoilslimmedMETs_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'PF', 1)
        heatmap, xedges, yedges = np.histogram2d(  Norm_Diff[getResponseIdx('recoilslimmedMETs_LongZ')], getResponse('recoilslimmedMETs_LongZ'),  bins=50,
                                                 range=[[0,20],
                                                        [0,2]])
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        #plt.clf()
        HM =plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm(), aspect='auto')
        plt.colorbar(HM)
        plt.legend()
        plt.savefig("%sHM_Response_PF_Diff_norm.png"%(plotsD), bbox_inches="tight")
        plt.close()







        fig=plt.figure(figsize=(10,6))
        fig.patch.set_facecolor('white')
        ax = plt.subplot(111)



        Histogram_Angle(Outputs['recoilpatpfNoPUMET_Phi'], 'No PU MET',0)
        Histogram_Angle(Outputs['recoilpatpfPUCorrectedMET_Phi'],'PU corrected MET',1)
        Histogram_Angle(Outputs['recoilpatpfPUMET_Phi'], 'PU MET',2)
        Histogram_Angle(Outputs['recoilpatpfTrackMET_Phi'], 'Track MET',3)
        Histogram_Angle(Outputs['recoilslimmedMETsPuppi_Phi'], 'Puppi MET',4)
        Histogram_Angle(Outputs['recoilslimmedMETs_Phi'], 'PF MET', 1)

        Histogram_Angle(Outputs['NN_Phi'], 'NN MET', 6)
        Histogram_Angle(Outputs[Target_Phi], 'Targets', 7)

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        handles, labels = ax.get_legend_handles_labels()
        #handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

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







        print('shape Outputs[NN_Pt]', Outputs['NN_Pt'].shape)
        print('nans Outputs[NN_Pt]', sum(np.isnan(Outputs['NN_Pt'])))
        print('shape Outputs[NN_Pt]+sum(Outputs[Target_Pt]==0)', len(Outputs['NN_Pt'])+sum(Outputs[Target_Pt]==0))

        '''
        fig=plt.figure(figsize=(10,6))
        fig.patch.set_facecolor('white')
        ax = plt.subplot(111)
        nbinsHist = 40

        Hist_Diff_norm(Outputs['recoilpatpfTrackMET_Pt'], Outputs['recoilpatpfTrackMET_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'Track', 0)
        Hist_Diff_norm(Outputs['recoilpatpfNoPUMET_Pt'], Outputs['recoilpatpfNoPUMET_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'NoPU', 2)
        Hist_Diff_norm(Outputs['recoilpatpfPUCorrectedMET_Pt'], Outputs['recoilpatpfPUCorrectedMET_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'PUCorrected', 3)
        Hist_Diff_norm(Outputs['recoilpatpfPUMET_Pt'], Outputs['recoilpatpfPUMET_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'PU', 4)
        Hist_Diff_norm(Outputs['recoilslimmedMETsPuppi_Pt'], Outputs['recoilslimmedMETsPuppi_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'Puppi', 5)
        Hist_Diff_norm(Outputs['LongZCorrectedRecoil_Pt'], Outputs['LongZCorrectedRecoil_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'GBRT', 7)
        Hist_Diff_norm(Outputs['recoilslimmedMETs_Pt'], Outputs['recoilslimmedMETs_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'PF', 1)
        print("Zeile 110")
        #Hist_Diff_norm(Outputs['NN_Pt'], Outputs['NN_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'NN', 6)
        print("Zeile 112")

        norm_ = np.sqrt(np.square(Outputs['NN_Pt']-pol2kar_x(Outputs[Target_Pt], Outputs[Target_Phi]))+np.square(Outputs['NN_Phi']-pol2kar_y(Outputs[Target_Pt], Outputs[Target_Phi])))
        print("Zeile 115")
        param = rayleigh.fit(norm_)
        bins = np.linspace(0,50, 50/nbinsHist)
        bins = range(0, 50 + 1, 1)
        #pdf_fitted = rayleigh.pdf(bins, loc=param[-2], scale=len(norm_))

        pdf_fitted2 = rayleigh.pdf(bins, param[:-2], loc=0, scale=1) * len(norm_)
        plt.plot(pdf_fitted2, label='Rayleigh-Fit 2')
        pdf_fitted = rayleigh.pdf(bins, param[:-2], loc=0, scale=1)
        plt.plot(pdf_fitted, label='Rayleigh-Fit ')
        #pdf_fitted3 = rayleigh.pdf(bins, param[:-2], loc=0, scale=len(norm_))
        #plt.plot(pdf_fitted3, label='Rayleigh-Fit 3')
        #pdf_fitted4 = rayleigh.pdf(bins,  loc=0, scale=len(norm_))
        #plt.plot(pdf_fitted4, label='Rayleigh-Fit 4')
        #mean_, std_ = np.mean(norm_), np.std(norm_)
        #ray = (bins/mean_)*np.exp(-np.multiply(bins,bins)/(2*std_*std_))
        #plt.plot(ray, label='Rayleigh-Fit 4')

        print("plot rayleigh")
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        handles, labels = ax.get_legend_handles_labels()

        plt.ylabel('Counts')
        plt.xlabel('$|\\vec{U}+\\vec{p}_T^Z|$ in GeV')
        plt.xlim(diff_p_min,diff_p_max)
        #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_{T}^Z} \\right) $ in GeV')
        #plt.title(' Histogram $p_{T,x}$')
        #plt.text('$p_T$ range restriction')

        ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
        plt.grid()
        #plt.ylim(ylimResMVAMin, ylimResMax)
        plt.savefig("%sHist_Diff_norm_Rayleighfit.png"%(plotsD), bbox_inches="tight")
        '''





        fig=plt.figure(figsize=(10,6))
        fig.patch.set_facecolor('white')
        ax = plt.subplot(111)
        nbinsHist = 40

        plt.hist(InputsTargets['Track'][0,:], bins=nbinsHist, range=[-50, 50], label='Track, mean=%.2f'%np.mean(InputsTargets['Track'][0,:]), histtype='step', ec=colors_InOut[0])
        plt.hist(InputsTargets['NoPU'][0,:], bins=nbinsHist, range=[-50, 50], label='NoPU, mean=%.2f'%np.mean(InputsTargets['NoPU'][0,:]), histtype='step', ec=colors_InOut[2])
        plt.hist(InputsTargets['PUCorrected'][0,:], bins=nbinsHist, range=[-50, 50], label='PUCorrected, mean=%.2f'%np.mean(InputsTargets['PUCorrected'][0,:]), histtype='step', ec=colors_InOut[3])
        plt.hist(InputsTargets['PU'][0,:], bins=nbinsHist, range=[-50, 50], label='PU, mean=%.2f'%np.mean(InputsTargets['PU'][0,:]), histtype='step', ec=colors_InOut[4])
        plt.hist(InputsTargets['Puppi'][0,:], bins=nbinsHist, range=[-50, 50], label='Puppi, mean=%.2f'%np.mean(InputsTargets['Puppi'][0,:]), histtype='step', ec=colors_InOut[5])
        plt.hist(InputsTargets['PF'][0,:], bins=nbinsHist, range=[-50, 50], label='PF, mean=%.2f'%np.mean(InputsTargets['PF'][0,:]), histtype='step', ec=colors_InOut[1], linewidth=1.5)
        plt.hist(pol2kar_x(Outputs[Target_Pt],Outputs[Target_Phi]), bins=nbinsHist, range=[-50, 50], label='Target, mean=%.2f'%np.mean(Outputs[Target_Pt]), histtype='step', ec=colors_InOut[6], linewidth=1.5)
        plt.hist(pol2kar_x(Outputs['NN_Pt'], Outputs['NN_Phi']), bins=nbinsHist, range=[-50, 50], label='Prediction, mean=%.2f'%np.mean(Outputs['NN_Pt']), histtype='step', ec=colors_InOut[7], linewidth=1.5)



        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        handles, labels = ax.get_legend_handles_labels()
        #handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

        plt.ylabel('Counts')
        plt.xlabel('$p_{T,x}$ in GeV')
        plt.xlim(-50,50)
        #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_{T}^Z} \\right) $ in GeV')
        #plt.title(' Histogram $p_{T,x}$')
        #plt.text('$p_T$ range restriction')

        ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
        plt.grid()
        #plt.ylim(ylimResMVAMin, ylimResMax)
        plt.savefig("%sHist_x.png"%(plotsD), bbox_inches="tight")
        plt.close()



        fig=plt.figure(figsize=(10,6))
        fig.patch.set_facecolor('white')
        ax = plt.subplot(111)
        nbinsHist = 40

        plt.hist(InputsTargets['Track'][1,:], bins=nbinsHist, range=[-50, 50], label='Track, mean=%.2f'%np.mean(InputsTargets['Track'][1,:]), histtype='step', ec=colors_InOut[0])
        plt.hist(InputsTargets['NoPU'][1,:], bins=nbinsHist, range=[-50, 50], label='NoPU, mean=%.2f'%np.mean(InputsTargets['NoPU'][1,:]), histtype='step', ec=colors_InOut[2])
        plt.hist(InputsTargets['PUCorrected'][1,:], bins=nbinsHist, range=[-50, 50], label='PUCorrected, mean=%.2f'%np.mean(InputsTargets['PUCorrected'][1,:]), histtype='step', ec=colors_InOut[3])
        plt.hist(InputsTargets['PU'][1,:], bins=nbinsHist, range=[-50, 50], label='PU, mean=%.2f'%np.mean(InputsTargets['PU'][1,:]), histtype='step', ec=colors_InOut[4])
        plt.hist(InputsTargets['Puppi'][1,:], bins=nbinsHist, range=[-50, 50], label='Puppi, mean=%.2f'%np.mean(InputsTargets['Puppi'][1,:]), histtype='step', ec=colors_InOut[5])
        plt.hist(InputsTargets['PF'][1,:], bins=nbinsHist, range=[-50, 50], label='PF, mean=%.2f'%np.mean(InputsTargets['PF'][1,:]), histtype='step', ec=colors_InOut[1], linewidth=1.5)
        plt.hist(pol2kar_y(Outputs[Target_Pt], Outputs[Target_Phi]), bins=nbinsHist, range=[-50, 50], label='Target, mean=%.2f'%np.mean(Outputs[Target_Phi]), histtype='step', ec=colors_InOut[6], linewidth=1.5)
        plt.hist(pol2kar_y(Outputs['NN_Pt'], Outputs['NN_Phi']), bins=nbinsHist, range=[-50, 50], label='Prediction, mean=%.2f'%np.mean(Outputs['NN_Phi']), histtype='step', ec=colors_InOut[7], linewidth=1.5)


        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        handles, labels = ax.get_legend_handles_labels()
        handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

        plt.ylabel('Counts')
        plt.xlabel('$ p_{T,y}$ in GeV')
        plt.xlim(-50,50)
        #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_{T}^Z} \\right) $ in GeV')
        #plt.title(' Histogram $ p_{T,y}$')
        #plt.text('$p_T$ range restriction')

        ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
        plt.grid()
        #plt.ylim(ylimResMVAMin, ylimResMax)
        plt.savefig("%sHist_y.png"%(plotsD), bbox_inches="tight")
        plt.close()


        fig=plt.figure(figsize=(10,6))
        fig.patch.set_facecolor('white')
        ax = plt.subplot(111)
        nbinsHist = 40
        Ind = (Outputs[Target_Pt]>0) & (Outputs[Target_Pt]<200) & (Outputs['NVertex']<=50)
        print(sum(Ind))
        plt.hist(np.subtract(InputsTargets['Track'][0,:], Outputs[Target_Pt][Ind]), bins=nbinsHist, range=[-50, 50], label='Track, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Track'][0,:], Outputs[Target_Pt][Ind])), np.std(np.subtract( InputsTargets['Track'][0,:] , Outputs[Target_Pt][Ind]))), histtype='step', ec=colors_InOut[0])
        plt.hist(np.subtract(InputsTargets['NoPU'][0,:], Outputs[Target_Pt][Ind]), bins=nbinsHist, range=[-50, 50], label='NoPU, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['NoPU'][0,:], Outputs[Target_Pt][Ind])), np.std(np.subtract( InputsTargets['NoPU'][0,:] , Outputs[Target_Pt][Ind]))), histtype='step', ec=colors_InOut[2])
        plt.hist(np.subtract(InputsTargets['PUCorrected'][0,:], Outputs[Target_Pt][Ind]), bins=nbinsHist, range=[-50, 50], label='PUCorrected, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PUCorrected'][0,:], Outputs[Target_Pt][Ind])), np.std(np.subtract(InputsTargets['PUCorrected'][0,:]  , Outputs[Target_Pt][Ind]))), histtype='step', ec=colors_InOut[3])
        plt.hist(np.subtract(InputsTargets['PU'][0,:], Outputs[Target_Pt][Ind]), bins=nbinsHist, range=[-50, 50], label='PU, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PU'][0,:], Outputs[Target_Pt][Ind])), np.std(np.subtract(InputsTargets['PU'][0,:]  , Outputs[Target_Pt][Ind]))), histtype='step', ec=colors_InOut[4])
        plt.hist(np.subtract(InputsTargets['Puppi'][0,:], Outputs[Target_Pt][Ind]), bins=nbinsHist, range=[-50, 50], label='Puppi, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Puppi'][0,:], Outputs[Target_Pt][Ind])), np.std(np.subtract(InputsTargets['Puppi'][0,:]  , Outputs[Target_Pt][Ind]))), histtype='step', ec=colors_InOut[5])
        #plt.hist(np.subtract(pol2pol2kar_x(Outputs['LongZCorrectedRecoil_Pt'],Outputs['LongZCorrectedRecoil_Phi']), Outputs[Target_Pt][Ind]), bins=nbinsHist, range=[-50, 50], label='Puppi, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Puppi'][0,:], Outputs[Target_Pt][Ind])), np.std(np.subtract(InputsTargets['Puppi'][0,:]  , Outputs[Target_Pt][Ind]))), histtype='step', ec=colors_InOut[5])

        plt.hist(np.subtract(InputsTargets['PF'][0,:], Outputs[Target_Pt][Ind]), bins=nbinsHist, range=[-50, 50], label='PF, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PF'][0,:], Outputs[Target_Pt][Ind])), np.std(np.subtract(InputsTargets['PF'][0,:]  , Outputs[Target_Pt][Ind]))), histtype='step', ec=colors_InOut[1], linewidth=1.5)
        plt.hist(NN_Diff_x[Ind], bins=nbinsHist, range=[-50, 50], label='NN, mean=%.2f$\pm$%.2f'%(np.mean(NN_Diff_x[Ind]), np.std(NN_Diff_x[Ind])), histtype='step', ec=colors_InOut[7], linewidth=1.5)


        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        handles, labels = ax.get_legend_handles_labels()
        handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

        plt.ylabel('Counts')
        plt.xlabel('$\\Delta p_{T,x}$ in GeV')
        plt.xlim(-50,50)
        #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_{T}^Z} \\right) $ in GeV')
        #plt.title(' Histogram Deviation to Target  $\\Delta p_{T,x}$')
        #plt.text('$p_T$ range restriction')

        ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
        plt.grid()
        #plt.ylim(ylimResMVAMin, ylimResMax)
        plt.savefig("%sHist_Delta_x1.png"%(plotsD), bbox_inches="tight")
        plt.close()

        fig=plt.figure(figsize=(10,6))
        fig.patch.set_facecolor('white')
        ax = plt.subplot(111)
        nbinsHist = 40
        #pTnorm = np.sqrt( np.multiply(Outputs[Target_Pt],Outputs[Target_Pt])+ np.multiply(Outputs[Target_Phi],Outputs[Target_Phi]))
        #IdxpT = Outputs.index[(Outputs[Target_Pt] <= 200) & (Outputs[Target_Pt] > 0) & ].tolist()

        plt.hist(np.subtract(InputsTargets['Track'][0,:], Targets[Ind,0]), bins=nbinsHist, range=[-50, 50], label='Track, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Track'][0,:], Targets[Ind,0])), np.std(np.subtract( InputsTargets['Track'][0,:] , Targets[Ind,0]))), histtype='step', ec=colors_InOut[0])
        plt.hist(np.subtract(InputsTargets['NoPU'][0,:], Targets[Ind,0]), bins=nbinsHist, range=[-50, 50], label='NoPU, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['NoPU'][0,:], Targets[Ind,0])), np.std(np.subtract( InputsTargets['NoPU'][0,:] , Targets[Ind,0]))), histtype='step', ec=colors_InOut[2])
        plt.hist(np.subtract(InputsTargets['PUCorrected'][0,:], Targets[Ind,0]), bins=nbinsHist, range=[-50, 50], label='PUCorrected, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PUCorrected'][0,:], Targets[Ind,0])), np.std(np.subtract(InputsTargets['PUCorrected'][0,:]  , Targets[Ind,0]))), histtype='step', ec=colors_InOut[3])
        plt.hist(np.subtract(InputsTargets['PU'][0,:], Targets[Ind,0]), bins=nbinsHist, range=[-50, 50], label='PU, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PU'][0,:], Targets[Ind,0])), np.std(np.subtract(InputsTargets['PU'][0,:]  , Targets[Ind,0]))), histtype='step', ec=colors_InOut[4])
        plt.hist(np.subtract(InputsTargets['Puppi'][0,:], Targets[Ind,0]), bins=nbinsHist, range=[-50, 50], label='Puppi, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Puppi'][0,:], Targets[Ind,0])), np.std(np.subtract(InputsTargets['Puppi'][0,:]  , Targets[Ind,0]))), histtype='step', ec=colors_InOut[5])
        #plt.hist(np.subtract(pol2pol2kar_x(Outputs['LongZCorrectedRecoil_Pt'],Outputs['LongZCorrectedRecoil_Phi']), Outputs[Target_Pt]), bins=nbinsHist, range=[-50, 50], label='Puppi, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Puppi'][0,:], Outputs[Target_Pt])), np.std(np.subtract(InputsTargets['Puppi'][0,:]  , Outputs[Target_Pt]))), histtype='step', ec=colors_InOut[5])

        plt.hist(np.subtract(InputsTargets['PF'][0,:], Targets[Ind,0]), bins=nbinsHist, range=[-50, 50], label='PF, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PF'][0,:], Targets[Ind,0])), np.std(np.subtract(InputsTargets['PF'][0,:]  , Targets[Ind,0]))), histtype='step', ec=colors_InOut[0], linewidth=1.5)
        plt.hist(NN_Diff_x[:], bins=nbinsHist, range=[-50, 50], label='NN, mean=%.2f$\pm$%.2f'%(np.mean(NN_Diff_y), np.std(NN_Diff_y)), histtype='step', ec=colors_InOut[7], linewidth=1.5)


        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        handles, labels = ax.get_legend_handles_labels()
        handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))


        plt.ylabel('Counts')
        plt.xlabel('$\\Delta p_{T,x}$ in GeV')
        plt.xlim(-50,50)

        ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
        plt.grid()
        plt.savefig("%sHist_Delta_x2.png"%(plotsD), bbox_inches="tight")
        plt.close()


        fig=plt.figure(figsize=(10,6))
        fig.patch.set_facecolor('white')
        ax = plt.subplot(111)
        nbinsHist = 40


        plt.hist(np.subtract(InputsTargets['Track'][1,:], Targets[Ind,1]), bins=nbinsHist, range=[-200, 200], label='Track, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Track'][1,:], Targets[Ind,1])), np.std(np.subtract( InputsTargets['Track'][1,:] , Targets[Ind,1]))), histtype='step', ec=colors_InOut[1])
        plt.hist(np.subtract(InputsTargets['NoPU'][1,:], Targets[Ind,1]), bins=nbinsHist, range=[-200, 200], label='NoPU, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['NoPU'][1,:], Targets[Ind,1])), np.std(np.subtract( InputsTargets['NoPU'][1,:] , Targets[Ind,1]))), histtype='step', ec=colors_InOut[2])
        plt.hist(np.subtract(InputsTargets['PUCorrected'][1,:], Targets[Ind,1]), bins=nbinsHist, range=[-200, 200], label='PUCorrected, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PUCorrected'][1,:], Targets[Ind,1])), np.std(np.subtract(InputsTargets['PUCorrected'][1,:]  , Targets[Ind,1]))), histtype='step', ec=colors_InOut[3])
        plt.hist(np.subtract(InputsTargets['PU'][1,:], Targets[Ind,1]), bins=nbinsHist, range=[-200, 200], label='PU, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PU'][1,:], Targets[Ind,1])), np.std(np.subtract(InputsTargets['PU'][1,:]  , Targets[Ind,1]))), histtype='step', ec=colors_InOut[4])
        plt.hist(np.subtract(InputsTargets['Puppi'][1,:], Targets[Ind,1]), bins=nbinsHist, range=[-200, 200], label='Puppi, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Puppi'][1,:], Targets[Ind,1])), np.std(np.subtract(InputsTargets['Puppi'][1,:]  , Targets[Ind,1]))), histtype='step', ec=colors_InOut[5])
        #plt.hist(np.subtract(pol2pol2kar_x(Outputs['LongZCorrectedRecoil_Pt'],Outputs['LongZCorrectedRecoil_Phi']), Outputs[Target_Pt]), bins=nbinsHist, range=[-200, 200], label='Puppi, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Puppi'][1,:], Outputs[Target_Pt])), np.std(np.subtract(InputsTargets['Puppi'][1,:]  , Outputs[Target_Pt]))), histtype='step', ec=colors_InOut[5])

        plt.hist(np.subtract(InputsTargets['PF'][1,:], Targets[Ind,1]), bins=nbinsHist, range=[-200, 200], label='PF, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PF'][1,:], Targets[Ind,1])), np.std(np.subtract(InputsTargets['PF'][1,:]  , Targets[Ind,1]))), histtype='step', ec=colors_InOut[1], linewidth=1.5)
        plt.hist(NN_Diff_y, bins=nbinsHist, range=[-200, 200], label='NN, mean=%.2f$\pm$%.2f'%(np.mean(NN_Diff_y), np.std(NN_Diff_y)), histtype='step', ec=colors_InOut[7], linewidth=1.5)


        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        handles, labels = ax.get_legend_handles_labels()
        handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))


        plt.ylabel('Counts')
        plt.xlabel('$\\Delta p_{T,y}$ in GeV')
        plt.xlim(-50,50)

        ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
        plt.grid()
        plt.savefig("%sHist_Delta_y.png"%(plotsD), bbox_inches="tight")
        plt.close()

        fig=plt.figure(figsize=(10,6))
        fig.patch.set_facecolor('white')
        ax = plt.subplot(111)
        nbinsHist = 40


        plt.hist(np.subtract(InputsTargets['Track'][0,:], Targets[Ind,0]), bins=nbinsHist, range=[-200, 200], label='Track, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Track'][0,:], Targets[Ind,0])), np.std(np.subtract( InputsTargets['Track'][0,:] , Targets[Ind,0]))), histtype='step', ec=colors_InOut[1])
        plt.hist(np.subtract(InputsTargets['NoPU'][0,:], Targets[Ind,0]), bins=nbinsHist, range=[-200, 200], label='NoPU, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['NoPU'][0,:], Targets[Ind,0])), np.std(np.subtract( InputsTargets['NoPU'][0,:] , Targets[Ind,0]))), histtype='step', ec=colors_InOut[2])
        plt.hist(np.subtract(InputsTargets['PUCorrected'][0,:], Targets[Ind,0]), bins=nbinsHist, range=[-200, 200], label='PUCorrected, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PUCorrected'][0,:], Targets[Ind,0])), np.std(np.subtract(InputsTargets['PUCorrected'][0,:]  , Targets[Ind,0]))), histtype='step', ec=colors_InOut[3])
        plt.hist(np.subtract(InputsTargets['PU'][0,:], Targets[Ind,0]), bins=nbinsHist, range=[-200, 200], label='PU, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PU'][0,:], Targets[Ind,0])), np.std(np.subtract(InputsTargets['PU'][0,:]  , Targets[Ind,0]))), histtype='step', ec=colors_InOut[4])
        plt.hist(np.subtract(InputsTargets['Puppi'][0,:], Targets[Ind,0]), bins=nbinsHist, range=[-200, 200], label='Puppi, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Puppi'][0,:], Targets[Ind,0])), np.std(np.subtract(InputsTargets['Puppi'][0,:]  , Targets[Ind,0]))), histtype='step', ec=colors_InOut[5])
        #plt.hist(np.subtract(pol2pol2kar_x(Outputs['LongZCorrectedRecoil_Pt'],Outputs['LongZCorrectedRecoil_Phi']), Outputs[Target_Pt]), bins=nbinsHist, range=[-200, 200], label='Puppi, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Puppi'][0,:], Outputs[Target_Pt])), np.std(np.subtract(InputsTargets['Puppi'][0,:]  , Outputs[Target_Pt]))), histtype='step', ec=colors_InOut[5])

        plt.hist(np.subtract(InputsTargets['PF'][0,:], Targets[Ind,0]), bins=nbinsHist, range=[-200, 200], label='PF, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PF'][0,:], Targets[Ind,0])), np.std(np.subtract(InputsTargets['PF'][0,:]  , Targets[Ind,0]))), histtype='step', ec=colors_InOut[1], linewidth=1.5)
        plt.hist(NN_Diff_x, bins=nbinsHist, range=[-200, 200], label='NN, mean=%.2f$\pm$%.2f'%(np.mean(NN_Diff_x), np.std(NN_Diff_x)), histtype='step', ec=colors_InOut[7], linewidth=1.5)


        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        handles, labels = ax.get_legend_handles_labels()
        handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))


        plt.ylabel('Counts')
        plt.xlabel('$\\Delta p_{T,y}$ in GeV')
        plt.xlim(-50,50)

        ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
        plt.grid()
        plt.savefig("%sHist_Delta_x.png"%(plotsD), bbox_inches="tight")
        plt.close()


        pTnorm = np.sqrt( np.multiply(Outputs[Target_Pt],Outputs[Target_Pt])+ np.multiply(Outputs[Target_Phi],Outputs[Target_Phi]))
        IdxpTlow = (Outputs[Target_Pt] <= ptMax_low)
        IdxpTmid = (Outputs[Target_Pt] > ptMin_mid) & (Outputs[Target_Pt] <= ptMax_mid)
        IdxpThigh = (Outputs[Target_Pt] > ptMin_high) & (Outputs[Target_Pt] <= ptMax_high)
        '''
        fig=plt.figure(figsize=(10,6))
        fig.patch.set_facecolor('white')
        ax = plt.subplot(111)
        nbinsHist = 40
        print('IdxpTlow', sum(IdxpTlow))
        print('InputsTargets[Track][0,IdxpTlow]', len(InputsTargets['Track'][0,:]))
        print('Targets[IdxpTlow,0]', len(Targets[:,0]))
        plt.hist(np.subtract(InputsTargets['Track'][0,IdxpTlow], Targets[IdxpTlow,0]), bins=nbinsHist, range=[-50, 50], label='Track, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Track'][0,IdxpTlow], Targets[IdxpTlow,0])), np.std(np.subtract( InputsTargets['Track'][0,IdxpTlow] , Targets[IdxpTlow,0]))), histtype='step', ec=colors_InOut[0])
        plt.hist(np.subtract(InputsTargets['NoPU'][0,IdxpTlow], Targets[IdxpTlow,0]), bins=nbinsHist, range=[-50, 50], label='NoPU, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['NoPU'][0,IdxpTlow], Targets[IdxpTlow,0])), np.std(np.subtract( InputsTargets['NoPU'][0,IdxpTlow] , Targets[IdxpTlow,0]))), histtype='step', ec=colors_InOut[2])
        plt.hist(np.subtract(InputsTargets['PUCorrected'][0,IdxpTlow], Targets[IdxpTlow,0]), bins=nbinsHist, range=[-50, 50], label='PUCorrected, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PUCorrected'][0,IdxpTlow], Targets[IdxpTlow,0])), np.std(np.subtract(InputsTargets['PUCorrected'][0,IdxpTlow]  , Targets[IdxpTlow,0]))), histtype='step', ec=colors_InOut[3])
        plt.hist(np.subtract(InputsTargets['PU'][0,IdxpTlow], Targets[IdxpTlow,0]), bins=nbinsHist, range=[-50, 50], label='PU, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PU'][0,IdxpTlow], Targets[IdxpTlow,0])), np.std(np.subtract(InputsTargets['PU'][0,IdxpTlow]  , Targets[IdxpTlow,0]))), histtype='step', ec=colors_InOut[4])
        plt.hist(np.subtract(InputsTargets['Puppi'][0,IdxpTlow], Targets[IdxpTlow,0]), bins=nbinsHist, range=[-50, 50], label='Puppi, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Puppi'][0,IdxpTlow], Targets[IdxpTlow,0])), np.std(np.subtract(InputsTargets['Puppi'][0,IdxpTlow]  , Targets[IdxpTlow,0]))), histtype='step', ec=colors_InOut[5])
        #plt.hist(np.subtract(pol2pol2kar_x(Outputs['LongZCorrectedRecoil_Pt'],Outputs['LongZCorrectedRecoil_Phi']), Targets[IdxpTlow,0]), bins=nbinsHist, range=[-50, 50], label='Puppi, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Puppi'][0,IdxpTlow], Targets[IdxpTlow,0])), np.std(np.subtract(InputsTargets['Puppi'][0,IdxpTlow]  , Targets[IdxpTlow,0]))), histtype='step', ec=colors_InOut[5])

        plt.hist(np.subtract(InputsTargets['PF'][0,IdxpTlow], Targets[IdxpTlow,0]), bins=nbinsHist, range=[-50, 50], label='PF, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PF'][0,IdxpTlow], Targets[IdxpTlow,0])), np.std(np.subtract(InputsTargets['PF'][0,IdxpTlow]  , Targets[IdxpTlow,0]))), histtype='step', ec=colors_InOut[1], linewidth=1.5)
        plt.hist(NN_Diff_x[IdxpTlow], bins=nbinsHist, range=[-50, 50], label='NN, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(Outputs['NN_Pt'][IdxpTlow], Targets[IdxpTlow,0])), np.std(np.subtract(Outputs['NN_Pt'][IdxpTlow]  , Targets[IdxpTlow,0]))), histtype='step', ec=colors_InOut[7], linewidth=1.5)


        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        handles, labels = ax.get_legend_handles_labels()
        handles.insert(0,mpatches.Patch(color='none', label=pTRangeString_low))


        plt.ylabel('Counts')
        plt.xlabel('$\\Delta p_{T,x}$ in GeV')
        plt.xlim(-50,50)

        ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
        plt.grid()
        plt.savefig("%sHist_Delta_x_low.png"%(plotsD), bbox_inches="tight")
        plt.close()

        fig=plt.figure(figsize=(10,6))
        fig.patch.set_facecolor('white')
        ax = plt.subplot(111)
        nbinsHist = 40
        pTnorm = Outputs[Target_Pt]
        plt.hist(np.subtract(InputsTargets['Track'][0,IdxpTmid], Targets[IdxpTmid,0]), bins=nbinsHist, range=[-50, 50], label='Track, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Track'][0,IdxpTmid], Targets[IdxpTmid,0])), np.std(np.subtract( InputsTargets['Track'][0,IdxpTmid] , Targets[IdxpTmid,0]))), histtype='step', ec=colors_InOut[0])
        plt.hist(np.subtract(InputsTargets['NoPU'][0,IdxpTmid], Targets[IdxpTmid,0]), bins=nbinsHist, range=[-50, 50], label='NoPU, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['NoPU'][0,IdxpTmid], Targets[IdxpTmid,0])), np.std(np.subtract( InputsTargets['NoPU'][0,IdxpTmid] , Targets[IdxpTmid,0]))), histtype='step', ec=colors_InOut[2])
        plt.hist(np.subtract(InputsTargets['PUCorrected'][0,IdxpTmid], Targets[IdxpTmid,0]), bins=nbinsHist, range=[-50, 50], label='PUCorrected, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PUCorrected'][0,IdxpTmid], Targets[IdxpTmid,0])), np.std(np.subtract(InputsTargets['PUCorrected'][0,IdxpTmid]  , Targets[IdxpTmid,0]))), histtype='step', ec=colors_InOut[3])
        plt.hist(np.subtract(InputsTargets['PU'][0,IdxpTmid], Targets[IdxpTmid,0]), bins=nbinsHist, range=[-50, 50], label='PU, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PU'][0,IdxpTmid], Targets[IdxpTmid,0])), np.std(np.subtract(InputsTargets['PU'][0,IdxpTmid]  , Targets[IdxpTmid,0]))), histtype='step', ec=colors_InOut[4])
        plt.hist(np.subtract(InputsTargets['Puppi'][0,IdxpTmid], Targets[IdxpTmid,0]), bins=nbinsHist, range=[-50, 50], label='Puppi, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Puppi'][0,IdxpTmid], Targets[IdxpTmid,0])), np.std(np.subtract(InputsTargets['Puppi'][0,IdxpTmid]  , Targets[IdxpTmid,0]))), histtype='step', ec=colors_InOut[5])
        #plt.hist(np.subtract(pol2pol2kar_x(Outputs['LongZCorrectedRecoil_Pt'],Outputs['LongZCorrectedRecoil_Phi']), Targets[IdxpTmid,0]), bins=nbinsHist, range=[-50, 50], label='Puppi, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Puppi'][0,IdxpTmid], Targets[IdxpTmid,0])), np.std(np.subtract(InputsTargets['Puppi'][0,IdxpTmid]  , Targets[IdxpTmid,0]))), histtype='step', ec=colors_InOut[5])

        plt.hist(np.subtract(InputsTargets['PF'][0,IdxpTmid], Targets[IdxpTmid,0]), bins=nbinsHist, range=[-50, 50], label='PF, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PF'][0,IdxpTmid], Targets[IdxpTmid,0])), np.std(np.subtract(InputsTargets['PF'][0,IdxpTmid]  , Targets[IdxpTmid,0]))), histtype='step', ec=colors_InOut[1], linewidth=1.5)
        plt.hist(NN_Diff_x[IdxpTmid], bins=nbinsHist, range=[-50, 50], label='NN, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(predictions[IdxpTmid,0], Targets[IdxpTmid,0])), np.std(np.subtract(predictions[IdxpTmid,0]  , Targets[IdxpTmid,0]))), histtype='step', ec=colors_InOut[7], linewidth=1.5)


        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        handles, labels = ax.get_legend_handles_labels()
        handles.insert(0,mpatches.Patch(color='none', label=pTRangeString_mid))


        plt.ylabel('Counts')
        plt.xlabel('$\\Delta p_{T,x}$ in GeV')
        plt.xlim(-50,50)

        ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
        plt.grid()
        plt.savefig("%sHist_Delta_x_mid.png"%(plotsD), bbox_inches="tight")
        plt.close()
        '''


        fig=plt.figure(figsize=(10,6))
        fig.patch.set_facecolor('white')
        ax = plt.subplot(111)
        nbinsHist = 40
        pTnorm = Outputs[Target_Pt]
        plt.hist(np.subtract(InputsTargets['Track'][0,IdxpThigh], Targets[IdxpThigh,0]), bins=nbinsHist, range=[-50, 50], label='Track, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Track'][0,IdxpThigh], Targets[IdxpThigh,0])), np.std(np.subtract( InputsTargets['Track'][0,IdxpThigh] , Targets[IdxpThigh,0]))), histtype='step', ec=colors_InOut[0])
        plt.hist(np.subtract(InputsTargets['NoPU'][0,IdxpThigh], Targets[IdxpThigh,0]), bins=nbinsHist, range=[-50, 50], label='NoPU, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['NoPU'][0,IdxpThigh], Targets[IdxpThigh,0])), np.std(np.subtract( InputsTargets['NoPU'][0,IdxpThigh] , Targets[IdxpThigh,0]))), histtype='step', ec=colors_InOut[2])
        plt.hist(np.subtract(InputsTargets['PUCorrected'][0,IdxpThigh], Targets[IdxpThigh,0]), bins=nbinsHist, range=[-50, 50], label='PUCorrected, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PUCorrected'][0,IdxpThigh], Targets[IdxpThigh,0])), np.std(np.subtract(InputsTargets['PUCorrected'][0,IdxpThigh]  , Targets[IdxpThigh,0]))), histtype='step', ec=colors_InOut[3])
        plt.hist(np.subtract(InputsTargets['PU'][0,IdxpThigh], Targets[IdxpThigh,0]), bins=nbinsHist, range=[-50, 50], label='PU, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PU'][0,IdxpThigh], Targets[IdxpThigh,0])), np.std(np.subtract(InputsTargets['PU'][0,IdxpThigh]  , Targets[IdxpThigh,0]))), histtype='step', ec=colors_InOut[4])
        plt.hist(np.subtract(InputsTargets['Puppi'][0,IdxpThigh], Targets[IdxpThigh,0]), bins=nbinsHist, range=[-50, 50], label='Puppi, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Puppi'][0,IdxpThigh], Targets[IdxpThigh,0])), np.std(np.subtract(InputsTargets['Puppi'][0,IdxpThigh]  , Targets[IdxpThigh,0]))), histtype='step', ec=colors_InOut[5])
        #plt.hist(np.subtract(pol2pol2kar_x(Outputs['LongZCorrectedRecoil_Pt'],Outputs['LongZCorrectedRecoil_Phi']), Targets[IdxpThigh,0]), bins=nbinsHist, range=[-50, 50], label='Puppi, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Puppi'][0,IdxpThigh], Targets[IdxpThigh,0])), np.std(np.subtract(InputsTargets['Puppi'][0,IdxpThigh]  , Targets[IdxpThigh,0]))), histtype='step', ec=colors_InOut[5])

        plt.hist(np.subtract(InputsTargets['PF'][0,IdxpThigh], Targets[IdxpThigh,0]), bins=nbinsHist, range=[-50, 50], label='PF, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PF'][0,IdxpThigh], Targets[IdxpThigh,0])), np.std(np.subtract(InputsTargets['PF'][0,IdxpThigh]  , Targets[IdxpThigh,0]))), histtype='step', ec=colors_InOut[1], linewidth=1.5)
        plt.hist(NN_Diff_x[IdxpThigh], bins=nbinsHist, range=[-50, 50], label='NN, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(predictions[IdxpThigh,0], Targets[IdxpThigh,0])), np.std(np.subtract(predictions[IdxpThigh,0]  , Targets[IdxpThigh,0]))), histtype='step', ec=colors_InOut[7], linewidth=1.5)


        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        handles, labels = ax.get_legend_handles_labels()
        handles.insert(0,mpatches.Patch(color='none', label=pTRangeString_high))


        plt.ylabel('Counts')
        plt.xlabel('$\\Delta p_{T,x}$ in GeV')
        plt.xlim(-50,50)

        ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
        plt.grid()
        plt.savefig("%sHist_Delta_x_high.png"%(plotsD), bbox_inches="tight")
        plt.close()



        '''
        fig=plt.figure(figsize=(10,6))
        fig.patch.set_facecolor('white')
        ax = plt.subplot(111)
        nbinsHist = 40

        plt.hist(np.subtract(InputsTargets['Track'][1,IdxpTlow], Targets[IdxpTlow,1]), bins=nbinsHist, range=[-50, 50], label='Track, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Track'][1,IdxpTlow], Targets[IdxpTlow,1])), np.std(np.subtract( InputsTargets['Track'][1,IdxpTlow] , Targets[IdxpTlow,1]))), histtype='step', ec=colors_InOut[0])
        plt.hist(np.subtract(InputsTargets['NoPU'][1,IdxpTlow], Targets[IdxpTlow,1]), bins=nbinsHist, range=[-50, 50], label='NoPU, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['NoPU'][1,IdxpTlow], Targets[IdxpTlow,1])), np.std(np.subtract( InputsTargets['NoPU'][1,IdxpTlow] , Targets[IdxpTlow,1]))), histtype='step', ec=colors_InOut[2])
        plt.hist(np.subtract(InputsTargets['PUCorrected'][1,IdxpTlow], Targets[IdxpTlow,1]), bins=nbinsHist, range=[-50, 50], label='PUCorrected, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PUCorrected'][1,IdxpTlow], Targets[IdxpTlow,1])), np.std(np.subtract(InputsTargets['PUCorrected'][1,IdxpTlow]  , Targets[IdxpTlow,1]))), histtype='step', ec=colors_InOut[3])
        plt.hist(np.subtract(InputsTargets['PU'][1,IdxpTlow], Targets[IdxpTlow,1]), bins=nbinsHist, range=[-50, 50], label='PU, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PU'][1,IdxpTlow], Targets[IdxpTlow,1])), np.std(np.subtract(InputsTargets['PU'][1,IdxpTlow]  , Targets[IdxpTlow,1]))), histtype='step', ec=colors_InOut[4])
        plt.hist(np.subtract(InputsTargets['Puppi'][1,IdxpTlow], Targets[IdxpTlow,1]), bins=nbinsHist, range=[-50, 50], label='Puppi, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Puppi'][1,IdxpTlow], Targets[IdxpTlow,1])), np.std(np.subtract(InputsTargets['Puppi'][1,IdxpTlow]  , Targets[IdxpTlow,1]))), histtype='step', ec=colors_InOut[5])
        #plt.hist(np.subtract(pol2pol2kar_x(Outputs['LongZCorrectedRecoil_Pt'],Outputs['LongZCorrectedRecoil_Phi']), Targets[IdxpTlow,1]), bins=nbinsHist, range=[-50, 50], label='Puppi, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Puppi'][1,IdxpTlow], Targets[IdxpTlow,1])), np.std(np.subtract(InputsTargets['Puppi'][1,IdxpTlow]  , Targets[IdxpTlow,1]))), histtype='step', ec=colors_InOut[5])

        plt.hist(np.subtract(InputsTargets['PF'][1,IdxpTlow], Targets[IdxpTlow,1]), bins=nbinsHist, range=[-50, 50], label='PF, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PF'][1,IdxpTlow], Targets[IdxpTlow,1])), np.std(np.subtract(InputsTargets['PF'][1,IdxpTlow]  , Targets[IdxpTlow,1]))), histtype='step', ec=colors_InOut[1], linewidth=1.5)
        plt.hist(NN_Diff_y[IdxpTlow], bins=nbinsHist, range=[-50, 50], label='NN, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(predictions[IdxpTlow,1], Targets[IdxpTlow,1])), np.std(np.subtract(predictions[IdxpTlow,1]  , Targets[IdxpTlow,1]))), histtype='step', ec=colors_InOut[7], linewidth=1.5)


        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        handles, labels = ax.get_legend_handles_labels()
        handles.insert(0,mpatches.Patch(color='none', label=pTRangeString_low))

        plt.ylabel('Counts')
        plt.xlabel('$\Delta p_{T,y}$ in GeV')
        plt.xlim(-50,50)
        #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_{T}^Z} \\right) $ in GeV')
        #plt.title(' Histogram Deviation to Target  $\Delta p_{T,y}$')
        #plt.text('$p_T$ range restriction')

        ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
        plt.grid()
        #plt.ylim(ylimResMVAMin, ylimResMax)
        plt.savefig("%sHist_Delta_y_low.png"%(plotsD), bbox_inches="tight")
        plt.close()


        fig=plt.figure(figsize=(10,6))
        fig.patch.set_facecolor('white')
        ax = plt.subplot(111)
        nbinsHist = 40

        plt.hist(np.subtract(InputsTargets['Track'][1,IdxpTmid], Targets[IdxpTmid,1]), bins=nbinsHist, range=[-50, 50], label='Track, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Track'][1,IdxpTmid], Targets[IdxpTmid,1])), np.std(np.subtract( InputsTargets['Track'][1,IdxpTmid] , Targets[IdxpTmid,1]))), histtype='step', ec=colors_InOut[0])
        plt.hist(np.subtract(InputsTargets['NoPU'][1,IdxpTmid], Targets[IdxpTmid,1]), bins=nbinsHist, range=[-50, 50], label='NoPU, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['NoPU'][1,IdxpTmid], Targets[IdxpTmid,1])), np.std(np.subtract( InputsTargets['NoPU'][1,IdxpTmid] , Targets[IdxpTmid,1]))), histtype='step', ec=colors_InOut[2])
        plt.hist(np.subtract(InputsTargets['PUCorrected'][1,IdxpTmid], Targets[IdxpTmid,1]), bins=nbinsHist, range=[-50, 50], label='PUCorrected, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PUCorrected'][1,IdxpTmid], Targets[IdxpTmid,1])), np.std(np.subtract(InputsTargets['PUCorrected'][1,IdxpTmid]  , Targets[IdxpTmid,1]))), histtype='step', ec=colors_InOut[3])
        plt.hist(np.subtract(InputsTargets['PU'][1,IdxpTmid], Targets[IdxpTmid,1]), bins=nbinsHist, range=[-50, 50], label='PU, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PU'][1,IdxpTmid], Targets[IdxpTmid,1])), np.std(np.subtract(InputsTargets['PU'][1,IdxpTmid]  , Targets[IdxpTmid,1]))), histtype='step', ec=colors_InOut[4])
        plt.hist(np.subtract(InputsTargets['Puppi'][1,IdxpTmid], Targets[IdxpTmid,1]), bins=nbinsHist, range=[-50, 50], label='Puppi, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Puppi'][1,IdxpTmid], Targets[IdxpTmid,1])), np.std(np.subtract(InputsTargets['Puppi'][1,IdxpTmid]  , Targets[IdxpTmid,1]))), histtype='step', ec=colors_InOut[5])
        #plt.hist(np.subtract(pol2pol2kar_x(Outputs['LongZCorrectedRecoil_Pt'],Outputs['LongZCorrectedRecoil_Phi']), Targets[IdxpTmid,1]), bins=nbinsHist, range=[-50, 50], label='Puppi, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Puppi'][1,IdxpTmid], Targets[IdxpTmid,1])), np.std(np.subtract(InputsTargets['Puppi'][1,IdxpTmid]  , Targets[IdxpTmid,1]))), histtype='step', ec=colors_InOut[5])

        plt.hist(np.subtract(InputsTargets['PF'][1,IdxpTmid], Targets[IdxpTmid,1]), bins=nbinsHist, range=[-50, 50], label='PF, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PF'][1,IdxpTmid], Targets[IdxpTmid,1])), np.std(np.subtract(InputsTargets['PF'][1,IdxpTmid]  , Targets[IdxpTmid,1]))), histtype='step', ec=colors_InOut[1], linewidth=1.5)
        plt.hist(NN_Diff_y[IdxpTmid], bins=nbinsHist, range=[-50, 50], label='NN, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(predictions[IdxpTmid,1], Targets[IdxpTmid,1])), np.std(np.subtract(predictions[IdxpTmid,1]  , Targets[IdxpTmid,1]))), histtype='step', ec=colors_InOut[7], linewidth=1.5)


        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        handles, labels = ax.get_legend_handles_labels()
        handles.insert(0,mpatches.Patch(color='none', label=pTRangeString_mid))

        plt.ylabel('Counts')
        plt.xlabel('$\Delta p_{T,y}$ in GeV')
        plt.xlim(-50,50)
        #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_{T}^Z} \\right) $ in GeV')
        #plt.title(' Histogram Deviation to Target  $\Delta p_{T,y}$')
        #plt.text('$p_T$ range restriction')

        ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
        plt.grid()
        #plt.ylim(ylimResMVAMin, ylimResMax)
        plt.savefig("%sHist_Delta_y_mid.png"%(plotsD), bbox_inches="tight")
        plt.close()
        '''

        fig=plt.figure(figsize=(10,6))
        fig.patch.set_facecolor('white')
        ax = plt.subplot(111)
        nbinsHist = 40

        plt.hist(np.subtract(InputsTargets['Track'][1,IdxpThigh], Targets[IdxpThigh,1]), bins=nbinsHist, range=[-50, 50], label='Track, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Track'][1,IdxpThigh], Targets[IdxpThigh,1])), np.std(np.subtract( InputsTargets['Track'][1,IdxpThigh] , Targets[IdxpThigh,1]))), histtype='step', ec=colors_InOut[0])
        plt.hist(np.subtract(InputsTargets['NoPU'][1,IdxpThigh], Targets[IdxpThigh,1]), bins=nbinsHist, range=[-50, 50], label='NoPU, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['NoPU'][1,IdxpThigh], Targets[IdxpThigh,1])), np.std(np.subtract( InputsTargets['NoPU'][1,IdxpThigh] , Targets[IdxpThigh,1]))), histtype='step', ec=colors_InOut[2])
        plt.hist(np.subtract(InputsTargets['PUCorrected'][1,IdxpThigh], Targets[IdxpThigh,1]), bins=nbinsHist, range=[-50, 50], label='PUCorrected, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PUCorrected'][1,IdxpThigh], Targets[IdxpThigh,1])), np.std(np.subtract(InputsTargets['PUCorrected'][1,IdxpThigh]  , Targets[IdxpThigh,1]))), histtype='step', ec=colors_InOut[3])
        plt.hist(np.subtract(InputsTargets['PU'][1,IdxpThigh], Targets[IdxpThigh,1]), bins=nbinsHist, range=[-50, 50], label='PU, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PU'][1,IdxpThigh], Targets[IdxpThigh,1])), np.std(np.subtract(InputsTargets['PU'][1,IdxpThigh]  , Targets[IdxpThigh,1]))), histtype='step', ec=colors_InOut[4])
        plt.hist(np.subtract(InputsTargets['Puppi'][1,IdxpThigh], Targets[IdxpThigh,1]), bins=nbinsHist, range=[-50, 50], label='Puppi, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Puppi'][1,IdxpThigh], Targets[IdxpThigh,1])), np.std(np.subtract(InputsTargets['Puppi'][1,IdxpThigh]  , Targets[IdxpThigh,1]))), histtype='step', ec=colors_InOut[5])
        #plt.hist(np.subtract(pol2pol2kar_x(Outputs['LongZCorrectedRecoil_Pt'],Outputs['LongZCorrectedRecoil_Phi']), Targets[IdxpThigh,1]), bins=nbinsHist, range=[-50, 50], label='Puppi, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Puppi'][1,IdxpThigh], Targets[IdxpThigh,1])), np.std(np.subtract(InputsTargets['Puppi'][1,IdxpThigh]  , Targets[IdxpThigh,1]))), histtype='step', ec=colors_InOut[5])

        plt.hist(np.subtract(InputsTargets['PF'][1,IdxpThigh], Targets[IdxpThigh,1]), bins=nbinsHist, range=[-50, 50], label='PF, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PF'][1,IdxpThigh], Targets[IdxpThigh,1])), np.std(np.subtract(InputsTargets['PF'][1,IdxpThigh]  , Targets[IdxpThigh,1]))), histtype='step', ec=colors_InOut[1], linewidth=1.5)
        plt.hist(NN_Diff_y[IdxpThigh], bins=nbinsHist, range=[-50, 50], label='NN, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(predictions[IdxpThigh,1], Targets[IdxpThigh,1])), np.std(np.subtract(predictions[IdxpThigh,1]  , Targets[IdxpThigh,1]))), histtype='step', ec=colors_InOut[7], linewidth=1.5)


        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        handles, labels = ax.get_legend_handles_labels()
        handles.insert(0,mpatches.Patch(color='none', label=pTRangeString_high))

        plt.ylabel('Counts')
        plt.xlabel('$\Delta p_{T,y}$ in GeV')
        plt.xlim(-50,50)
        #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_{T}^Z} \\right) $ in GeV')
        #plt.title(' Histogram Deviation to Target  $\Delta p_{T,y}$')
        #plt.text('$p_T$ range restriction')

        ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
        plt.grid()
        #plt.ylim(ylimResMVAMin, ylimResMax)
        plt.savefig("%sHist_Delta_y_high.png"%(plotsD), bbox_inches="tight")
        plt.close()

        fig=plt.figure(figsize=(10,6))
        fig.patch.set_facecolor('white')
        ax = plt.subplot(111)
        nbinsHist = 40

        plt.hist(np.subtract(InputsTargets['Track'][1,Ind], Outputs[Target_Pt][Ind]), bins=nbinsHist, range=[-50, 50], label='Track, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Track'][1,Ind], Outputs[Target_Pt][Ind])), np.std(np.subtract( InputsTargets['Track'][1,Ind] , Outputs[Target_Pt][Ind]))), histtype='step', ec=colors_InOut[0])
        plt.hist(np.subtract(InputsTargets['NoPU'][1,Ind], Outputs[Target_Pt][Ind]), bins=nbinsHist, range=[-50, 50], label='NoPU, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['NoPU'][1,Ind], Outputs[Target_Pt][Ind])), np.std(np.subtract( InputsTargets['NoPU'][1,Ind] , Outputs[Target_Pt][Ind]))), histtype='step', ec=colors_InOut[2])
        plt.hist(np.subtract(InputsTargets['PUCorrected'][1,Ind], Outputs[Target_Pt][Ind]), bins=nbinsHist, range=[-50, 50], label='PUCorrected, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PUCorrected'][1,Ind], Outputs[Target_Pt][Ind])), np.std(np.subtract(InputsTargets['PUCorrected'][1,Ind]  , Outputs[Target_Pt][Ind]))), histtype='step', ec=colors_InOut[3])
        plt.hist(np.subtract(InputsTargets['PU'][1,Ind], Outputs[Target_Pt][Ind]), bins=nbinsHist, range=[-50, 50], label='PU, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PU'][1,Ind], Outputs[Target_Pt][Ind])), np.std(np.subtract(InputsTargets['PU'][1,Ind]  , Outputs[Target_Pt][Ind]))), histtype='step', ec=colors_InOut[4])
        plt.hist(np.subtract(InputsTargets['Puppi'][1,Ind], Outputs[Target_Pt][Ind]), bins=nbinsHist, range=[-50, 50], label='Puppi, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Puppi'][1,Ind], Outputs[Target_Pt][Ind])), np.std(np.subtract(InputsTargets['Puppi'][1,Ind]  , Outputs[Target_Pt][Ind]))), histtype='step', ec=colors_InOut[5])
        #plt.hist(np.subtract(pol2pol2kar_x(Outputs['LongZCorrectedRecoil_Pt'],Outputs['LongZCorrectedRecoil_Pt']), Outputs[Target_Pt][Ind]), bins=nbinsHist, range=[-50, 50], label='Puppi, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Puppi'][1,Ind], Outputs[Target_Pt][Ind])), np.std(np.subtract(InputsTargets['Puppi'][1,Ind]  , Outputs[Target_Pt][Ind]))), histtype='step', ec=colors_InOut[5])

        plt.hist(np.subtract(InputsTargets['PF'][1,Ind], Outputs[Target_Pt][Ind]), bins=nbinsHist, range=[-50, 50], label='PF, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PF'][1,Ind], Outputs[Target_Pt][Ind])), np.std(np.subtract(InputsTargets['PF'][1,Ind]  , Outputs[Target_Pt][Ind]))), histtype='step', ec=colors_InOut[1], linewidth=1.5)
        plt.hist(NN_Diff_y[Ind], bins=nbinsHist, range=[-50, 50], label='NN, mean=%.2f$\pm$%.2f'%(np.mean(NN_Diff_y[Ind]), np.std(NN_Diff_y[Ind])), histtype='step', ec=colors_InOut[7], linewidth=1.5)


        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        handles, labels = ax.get_legend_handles_labels()
        handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

        plt.ylabel('Counts')
        plt.xlabel('$\Delta p_{T,y}$ in GeV')
        plt.xlim(-50,50)
        #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_{T}^Z} \\right) $ in GeV')
        #plt.title(' Histogram Deviation to Target  $\Delta p_{T,y}$')
        #plt.text('$p_T$ range restriction')

        ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
        plt.grid()
        #plt.ylim(ylimResMVAMin, ylimResMax)
        plt.savefig("%sHist_Delta_y2.png"%(plotsD), bbox_inches="tight")
        plt.close()
        '''


        fig=plt.figure(figsize=(10,6))
        fig.patch.set_facecolor('white')
        ax = plt.subplot(111)
        nbinsHist = 40

        plt.hist(np.subtract(InputsTargets['Track'][0,:], Outputs[Target_Pt]), bins=nbinsHist, range=[-50, 50], label='Track, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Track'][0,:], Outputs[Target_Pt])), np.std(np.subtract( InputsTargets['Track'][0,:] , Outputs[Target_Pt]))), histtype='step', ec=colors_InOut[0], normed=True)
        plt.hist(np.subtract(InputsTargets['NoPU'][0,:], Outputs[Target_Pt]), bins=nbinsHist, range=[-50, 50], label='NoPU, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['NoPU'][0,:], Outputs[Target_Pt])), np.std(np.subtract( InputsTargets['NoPU'][0,:] , Outputs[Target_Pt]))), histtype='step', ec=colors_InOut[2], normed=True)
        plt.hist(np.subtract(InputsTargets['PUCorrected'][0,:], Outputs[Target_Pt]), bins=nbinsHist, range=[-50, 50], label='PUCorrected, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PUCorrected'][0,:], Outputs[Target_Pt])), np.std(np.subtract(InputsTargets['PUCorrected'][0,:]  , Outputs[Target_Pt]))), histtype='step', ec=colors_InOut[3], normed=True)
        plt.hist(np.subtract(InputsTargets['PU'][0,:], Outputs[Target_Pt]), bins=nbinsHist, range=[-50, 50], label='PU, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PU'][0,:], Outputs[Target_Pt])), np.std(np.subtract(InputsTargets['PU'][0,:]  , Outputs[Target_Pt]))), histtype='step', ec=colors_InOut[4], normed=True)
        plt.hist(np.subtract(InputsTargets['Puppi'][0,:], Outputs[Target_Pt]), bins=nbinsHist, range=[-50, 50], label='Puppi, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Puppi'][0,:], Outputs[Target_Pt])), np.std(np.subtract(InputsTargets['Puppi'][0,:]  , Outputs[Target_Pt]))), histtype='step', ec=colors_InOut[5], normed=True)
        #plt.hist(np.subtract(pol2pol2kar_x(Outputs['LongZCorrectedRecoil_Pt'],Outputs['LongZCorrectedRecoil_Phi']), Outputs[Target_Pt]), bins=nbinsHist, range=[-50, 50], label='Puppi, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Puppi'][0,:], Outputs[Target_Pt])), np.std(np.subtract(InputsTargets['Puppi'][0,:]  , Outputs[Target_Pt]))), histtype='step', ec=colors_InOut[5])

        plt.hist(np.subtract(InputsTargets['PF'][0,:], Outputs[Target_Pt]), bins=nbinsHist, range=[-50, 50], label='PF, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PF'][0,:], Outputs[Target_Pt])), np.std(np.subtract(InputsTargets['PF'][0,:]  , Outputs[Target_Pt]))), histtype='step', ec=colors_InOut[1], linewidth=1.5, normed=True)
        plt.hist(NN_Diff_x, bins=nbinsHist, range=[-50, 50], label='NN, mean=%.2f$\pm$%.2f'%(np.mean(NN_Diff_x), np.std(NN_Diff_x)), histtype='step', ec=colors_InOut[7], linewidth=1.5, normed=True)

        x = np.linspace(-50,50, nbinsHist)
        plt.plot(x, mlab.normpdf(x, np.mean(NN_Diff_x), np.std(NN_Diff_x)), 'k--',  label='Gaussian Fit, mean=%.2f$\pm$%.2f'%(np.mean(NN_Diff_x), np.std(NN_Diff_x)))

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        handles, labels = ax.get_legend_handles_labels()
        #handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

        plt.ylabel('probability')
        plt.xlabel('$\\Delta p_{T,x}$ in GeV')
        plt.xlim(-50,50)
        #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_{T}^Z} \\right) $ in GeV')
        #plt.title(' Histogram Deviation to Target  $\\Delta p_{T,x}$')
        #plt.text('$p_T$ range restriction')

        ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
        plt.grid()
        #plt.ylim(ylimResMVAMin, ylimResMax)
        plt.savefig("%sHist_Delta_x_Gaussfit.png"%(plotsD), bbox_inches="tight")
        plt.close()




        fig=plt.figure(figsize=(10,6))
        fig.patch.set_facecolor('white')
        ax = plt.subplot(111)
        nbinsHist = 40

        plt.hist(np.subtract(InputsTargets['Track'][1,:], Outputs[Target_Phi]), bins=nbinsHist, range=[-50, 50], label='Track, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Track'][1,:], Outputs[Target_Phi])), np.std(np.subtract( InputsTargets['Track'][1,:] , Outputs[Target_Phi]))), histtype='step', ec=colors_InOut[0], normed=True)
        plt.hist(np.subtract(InputsTargets['NoPU'][1,:], Outputs[Target_Phi]), bins=nbinsHist, range=[-50, 50], label='NoPU, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['NoPU'][1,:], Outputs[Target_Phi])), np.std(np.subtract( InputsTargets['NoPU'][1,:] , Outputs[Target_Phi]))), histtype='step', ec=colors_InOut[2], normed=True)
        plt.hist(np.subtract(InputsTargets['PUCorrected'][1,:], Outputs[Target_Phi]), bins=nbinsHist, range=[-50, 50], label='PUCorrected, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PUCorrected'][1,:], Outputs[Target_Phi])), np.std(np.subtract(InputsTargets['PUCorrected'][1,:]  , Outputs[Target_Phi]))), histtype='step', ec=colors_InOut[3], normed=True)
        plt.hist(np.subtract(InputsTargets['PU'][1,:], Outputs[Target_Phi]), bins=nbinsHist, range=[-50, 50], label='PU, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PU'][1,:], Outputs[Target_Phi])), np.std(np.subtract(InputsTargets['PU'][1,:]  , Outputs[Target_Phi]))), histtype='step', ec=colors_InOut[4], normed=True)
        plt.hist(np.subtract(InputsTargets['Puppi'][1,:], Outputs[Target_Phi]), bins=nbinsHist, range=[-50, 50], label='Puppi, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Puppi'][1,:], Outputs[Target_Phi])), np.std(np.subtract(InputsTargets['Puppi'][1,:]  , Outputs[Target_Phi]))), histtype='step', ec=colors_InOut[5], normed=True)
        #plt.hist(np.subtract(pol2pol2kar_x(Outputs['LongZCorrectedRecoil_Pt'],Outputs['LongZCorrectedRecoil_Phi']), Outputs[Target_Phi]), bins=nbinsHist, range=[-50, 50], label='Puppi, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Puppi'][1,:], Outputs[Target_Phi])), np.std(np.subtract(InputsTargets['Puppi'][1,:]  , Outputs[Target_Phi]))), histtype='step', ec=colors_InOut[5])

        plt.hist(np.subtract(InputsTargets['PF'][1,:], Outputs[Target_Phi]), bins=nbinsHist, range=[-50, 50], label='PF, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PF'][1,:], Outputs[Target_Phi])), np.std(np.subtract(InputsTargets['PF'][1,:]  , Outputs[Target_Phi]))), histtype='step', ec=colors_InOut[1], linewidth=1.5, normed=True)
        plt.hist(NN_Diff_y, bins=nbinsHist, range=[-50, 50], label='NN, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(Outputs['NN_Phi'], Outputs[Target_Phi])), np.std(np.subtract(Outputs['NN_Phi']  , Outputs[Target_Phi]))), histtype='step', ec=colors_InOut[7], linewidth=1.5, normed=True)

        x = np.linspace(-50,50, nbinsHist)
        plt.plot(x, mlab.normpdf(x, np.mean(NN_Diff_y), np.std(NN_Diff_y)), 'k--', label='Gaussian, mean=%.2f$\pm$%.2f'%(np.mean(NN_Diff_y), np.std(NN_Diff_y)))

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        handles, labels = ax.get_legend_handles_labels()
        #handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

        plt.ylabel('probability')
        plt.xlabel('$\Delta p_{T,y}$ in GeV')
        plt.xlim(-50,50)
        #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_{T}^Z} \\right) $ in GeV')
        #plt.title(' Histogram Deviation to Target  $\Delta p_{T,y}$')
        #plt.text('$p_T$ range restriction')

        ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
        plt.grid()
        #plt.ylim(ylimResMVAMin, ylimResMax)
        plt.savefig("%sHist_Delta_y_Gaussfit.png"%(plotsD), bbox_inches="tight")
        plt.close()
        '''

    plt.clf()
    plt.figure()
    #plt.suptitle('x-Korrektur PF-Target ')
    plt.xlabel("$ p_{T,x}^Z$")
    plt.ylabel("PF-Target")
    heatmap, xedges, yedges = np.histogram2d(   Outputs[Target_Pt], np.subtract(InputsTargets['PF'][0,:],  Outputs[Target_Pt]), bins=50,
                                             range=[[np.percentile(Outputs[Target_Pt],5),np.percentile(Outputs[Target_Pt],95)],
                                                    [np.percentile(np.subtract(InputsTargets['PF'][0,:],  Outputs[Target_Pt]),5),
                                                    np.percentile(np.subtract(InputsTargets['PF'][0,:],  Outputs[Target_Pt]),95)]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #plt.clf()
    HM =plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm())
    plt.colorbar(HM)

    plt.legend()
    plt.savefig("%sHM_PF-Tar_Tar_x.png"%(plotsD), bbox_inches="tight")
    plt.close()



    plt.clf()
    plt.figure()
    #plt.suptitle('y-Korrektur: PF-Target ')
    plt.xlabel("$ p_{T,y}^Z$")
    plt.ylabel("PF-Target")
    heatmap, xedges, yedges = np.histogram2d(  Outputs[Target_Phi], np.subtract(InputsTargets['PF'][1,:],  Outputs[Target_Phi]), bins=50,
                                             range=[[np.percentile(Outputs[Target_Phi],5),np.percentile(Outputs[Target_Phi],95)],
                                                    [np.percentile(map(np.subtract, InputsTargets['PF'][1,:],  Outputs[Target_Phi]),5),
                                                     np.percentile(map(np.subtract, InputsTargets['PF'][1,:],  Outputs[Target_Phi]),95)]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    HM= plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm())
    plt.colorbar(HM)
    plt.legend()
    plt.savefig("%sHM_PF-Tar_Tar_y.png"%(plotsD), bbox_inches="tight")

    '''
    plt.clf()
    plt.figure()
    #plt.suptitle('x-Korrektur Prediction-Target ')
    plt.xlabel("$ p_{T,x}^Z$")
    plt.ylabel("Prediction-Target")
    heatmap, xedges, yedges = np.histogram2d(  Outputs[Target_Pt], NN_Diff_x,  bins=50,
                                             range=[[-20,20],
                                                    [-20,20]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #plt.clf()
    HM =plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm())
    plt.colorbar(HM)

    plt.legend()
    plt.savefig("%sHM_Delta_x_Tar_x.png"%(plotsD), bbox_inches="tight")
    plt.close()
    '''





    plt.clf()
    plt.figure()
    #plt.suptitle('x-Korrektur Prediction-Target ')
    plt.xlabel("$ \Delta p_{T,x}^Z$")
    plt.ylabel("$ \Delta p_{T,y}^Z$")
    heatmap, xedges, yedges = np.histogram2d(  NN_Diff_x, NN_Diff_y,  bins=50,
                                             range=[[-20,20],
                                                    [-20,20]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #plt.clf()
    HM =plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm())
    plt.colorbar(HM)

    plt.legend()
    plt.savefig("%sHM_Delta_x_Delta_y.png"%(plotsD), bbox_inches="tight")


    plt.clf()
    plt.figure()
    #plt.suptitle('y-Korrektur: Prediction-Target ')
    plt.xlabel("$ p_{T,y}^Z$")
    plt.ylabel("Prediction-Target")
    heatmap, xedges, yedges = np.histogram2d( Outputs[Target_Phi], NN_Diff_y, bins=50,
                                             range=[[-20,20],[-20,20]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    HM= plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm())
    plt.colorbar(HM)
    plt.legend()
    plt.savefig("%sHM_Delta_y_Tar_y.png"%(plotsD), bbox_inches="tight")
    plt.close()


    plt.clf()
    plt.figure()
    #plt.suptitle('Targets: x-y-correlation ')
    plt.ylabel("$p_{T,y}^Z$")
    plt.xlabel("$p_{T,x}^Z$")
    heatmap, xedges, yedges = np.histogram2d(  Outputs[Target_Phi] , Outputs[Target_Pt], bins=50, range=[[-40,40],[-40, 40]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    HM = plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm() )
    plt.colorbar(HM)
    plt.legend()
    plt.savefig("%sHM_Targets_Correlation_Kar.png"%(plotsD), bbox_inches="tight")
    plt.close()


    PF_x, PF_y = pol2kar(Outputs['recoilslimmedMETs_Pt'], Outputs['recoilslimmedMETs_Phi'])
    plt.clf()
    plt.figure()
    #plt.suptitle('Targets: x-y-correlation ')
    plt.ylabel("$p_{T,y}$")
    plt.xlabel("$p_{T,x}$")
    heatmap, xedges, yedges = np.histogram2d(  PF_y , PF_x, bins=50, range=[[-40,40],[-40, 40]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    HM = plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm() )
    plt.colorbar(HM)
    plt.legend()
    plt.savefig("%sHM_PF_Correlation_Kar.png"%(plotsD), bbox_inches="tight")
    plt.close()


    plt.clf()
    plt.figure()
    #plt.suptitle('Targets: x-y-correlation ')
    plt.xlabel("$\\phi_Z$")
    plt.ylabel("$p_T^Z$")
    heatmap, xedges, yedges = np.histogram2d(  Outputs[Target_Phi] , Outputs[Target_Pt], bins=[20,100], range=[[-np.pi, np.pi],[0,30]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    HM = plt.imshow(heatmap.T, extent=extent, origin='lower' , norm=LogNorm())
    plt.colorbar(HM)
    plt.legend()
    plt.savefig("%sHM_Targets_Correlation_Pol.png"%(plotsD), bbox_inches="tight")
    plt.close()


    plt.clf()
    plt.figure()
    #plt.suptitle('predictions: x-y-correlation ')
    plt.ylabel("$p_{T,y}^Z$")
    plt.xlabel("$p_{T,x}^Z$")
    heatmap, xedges, yedges = np.histogram2d(  Outputs['NN_Phi'] , Outputs['NN_Pt'], bins=50,  range=[[-25,25],[-25, 25]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    HM= plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm())
    plt.colorbar(HM)
    plt.legend()
    plt.savefig("%sHM_predictions_Correlation.png"%(plotsD), bbox_inches="tight")
    plt.close()



    #print('Summe prediction 0', np.sum(Outputs['NN_Pt']))
    #print('Summe prediction 1', np.sum(Outputs['NN_Phi']))

    print("y-Bias: np.mean(Outputs['NN_Phi'])-np.mean(Outputs[Target_Phi])", np.mean(Outputs['NN_Phi'])-np.mean(Outputs[Target_Phi]))

    if NN_mode =='xyr' or NN_mode =='xyd' or NN_mode =='nr' or NN_mode =='xyd':
        plt.clf()
        plt.figure()
        #plt.suptitle('y-Korrektur: Prediction-Target ')
        plt.xlabel("$|-\\vec{p}_T^Z|$ in GeV")
        plt.ylabel("Prediction-Target")
        heatmap, xedges, yedges = np.histogram2d(  map(np.subtract, predictions[:,2],  Targets[:,2]) , Targets[:,2], bins=50)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        #plt.clf()
        plt.imshow(heatmap.T, extent=extent, origin='lower')
        #plt.colorbar(heatmap.T)
        plt.legend()
        plt.savefig("%sHM_Pred-Tar_Tar_r.png"%(plotsD), bbox_inches="tight")

    print('Mean x deviation', np.mean(np.subtract(predictions[:,0], Targets[:,0])))
    print('Std x deviation', np.std(np.subtract(predictions[:,0], Targets[:,0])))
    print('Mean y deviation', np.mean(np.subtract(predictions[:,1], Targets[:,1])))
    print('Std y deviation', np.std(np.subtract(predictions[:,1], Targets[:,1])))


    epochs = range(1, 1+len(loss))
    plt.figure()
    plt.xlabel("Epochs all")
    plt.ylabel("Loss")
    plt.plot(epochs, loss, 'g^', label="Loss")
    plt.plot(epochs, val_loss, 'r^', label="Validation Loss")
    plt.legend(["Loss","Validation Loss"], loc='upper left')
    plt.yscale('log')
    plt.savefig("%sLoss.png"%(plotsD), bbox_inches="tight")
    #print("Korrellationskoeffizient zwischen ")
    print("Korrellationskoeffizient zwischen Targets x, y", np.corrcoef(Outputs[Target_Pt],Outputs[Target_Phi]))
    print('sum(np.isnan(Outputs[NN_LongZ]))', sum(np.isnan(Outputs['NN_LongZ'])))
    print('sum(np.isnan(Outputs[NN_PerpZ]))', sum(np.isnan(Outputs['NN_PerpZ'])))
    print('sum(np.isnan(Outputs[NN_Pt]))', sum(np.isnan(Outputs['NN_Pt'])))
    print('sum(np.isnan(Outputs[NN_Phi]))', sum(np.isnan(Outputs['NN_Phi'])))
    print("Korrellationskoeffizient zwischen Predictions x, y", np.corrcoef(Outputs['NN_Pt'],Outputs['NN_Phi']))
    print("Korrellationskoeffizient zwischen Delta x, Delta y", np.corrcoef(Outputs['NN_Pt']-Outputs[Target_Pt],Outputs['NN_Phi']-Outputs[Target_Phi]))

    NN_Output.close()
    NN_Output_applied.close()

if __name__ == "__main__":
    outputDir = sys.argv[1]
    optim = str(sys.argv[2])
    loss_fct = str(sys.argv[3])
    NN_mode = sys.argv[4]
    plotsD = sys.argv[5]
    PhysicsProcess = sys.argv[6]
    rootInput = sys.argv[7]
    if PhysicsProcess == 'Tau':
        Target_Pt = 'Boson_Pt'
        Target_Phi = 'Boson_Phi'
        Outputs = loadData_woutGBRT(outputDir, rootInput, Target_Pt, Target_Phi, NN_mode, PhysicsProcess)
    else:
        Target_Pt = 'Boson_Pt'
        Target_Phi = 'Boson_Phi'
        Outputs = loadData_woutGBRT(outputDir, rootInput, Target_Pt, Target_Phi, NN_mode, PhysicsProcess)
    #Outputs = Outputs[~np.isnan(Outputs)]
    Outputs = Outputs[Outputs[Target_Pt]>0]
    Outputs = Outputs[Outputs[Target_Pt]<=200]
    Outputs = Outputs[Outputs['NVertex']<=50]
    print('len(Outputs)', len(Outputs))
    print('sum isnan Outputs', sum(np.isnan(Outputs['NN_LongZ'])))
    print('isnan Outputs', Outputs[np.isnan(Outputs['NN_LongZ'])])
    DFName = Outputs
    print(outputDir)
    plotTraining(outputDir, optim, loss_fct, NN_mode, plotsD, rootInput, PhysicsProcess, Target_Pt, Target_Phi)
