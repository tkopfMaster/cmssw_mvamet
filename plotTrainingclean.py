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

def Hist_Diff_norm_full(r,phi,pr,pphi, labelName, col):
    if labelName=='NN':
        x=r
        y=phi
    else:
        x = pol2kar_x(r,phi)
        y = pol2kar_y(r,phi)
    px = pol2kar_x(pr,pphi)
    py = pol2kar_y(pr,pphi)
    delta_x_sq = np.square(np.subtract(x,px))
    delta_y_sq = np.square(np.subtract(y,py))
    norm = np.sqrt(delta_x_sq+delta_y_sq)
    nbinsHist = 40
    if labelName in ['NN', 'PF'] :
        plt.hist(norm, bins=nbinsHist, range=[0, 200], label=labelName+', mean=%.2f$\pm$%.2f'%(np.mean(norm), np.std(norm)), histtype='step', ec=colors_InOut[col], linewidth=1.5, normed=False)
    else:
        plt.hist(norm, bins=nbinsHist, range=[0, 200], label=labelName+', mean=%.2f$\pm$%.2f'%(np.mean(norm), np.std(norm)), histtype='step', ec=colors_InOut[col], normed=False)



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
    pTRangeString_Err = '$20\ \mathrm{GeV} < |-\\vec{p}_T^Z| \leq 200\ \mathrm{GeV}$ \n $\mathrm{\# Vertex} \leq 50$'
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

        #plt.hist(np.subtract(InputsTargets['Track'][0,:], Targets[:,0]), bins=nbinsHist, range=[-50, 50], label='Track, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Track'][0,:], Targets[:,0])), np.std(np.subtract( InputsTargets['Track'][0,:] , Targets[:,0]))), histtype='step', ec=colors_InOut[0])
        #plt.hist(np.subtract(InputsTargets['NoPU'][0,:], Targets[:,0]), bins=nbinsHist, range=[-50, 50], label='NoPU, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['NoPU'][0,:], Targets[:,0])), np.std(np.subtract( InputsTargets['NoPU'][0,:] , Targets[:,0]))), histtype='step', ec=colors_InOut[2])
        #plt.hist(np.subtract(InputsTargets['PUCorrected'][0,:], Targets[:,0]), bins=nbinsHist, range=[-50, 50], label='PUCorrected, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PUCorrected'][0,:], Targets[:,0])), np.std(np.subtract(InputsTargets['PUCorrected'][0,:]  , Targets[:,0]))), histtype='step', ec=colors_InOut[3])
        #plt.hist(np.subtract(InputsTargets['PU'][0,:], Targets[:,0]), bins=nbinsHist, range=[-50, 50], label='PU, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PU'][0,:], Targets[:,0])), np.std(np.subtract(InputsTargets['PU'][0,:]  , Targets[:,0]))), histtype='step', ec=colors_InOut[4])
        plt.hist(np.subtract(InputsTargets['Puppi'][0,:], Targets[:,0]), bins=nbinsHist, range=[-50, 50], label='Puppi, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Puppi'][0,:], Targets[:,0])), np.std(np.subtract(InputsTargets['Puppi'][0,:]  , Targets[:,0]))), histtype='step', ec=colors_InOut[5])
        #plt.hist(np.subtract(pol2pol2kar_x(Outputs['LongZCorrectedRecoil_Pt'],Outputs['LongZCorrectedRecoil_Phi']), Targets[:,0]), bins=nbinsHist, range=[-50, 50], label='Puppi, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Puppi'][0,:], Targets[:,0])), np.std(np.subtract(InputsTargets['Puppi'][0,:]  , Targets[:,0]))), histtype='step', ec=colors_InOut[5])

        plt.hist(np.subtract(InputsTargets['PF'][0,:], Targets[:,0]), bins=nbinsHist, range=[-50, 50], label='PF, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PF'][0,:], Targets[:,0])), np.std(np.subtract(InputsTargets['PF'][0,:]  , Targets[:,0]))), histtype='step', ec=colors_InOut[1], linewidth=1.5)
        plt.hist(NN_Diff_x[:], bins=nbinsHist, range=[-50, 50], label='NN, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(predictions[:,0], Targets[:,0])), np.std(np.subtract(predictions[:,0]  , Targets[:,0]))), histtype='step', ec=colors_InOut[7], linewidth=1.5)


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
        plt.savefig("%sHist_Delta_x.png"%(plotsD), bbox_inches="tight")
        plt.close()


        fig=plt.figure(figsize=(10,6))
        fig.patch.set_facecolor('white')
        ax = plt.subplot(111)
        nbinsHist = 40

        #plt.hist(np.subtract(InputsTargets['Track'][1,:], Targets[:,1]), bins=nbinsHist, range=[-50, 50], label='Track, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Track'][1,:], Targets[:,1])), np.std(np.subtract( InputsTargets['Track'][1,:] , Targets[:,1]))), histtype='step', ec=colors_InOut[0])
        #plt.hist(np.subtract(InputsTargets['NoPU'][1,:], Targets[:,1]), bins=nbinsHist, range=[-50, 50], label='NoPU, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['NoPU'][1,:], Targets[:,1])), np.std(np.subtract( InputsTargets['NoPU'][1,:] , Targets[:,1]))), histtype='step', ec=colors_InOut[2])
        #plt.hist(np.subtract(InputsTargets['PUCorrected'][1,:], Targets[:,1]), bins=nbinsHist, range=[-50, 50], label='PUCorrected, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PUCorrected'][1,:], Targets[:,1])), np.std(np.subtract(InputsTargets['PUCorrected'][1,:]  , Targets[:,1]))), histtype='step', ec=colors_InOut[3])
        #plt.hist(np.subtract(InputsTargets['PU'][1,:], Targets[:,1]), bins=nbinsHist, range=[-50, 50], label='PU, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PU'][1,:], Targets[:,1])), np.std(np.subtract(InputsTargets['PU'][1,:]  , Targets[:,1]))), histtype='step', ec=colors_InOut[4])
        plt.hist(np.subtract(InputsTargets['Puppi'][1,:], Targets[:,1]), bins=nbinsHist, range=[-50, 50], label='Puppi, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Puppi'][1,:], Targets[:,1])), np.std(np.subtract(InputsTargets['Puppi'][1,:]  , Targets[:,1]))), histtype='step', ec=colors_InOut[5])
        #plt.hist(np.subtract(pol2pol2kar_x(Outputs['LongZCorrectedRecoil_Pt'],Outputs['LongZCorrectedRecoil_Phi']), Targets[:,1]), bins=nbinsHist, range=[-50, 50], label='Puppi, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Puppi'][1,:], Targets[:,1])), np.std(np.subtract(InputsTargets['Puppi'][1,:]  , Targets[:,1]))), histtype='step', ec=colors_InOut[5])

        plt.hist(np.subtract(InputsTargets['PF'][1,:], Targets[:,1]), bins=nbinsHist, range=[-50, 50], label='PF, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PF'][1,:], Targets[:,1])), np.std(np.subtract(InputsTargets['PF'][1,:]  , Targets[:,1]))), histtype='step', ec=colors_InOut[1], linewidth=1.5)
        plt.hist(NN_Diff_y[:], bins=nbinsHist, range=[-50, 50], label='NN, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(predictions[:,1], Targets[:,1])), np.std(np.subtract(predictions[:,1]  , Targets[:,1]))), histtype='step', ec=colors_InOut[7], linewidth=1.5)


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
        plt.savefig("%sHist_Delta_y.png"%(plotsD), bbox_inches="tight")
        plt.close()

        fig=plt.figure(figsize=(10,6))
        fig.patch.set_facecolor('white')
        ax = plt.subplot(111)
        nbinsHist = 40

        PF_Diff_x, PF_Diff_y = np.subtract(InputsTargets['PF'][0,:], Targets[:,0]), np.subtract(InputsTargets['PF'][1,:], Targets[:,1])
        PF_Diff_norm = np.sqrt(np.multiply(PF_Diff_x,PF_Diff_x)+np.multiply(PF_Diff_y,PF_Diff_y))
        Puppi_Diff_x, Puppi_Diff_y = np.subtract(InputsTargets['Puppi'][0,:], Targets[:,0]), np.subtract(InputsTargets['Puppi'][1,:], Targets[:,1])
        Puppi_Diff_norm = np.sqrt(np.multiply(Puppi_Diff_x,Puppi_Diff_x)+np.multiply(Puppi_Diff_y,Puppi_Diff_y))

        plt.hist(np.sqrt(np.multiply(Puppi_Diff_x, Puppi_Diff_x)+np.multiply(Puppi_Diff_y, Puppi_Diff_y)),  range=[0,200], label='Puppi, mean=%.2f $\pm$ %.2f'%(np.mean(np.sqrt(np.multiply(Puppi_Diff_x, Puppi_Diff_x)+np.multiply(Puppi_Diff_y, Puppi_Diff_y))), np.std(np.sqrt(np.multiply(Puppi_Diff_x, Puppi_Diff_x)+np.multiply(Puppi_Diff_y, Puppi_Diff_y)))), histtype='step', bins=nbinsHist, ec=colors_InOut[5])
        plt.hist(np.sqrt(np.multiply(PF_Diff_x, PF_Diff_x)+np.multiply(PF_Diff_y, PF_Diff_y)),  range=[0,200], label='PF, mean=%.2f $\pm$ %.2f'%(np.mean(np.sqrt(np.multiply(PF_Diff_x, PF_Diff_x)+np.multiply(PF_Diff_y, PF_Diff_y))), np.std(np.sqrt(np.multiply(PF_Diff_x, PF_Diff_x)+np.multiply(PF_Diff_y, PF_Diff_y)))), histtype='step', bins=nbinsHist, ec=colors_InOut[1])
        plt.hist(np.sqrt(np.multiply(NN_Diff_x, NN_Diff_x)+np.multiply(NN_Diff_y, NN_Diff_y)),  range=[0,200], label='NN, mean=%.2f $\pm$ %.2f'%(np.mean(np.sqrt(np.multiply(NN_Diff_x, NN_Diff_x)+np.multiply(NN_Diff_y, NN_Diff_y))), np.std(np.sqrt(np.multiply(NN_Diff_x, NN_Diff_x)+np.multiply(NN_Diff_y, NN_Diff_y)))), histtype='step', bins=nbinsHist, ec=colors_InOut[6])


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
    Outputs = Outputs[Outputs[Target_Pt]>20]
    Outputs = Outputs[Outputs[Target_Pt]<=200]
    Outputs = Outputs[Outputs['NVertex']<=50]
    print('len(Outputs)', len(Outputs))
    print('sum isnan Outputs', sum(np.isnan(Outputs['NN_LongZ'])))
    print('isnan Outputs', Outputs[np.isnan(Outputs['NN_LongZ'])])
    DFName = Outputs
    print(outputDir)
    plotTraining(outputDir, optim, loss_fct, NN_mode, plotsD, rootInput, PhysicsProcess, Target_Pt, Target_Phi)
