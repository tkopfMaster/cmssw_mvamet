import h5py
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
from matplotlib.colors import LogNorm
import matplotlib.patches as mpatches
from getPlotsOutputclean import loadData, loadData_woutGBRT
from getResponse import getResponse, getResponseIdx
from prepareInput import pol2kar_x, pol2kar_y, kar2pol, pol2kar, angularrange



nbinsHist = 400
LegendTitle = '$\mathrm{Summer\ 17\ campaign}$' '\n'  '$\mathrm{Z \  \\rightarrow \ \mu \mu}$'
colors_InOut = cm.brg(np.linspace(0, 1, 8))
diff_p_min, diff_p_max = 0, 50
pTRangeString= '$0\ \mathrm{GeV} < |\\vec{\mathrm{MET}}| \leq 200\ \mathrm{GeV}$ \n $\mathrm{\# Vertex} \leq 50$'
pTRangeString_low= '$0\ \mathrm{GeV} < |\\vec{\mathrm{MET}}| \leq 10\ \mathrm{GeV}$ \n $\mathrm{\# Vertex} \leq 50$'
pTRangeString_mid= '$10\ \mathrm{GeV} < |\\vec{\mathrm{MET}}| \leq 125\ \mathrm{GeV}$ \n $\mathrm{\# Vertex} \leq 50$'
pTRangeString_high= '$125\ \mathrm{GeV} < |\\vec{\mathrm{MET}}| \leq 200\ \mathrm{GeV}$ \n $\mathrm{\# Vertex} \leq 50$'

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
    if labelName=='NN':
        x = r[(pr > rangemin) & (pr < rangemax)]
        y = phi[(pr > rangemin) & (pr < rangemax)]
    else:
        x = pol2kar_x(r[(pr > rangemin) & (pr < rangemax)],phi[(pr > rangemin) & (pr < rangemax)])
        y = pol2kar_y(r[(pr > rangemin) & (pr < rangemax)],phi[(pr > rangemin) & (pr < rangemax)])
    px = -pol2kar_x(pr[(pr > rangemin) & (pr < rangemax)],pphi[(pr > rangemin) & (pr < rangemax)])
    py = -pol2kar_y(pr[(pr > rangemin) & (pr < rangemax)],pphi[(pr > rangemin) & (pr < rangemax)])
    delta_x_sq = np.square(np.subtract(x,px))
    delta_y_sq = np.square(np.subtract(y,py))
    norm = np.sqrt(delta_x_sq+delta_y_sq)

    if labelName in ['NN', 'PF'] :
        plt.hist(norm, bins=nbinsHist, range=[diff_p_min, diff_p_max], label=labelName+', mean=%.2f$\pm$%.2f'%(np.mean(norm), np.std(norm)), histtype='step', ec=colors_InOut[col], linewidth=1.5, normed=True)
    else:
        plt.hist(norm, bins=nbinsHist, range=[diff_p_min, diff_p_max], label=labelName+', mean=%.2f$\pm$%.2f'%(np.mean(norm), np.std(norm)), histtype='step', ec=colors_InOut[col], normed=True)

def HM_Diff_norm(r,phi,pr,pphi, labelName, col):
    if labelName=='NN':
        x = r
        y = phi
    else:
        x = pol2kar_x(r,phi)
        y = pol2kar_y(r,phi)
    px = -pol2kar_x(pr,pphi)
    py = -pol2kar_y(pr,pphi)
    delta_x_sq = np.square(np.subtract(x,px))
    delta_y_sq = np.square(np.subtract(y,py))
    norm = np.sqrt(delta_x_sq+delta_y_sq)
    return norm


def plotTraining(outputD, optim, loss_fct, NN_mode, plotsD, rootOutput, PhysicsProcess):
    NN_Output_applied = h5py.File("%sNN_Output_applied_%s.h5"%(outputD,NN_mode), "r")
    predictions = NN_Output_applied["MET_Predictions"]
    Targets = NN_Output_applied["MET_GroundTruth"]
    print("Shape predictions", predictions)
    print("Shape Targets", Targets)
    print('np.subtract(predictions[:,0], Targets[:,0])', np.subtract(predictions[:,0], Targets[:,0]))
    print('np.mean(np.subtract(predictions[:,0], Targets[:,0]))', np.mean(np.subtract(predictions[:,0], Targets[:,0])))
    print('np.subtract(predictions[:,1], Targets[:,1])', np.subtract(predictions[:,1], Targets[:,1]))
    print('np.mean(np.subtract(predictions[:,0], Targets[:,0]))', np.mean(np.subtract(predictions[:,1], Targets[:,1])))

    NN_Diff_x =  np.subtract(predictions[:,0], Targets[:,0])
    NN_Diff_y = np.subtract(predictions[:,1], Targets[:,1])

    NN_Output = h5py.File("%sNN_Output_%s.h5"%(outputD,NN_mode), "r")
    loss = NN_Output["loss"]
    val_loss = NN_Output["val_loss"]



    if NN_mode == 'xy':
        #Load Inputs and Targets with Name
        InputsTargets = h5py.File("%sNN_Input_%s.h5" % (outputD,NN_mode), "r")



        fig=plt.figure(figsize=(10,6))
        fig.patch.set_facecolor('white')
        ax = plt.subplot(111)
        nbinsHist = 250

        Hist_Diff_norm(Outputs['recoilpatpfTrackMET_Pt'], Outputs['recoilpatpfTrackMET_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'Track', 0, 0, 200)
        Hist_Diff_norm(Outputs['recoilpatpfNoPUMET_Pt'], Outputs['recoilpatpfNoPUMET_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'NoPU', 2, 0, 200)
        Hist_Diff_norm(Outputs['recoilpatpfPUCorrectedMET_Pt'], Outputs['recoilpatpfPUCorrectedMET_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'PUCorrected', 3, 0, 200)
        Hist_Diff_norm(Outputs['recoilpatpfPUMET_Pt'], Outputs['recoilpatpfPUMET_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'PU', 4, 0, 200)
        Hist_Diff_norm(Outputs['recoilslimmedMETsPuppi_Pt'], Outputs['recoilslimmedMETsPuppi_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'Puppi', 5, 0, 200)
        if not PhysicsProcess=='Tau':
            Hist_Diff_norm(Outputs['LongZCorrectedRecoil_Pt'], Outputs['LongZCorrectedRecoil_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'GBRT', 7, 0, 200)
        Hist_Diff_norm(Outputs['recoilslimmedMETs_Pt'], Outputs['recoilslimmedMETs_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'PF', 1, 0, 200)
        Hist_Diff_norm(predictions[:,0], predictions[:,1], Outputs[Target_Pt], Outputs[Target_Phi], 'NN', 6, 0, 200)


        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        handles, labels = ax.get_legend_handles_labels()

        plt.ylabel('Counts')
        plt.xlabel('$|\\vec{U}-\\vec{MET}|$ in GeV')
        plt.xlim(diff_p_min,diff_p_max)

        ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
        plt.grid()
        #plt.ylim(ylimResMVAMin, ylimResMax)
        plt.savefig("%sHist_Diff_norm.png"%(plotsD), bbox_inches="tight")


        Norm_Diff = HM_Diff_norm(Outputs['recoilslimmedMETs_Pt'], Outputs['recoilslimmedMETs_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'PF', 1)
        '''
        plt.clf()
        plt.figure()
        #plt.suptitle('x-Korrektur Prediction-Target ')
        plt.ylabel("$ |\\vec{U}- \\vec{\mathrm{MET}}| $")
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
        plt.ylabel("$ \\frac{|\\vec{U}- \\vec{\mathrm{MET}}|}{|\\vec{\mathrm{MET}}|} $")
        plt.xlabel("Response")
        rel_Norm_Diff = np.divide(HM_Diff_norm(predictions[:,0], predictions[:,1], Outputs[Target_Pt], Outputs[Target_Phi], 'NN', 6), Outputs[Target_Pt])
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
        plt.ylabel("$ \\frac{|\\vec{U}- \\vec{\mathrm{MET}}|}{|\\vec{\mathrm{MET}}|} $")
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
        '''

        plt.clf()
        plt.figure()
        #plt.suptitle('x-Korrektur Prediction-Target ')
        plt.ylabel("$ \\frac{|\\vec{U}- \\vec{\mathrm{MET}}|}{|\\vec{\mathrm{MET}}|} $")
        plt.xlabel("$U_{\parallel}-\\vec{\mathrm{MET}}$")
        rel_Norm_Diff = np.divide(HM_Diff_norm(predictions[:,0], predictions[:,1], Outputs[Target_Pt], Outputs[Target_Phi], 'NN', 6), Outputs[Target_Pt])
        Resolution_para = -Outputs['NN_LongZ']-Outputs[Target_Pt]
        heatmap, xedges, yedges = np.histogram2d(  Resolution_para[~np.isnan(rel_Norm_Diff)], rel_Norm_Diff[~np.isnan(rel_Norm_Diff)],  bins=50,
                                                 range=[[0,3],
                                                        [0,4]])
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        #plt.clf()
        HM =plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm(), aspect='auto')
        plt.colorbar(HM)
        plt.legend()
        plt.savefig("%sHM_NN_Resolution_para_relDiffnorm.png"%(plotsD), bbox_inches="tight")


        plt.clf()
        plt.figure()
        #plt.suptitle('x-Korrektur Prediction-Target ')
        plt.ylabel("$ \\frac{|\\vec{U}- \\vec{\mathrm{MET}}|}{|\\vec{\mathrm{MET}}|} $")
        plt.xlabel("$U_{\parallel}-\\vec{\mathrm{MET}}$")
        rel_Norm_Diff = np.divide(HM_Diff_norm(Outputs['recoilslimmedMETs_Pt'], Outputs['recoilslimmedMETs_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'PF', 1), Outputs[Target_Pt])
        Resolution_para = -Outputs['recoilslimmedMETs_LongZ']-Outputs[Target_Pt]
        heatmap, xedges, yedges = np.histogram2d(  Resolution_para[~np.isnan(rel_Norm_Diff)], rel_Norm_Diff[~np.isnan(rel_Norm_Diff)],  bins=50,
                                                 range=[[0,3],
                                                        [0,4]])
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        #plt.clf()
        HM =plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm(), aspect='auto')
        plt.colorbar(HM)
        plt.legend()
        plt.savefig("%sHM_PF_Resolution_para_relDiffnorm.png"%(plotsD), bbox_inches="tight")

        plt.clf()
        plt.figure()
        #plt.suptitle('x-Korrektur Prediction-Target ')
        plt.ylabel("$ \\frac{|\\vec{U}- \\vec{\mathrm{MET}}|}{|\\vec{\mathrm{MET}}|} $")
        plt.xlabel("$U_{\perp}$")
        rel_Norm_Diff = np.divide(HM_Diff_norm(predictions[:,0], predictions[:,1], Outputs[Target_Pt], Outputs[Target_Phi], 'NN', 6), Outputs[Target_Pt])
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
        plt.ylabel("$ \\frac{|\\vec{U}- \\vec{\mathrm{MET}}|}{|\\vec{\mathrm{MET}}|} $")
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

        '''
        plt.clf()
        plt.figure()
        #plt.suptitle('x-Korrektur Prediction-Target ')
        plt.xlabel("$ |\\vec{U}- \\vec{\mathrm{MET}}| $")
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
        '''






        fig=plt.figure(figsize=(10,6))
        fig.patch.set_facecolor('white')
        ax = plt.subplot(111)



        Histogram_Angle(Outputs['recoilpatpfNoPUMET_Phi'], 'No PU MET',0)
        Histogram_Angle(Outputs['recoilpatpfPUCorrectedMET_Phi'],'PU corrected MET',1)
        Histogram_Angle(Outputs['recoilpatpfPUMET_Phi'], 'PU MET',2)
        Histogram_Angle(Outputs['recoilpatpfTrackMET_Phi'], 'Track MET',3)
        Histogram_Angle(Outputs['recoilslimmedMETsPuppi_Phi'], 'Puppi MET',4)
        if not PhysicsProcess=='Tau':
            Histogram_Angle(Outputs['LongZCorrectedRecoil_Phi'], 'GBRT MET', 5)
        Histogram_Angle(Outputs['recoilslimmedMETs_Phi'], 'PF MET', 1)
        NN_r, NN_phi = kar2pol(predictions[:,0], predictions[:,1])
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








        print('shape predictions[:,0]', predictions[:,0].shape)
        print('shape predictions[:,0]+sum(Outputs[Boson_Pt]==0)', len(predictions[:,0])+sum(Outputs[Target_Pt]==0))

        '''
        fig=plt.figure(figsize=(10,6))
        fig.patch.set_facecolor('white')
        ax = plt.subplot(111)
        nbinsHist = 250

        Hist_Diff_norm(Outputs['recoilpatpfTrackMET_Pt'], Outputs['recoilpatpfTrackMET_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'Track', 0)
        Hist_Diff_norm(Outputs['recoilpatpfNoPUMET_Pt'], Outputs['recoilpatpfNoPUMET_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'NoPU', 2)
        Hist_Diff_norm(Outputs['recoilpatpfPUCorrectedMET_Pt'], Outputs['recoilpatpfPUCorrectedMET_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'PUCorrected', 3)
        Hist_Diff_norm(Outputs['recoilpatpfPUMET_Pt'], Outputs['recoilpatpfPUMET_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'PU', 4)
        Hist_Diff_norm(Outputs['recoilslimmedMETsPuppi_Pt'], Outputs['recoilslimmedMETsPuppi_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'Puppi', 5)
        Hist_Diff_norm(Outputs['LongZCorrectedRecoil_Pt'], Outputs['LongZCorrectedRecoil_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'GBRT', 7)
        Hist_Diff_norm(Outputs['recoilslimmedMETs_Pt'], Outputs['recoilslimmedMETs_Phi'], Outputs[Target_Pt], Outputs[Target_Phi], 'PF', 1)
        print("Zeile 110")
        #Hist_Diff_norm(predictions[:,0], predictions[:,1], Outputs[Target_Pt], Outputs[Target_Phi], 'NN', 6)
        print("Zeile 112")

        norm_ = np.sqrt(np.square(predictions[:,0]-pol2kar_x(Outputs[Target_Pt], Outputs[Target_Phi]))+np.square(predictions[:,1]-pol2kar_y(Outputs[Target_Pt], Outputs[Target_Phi])))
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
        plt.xlabel('$|\\vec{U}-\\vec{MET}|$ in GeV')
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
        nbinsHist = 250

        plt.hist(InputsTargets['Track'][0,:], bins=nbinsHist, range=[-50, 50], label='Track, mean=%.2f'%np.mean(InputsTargets['Track'][0,:]), histtype='step', ec=colors_InOut[0])
        plt.hist(InputsTargets['NoPU'][0,:], bins=nbinsHist, range=[-50, 50], label='NoPU, mean=%.2f'%np.mean(InputsTargets['NoPU'][0,:]), histtype='step', ec=colors_InOut[2])
        plt.hist(InputsTargets['PUCorrected'][0,:], bins=nbinsHist, range=[-50, 50], label='PUCorrected, mean=%.2f'%np.mean(InputsTargets['PUCorrected'][0,:]), histtype='step', ec=colors_InOut[3])
        plt.hist(InputsTargets['PU'][0,:], bins=nbinsHist, range=[-50, 50], label='PU, mean=%.2f'%np.mean(InputsTargets['PU'][0,:]), histtype='step', ec=colors_InOut[4])
        plt.hist(InputsTargets['Puppi'][0,:], bins=nbinsHist, range=[-50, 50], label='Puppi, mean=%.2f'%np.mean(InputsTargets['Puppi'][0,:]), histtype='step', ec=colors_InOut[5])
        plt.hist(InputsTargets['PF'][0,:], bins=nbinsHist, range=[-50, 50], label='PF, mean=%.2f'%np.mean(InputsTargets['PF'][0,:]), histtype='step', ec=colors_InOut[1], linewidth=1.5)
        plt.hist(Targets[:,0], bins=nbinsHist, range=[-50, 50], label='Target, mean=%.2f'%np.mean(Targets[:,0]), histtype='step', ec=colors_InOut[6], linewidth=1.5)
        plt.hist(predictions[:,0], bins=nbinsHist, range=[-50, 50], label='Prediction, mean=%.2f'%np.mean(predictions[:,0]), histtype='step', ec=colors_InOut[7], linewidth=1.5)



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
        nbinsHist = 250

        plt.hist(InputsTargets['Track'][1,:], bins=nbinsHist, range=[-50, 50], label='Track, mean=%.2f'%np.mean(InputsTargets['Track'][1,:]), histtype='step', ec=colors_InOut[0])
        plt.hist(InputsTargets['NoPU'][1,:], bins=nbinsHist, range=[-50, 50], label='NoPU, mean=%.2f'%np.mean(InputsTargets['NoPU'][1,:]), histtype='step', ec=colors_InOut[2])
        plt.hist(InputsTargets['PUCorrected'][1,:], bins=nbinsHist, range=[-50, 50], label='PUCorrected, mean=%.2f'%np.mean(InputsTargets['PUCorrected'][1,:]), histtype='step', ec=colors_InOut[3])
        plt.hist(InputsTargets['PU'][1,:], bins=nbinsHist, range=[-50, 50], label='PU, mean=%.2f'%np.mean(InputsTargets['PU'][1,:]), histtype='step', ec=colors_InOut[4])
        plt.hist(InputsTargets['Puppi'][1,:], bins=nbinsHist, range=[-50, 50], label='Puppi, mean=%.2f'%np.mean(InputsTargets['Puppi'][1,:]), histtype='step', ec=colors_InOut[5])
        plt.hist(InputsTargets['PF'][1,:], bins=nbinsHist, range=[-50, 50], label='PF, mean=%.2f'%np.mean(InputsTargets['PF'][1,:]), histtype='step', ec=colors_InOut[1], linewidth=1.5)
        plt.hist(Targets[:,1], bins=nbinsHist, range=[-50, 50], label='Target, mean=%.2f'%np.mean(Targets[:,1]), histtype='step', ec=colors_InOut[6], linewidth=1.5)
        plt.hist(predictions[:,1], bins=nbinsHist, range=[-50, 50], label='Prediction, mean=%.2f'%np.mean(predictions[:,1]), histtype='step', ec=colors_InOut[7], linewidth=1.5)


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
        nbinsHist = 250

        plt.hist(np.subtract(InputsTargets['Track'][0,:], Targets[:,0]), bins=nbinsHist, range=[-50, 50], label='Track, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Track'][0,:], Targets[:,0])), np.std(np.subtract( InputsTargets['Track'][0,:] , Targets[:,0]))), histtype='step', ec=colors_InOut[0])
        plt.hist(np.subtract(InputsTargets['NoPU'][0,:], Targets[:,0]), bins=nbinsHist, range=[-50, 50], label='NoPU, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['NoPU'][0,:], Targets[:,0])), np.std(np.subtract( InputsTargets['NoPU'][0,:] , Targets[:,0]))), histtype='step', ec=colors_InOut[2])
        plt.hist(np.subtract(InputsTargets['PUCorrected'][0,:], Targets[:,0]), bins=nbinsHist, range=[-50, 50], label='PUCorrected, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PUCorrected'][0,:], Targets[:,0])), np.std(np.subtract(InputsTargets['PUCorrected'][0,:]  , Targets[:,0]))), histtype='step', ec=colors_InOut[3])
        plt.hist(np.subtract(InputsTargets['PU'][0,:], Targets[:,0]), bins=nbinsHist, range=[-50, 50], label='PU, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PU'][0,:], Targets[:,0])), np.std(np.subtract(InputsTargets['PU'][0,:]  , Targets[:,0]))), histtype='step', ec=colors_InOut[4])
        plt.hist(np.subtract(InputsTargets['Puppi'][0,:], Targets[:,0]), bins=nbinsHist, range=[-50, 50], label='Puppi, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Puppi'][0,:], Targets[:,0])), np.std(np.subtract(InputsTargets['Puppi'][0,:]  , Targets[:,0]))), histtype='step', ec=colors_InOut[5])
        #plt.hist(np.subtract(pol2pol2kar_x(Outputs['LongZCorrectedRecoil_Pt'],Outputs['LongZCorrectedRecoil_Phi']), Targets[:,0]), bins=nbinsHist, range=[-50, 50], label='Puppi, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Puppi'][0,:], Targets[:,0])), np.std(np.subtract(InputsTargets['Puppi'][0,:]  , Targets[:,0]))), histtype='step', ec=colors_InOut[5])

        plt.hist(np.subtract(InputsTargets['PF'][0,:], Targets[:,0]), bins=nbinsHist, range=[-50, 50], label='PF, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PF'][0,:], Targets[:,0])), np.std(np.subtract(InputsTargets['PF'][0,:]  , Targets[:,0]))), histtype='step', ec=colors_InOut[1], linewidth=1.5)
        plt.hist(NN_Diff_x, bins=nbinsHist, range=[-50, 50], label='NN, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(predictions[:,0], Targets[:,0])), np.std(np.subtract(predictions[:,0]  , Targets[:,0]))), histtype='step', ec=colors_InOut[7], linewidth=1.5)


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
        plt.savefig("%sHist_Delta_x.png"%(plotsD), bbox_inches="tight")
        plt.close()


        fig=plt.figure(figsize=(10,6))
        fig.patch.set_facecolor('white')
        ax = plt.subplot(111)
        nbinsHist = 250
        pTnorm = np.sqrt( np.multiply(Targets[:,0],Targets[:,0])+ np.multiply(Targets[:,1],Targets[:,1]))
        IdxpTlow = pTnorm<=10
        IdxpTmid = (pTnorm>10) & (pTnorm<=125)
        IdxpThigh = (pTnorm>125) & (pTnorm<=200)
        plt.hist(np.subtract(InputsTargets['Track'][0,IdxpTlow], Targets[IdxpTlow,0]), bins=nbinsHist, range=[-50, 50], label='Track, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Track'][0,:], Targets[:,0])), np.std(np.subtract( InputsTargets['Track'][0,:] , Targets[:,0]))), histtype='step', ec=colors_InOut[0])
        plt.hist(np.subtract(InputsTargets['NoPU'][0,IdxpTlow], Targets[IdxpTlow,0]), bins=nbinsHist, range=[-50, 50], label='NoPU, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['NoPU'][0,:], Targets[:,0])), np.std(np.subtract( InputsTargets['NoPU'][0,:] , Targets[:,0]))), histtype='step', ec=colors_InOut[2])
        plt.hist(np.subtract(InputsTargets['PUCorrected'][0,IdxpTlow], Targets[IdxpTlow,0]), bins=nbinsHist, range=[-50, 50], label='PUCorrected, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PUCorrected'][0,:], Targets[:,0])), np.std(np.subtract(InputsTargets['PUCorrected'][0,:]  , Targets[:,0]))), histtype='step', ec=colors_InOut[3])
        plt.hist(np.subtract(InputsTargets['PU'][0,IdxpTlow], Targets[IdxpTlow,0]), bins=nbinsHist, range=[-50, 50], label='PU, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PU'][0,:], Targets[:,0])), np.std(np.subtract(InputsTargets['PU'][0,:]  , Targets[:,0]))), histtype='step', ec=colors_InOut[4])
        plt.hist(np.subtract(InputsTargets['Puppi'][0,IdxpTlow], Targets[IdxpTlow,0]), bins=nbinsHist, range=[-50, 50], label='Puppi, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Puppi'][0,:], Targets[:,0])), np.std(np.subtract(InputsTargets['Puppi'][0,:]  , Targets[:,0]))), histtype='step', ec=colors_InOut[5])
        #plt.hist(np.subtract(pol2pol2kar_x(Outputs['LongZCorrectedRecoil_Pt'],Outputs['LongZCorrectedRecoil_Phi']), Targets[:,0]), bins=nbinsHist, range=[-50, 50], label='Puppi, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Puppi'][0,:], Targets[:,0])), np.std(np.subtract(InputsTargets['Puppi'][0,:]  , Targets[:,0]))), histtype='step', ec=colors_InOut[5])

        plt.hist(np.subtract(InputsTargets['PF'][0,IdxpTlow], Targets[IdxpTlow,0]), bins=nbinsHist, range=[-50, 50], label='PF, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PF'][0,:], Targets[:,0])), np.std(np.subtract(InputsTargets['PF'][0,:]  , Targets[:,0]))), histtype='step', ec=colors_InOut[1], linewidth=1.5)
        plt.hist(NN_Diff_x[IdxpTlow], bins=nbinsHist, range=[-50, 50], label='NN, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(predictions[:,0], Targets[:,0])), np.std(np.subtract(predictions[:,0]  , Targets[:,0]))), histtype='step', ec=colors_InOut[7], linewidth=1.5)


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
        nbinsHist = 250
        pTnorm = np.sqrt( np.multiply(Targets[:,0],Targets[:,0])+ np.multiply(Targets[:,1],Targets[:,1]))
        IdxpTlow = InputsTargets[pTnorm<=10]
        IdxpTmid = InputsTargets[(pTnorm>10) & (pTnorm<=125)]
        IdxpThigh = InputsTargets[(pTnorm>125) & (pTnorm<=200)]
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



        fig=plt.figure(figsize=(10,6))
        fig.patch.set_facecolor('white')
        ax = plt.subplot(111)
        nbinsHist = 250
        pTnorm = np.sqrt( np.multiply(Targets[:,0],Targets[:,0])+ np.multiply(Targets[:,1],Targets[:,1]))
        IdxpTlow = InputsTargets[pTnorm<=10]
        IdxpTmid = InputsTargets[(pTnorm>10) & (pTnorm<=125)]
        IdxpTmid = InputsTargets[(pTnorm>125) & (pTnorm<=200)]
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




        fig=plt.figure(figsize=(10,6))
        fig.patch.set_facecolor('white')
        ax = plt.subplot(111)
        nbinsHist = 250

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
        nbinsHist = 250

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

        fig=plt.figure(figsize=(10,6))
        fig.patch.set_facecolor('white')
        ax = plt.subplot(111)
        nbinsHist = 250

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
        nbinsHist = 250

        plt.hist(np.subtract(InputsTargets['Track'][1,:], Targets[:,1]), bins=nbinsHist, range=[-50, 50], label='Track, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Track'][1,:], Targets[:,1])), np.std(np.subtract( InputsTargets['Track'][1,:] , Targets[:,1]))), histtype='step', ec=colors_InOut[0])
        plt.hist(np.subtract(InputsTargets['NoPU'][1,:], Targets[:,1]), bins=nbinsHist, range=[-50, 50], label='NoPU, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['NoPU'][1,:], Targets[:,1])), np.std(np.subtract( InputsTargets['NoPU'][1,:] , Targets[:,1]))), histtype='step', ec=colors_InOut[2])
        plt.hist(np.subtract(InputsTargets['PUCorrected'][1,:], Targets[:,1]), bins=nbinsHist, range=[-50, 50], label='PUCorrected, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PUCorrected'][1,:], Targets[:,1])), np.std(np.subtract(InputsTargets['PUCorrected'][1,:]  , Targets[:,1]))), histtype='step', ec=colors_InOut[3])
        plt.hist(np.subtract(InputsTargets['PU'][1,:], Targets[:,1]), bins=nbinsHist, range=[-50, 50], label='PU, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PU'][1,:], Targets[:,1])), np.std(np.subtract(InputsTargets['PU'][1,:]  , Targets[:,1]))), histtype='step', ec=colors_InOut[4])
        plt.hist(np.subtract(InputsTargets['Puppi'][1,:], Targets[:,1]), bins=nbinsHist, range=[-50, 50], label='Puppi, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Puppi'][1,:], Targets[:,1])), np.std(np.subtract(InputsTargets['Puppi'][1,:]  , Targets[:,1]))), histtype='step', ec=colors_InOut[5])
        #plt.hist(np.subtract(pol2pol2kar_x(Outputs['LongZCorrectedRecoil_Pt'],Outputs['LongZCorrectedRecoil_Phi']), Targets[:,1]), bins=nbinsHist, range=[-50, 50], label='Puppi, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Puppi'][1,:], Targets[:,1])), np.std(np.subtract(InputsTargets['Puppi'][1,:]  , Targets[:,1]))), histtype='step', ec=colors_InOut[5])

        plt.hist(np.subtract(InputsTargets['PF'][1,:], Targets[:,1]), bins=nbinsHist, range=[-50, 50], label='PF, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PF'][1,:], Targets[:,1])), np.std(np.subtract(InputsTargets['PF'][1,:]  , Targets[:,1]))), histtype='step', ec=colors_InOut[1], linewidth=1.5)
        plt.hist(NN_Diff_y[:], bins=nbinsHist, range=[-50, 50], label='NN, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(predictions[:,1], Targets[:,1])), np.std(np.subtract(predictions[:,1]  , Targets[:,1]))), histtype='step', ec=colors_InOut[7], linewidth=1.5)


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
        plt.savefig("%sHist_Delta_y.png"%(plotsD), bbox_inches="tight")
        plt.close()



        fig=plt.figure(figsize=(10,6))
        fig.patch.set_facecolor('white')
        ax = plt.subplot(111)
        nbinsHist = 250

        plt.hist(np.subtract(InputsTargets['Track'][0,:], Targets[:,0]), bins=nbinsHist, range=[-50, 50], label='Track, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Track'][0,:], Targets[:,0])), np.std(np.subtract( InputsTargets['Track'][0,:] , Targets[:,0]))), histtype='step', ec=colors_InOut[0], normed=True)
        plt.hist(np.subtract(InputsTargets['NoPU'][0,:], Targets[:,0]), bins=nbinsHist, range=[-50, 50], label='NoPU, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['NoPU'][0,:], Targets[:,0])), np.std(np.subtract( InputsTargets['NoPU'][0,:] , Targets[:,0]))), histtype='step', ec=colors_InOut[2], normed=True)
        plt.hist(np.subtract(InputsTargets['PUCorrected'][0,:], Targets[:,0]), bins=nbinsHist, range=[-50, 50], label='PUCorrected, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PUCorrected'][0,:], Targets[:,0])), np.std(np.subtract(InputsTargets['PUCorrected'][0,:]  , Targets[:,0]))), histtype='step', ec=colors_InOut[3], normed=True)
        plt.hist(np.subtract(InputsTargets['PU'][0,:], Targets[:,0]), bins=nbinsHist, range=[-50, 50], label='PU, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PU'][0,:], Targets[:,0])), np.std(np.subtract(InputsTargets['PU'][0,:]  , Targets[:,0]))), histtype='step', ec=colors_InOut[4], normed=True)
        plt.hist(np.subtract(InputsTargets['Puppi'][0,:], Targets[:,0]), bins=nbinsHist, range=[-50, 50], label='Puppi, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Puppi'][0,:], Targets[:,0])), np.std(np.subtract(InputsTargets['Puppi'][0,:]  , Targets[:,0]))), histtype='step', ec=colors_InOut[5], normed=True)
        #plt.hist(np.subtract(pol2pol2kar_x(Outputs['LongZCorrectedRecoil_Pt'],Outputs['LongZCorrectedRecoil_Phi']), Targets[:,0]), bins=nbinsHist, range=[-50, 50], label='Puppi, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Puppi'][0,:], Targets[:,0])), np.std(np.subtract(InputsTargets['Puppi'][0,:]  , Targets[:,0]))), histtype='step', ec=colors_InOut[5])

        plt.hist(np.subtract(InputsTargets['PF'][0,:], Targets[:,0]), bins=nbinsHist, range=[-50, 50], label='PF, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PF'][0,:], Targets[:,0])), np.std(np.subtract(InputsTargets['PF'][0,:]  , Targets[:,0]))), histtype='step', ec=colors_InOut[1], linewidth=1.5, normed=True)
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
        nbinsHist = 250

        plt.hist(np.subtract(InputsTargets['Track'][1,:], Targets[:,1]), bins=nbinsHist, range=[-50, 50], label='Track, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Track'][1,:], Targets[:,1])), np.std(np.subtract( InputsTargets['Track'][1,:] , Targets[:,1]))), histtype='step', ec=colors_InOut[0], normed=True)
        plt.hist(np.subtract(InputsTargets['NoPU'][1,:], Targets[:,1]), bins=nbinsHist, range=[-50, 50], label='NoPU, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['NoPU'][1,:], Targets[:,1])), np.std(np.subtract( InputsTargets['NoPU'][1,:] , Targets[:,1]))), histtype='step', ec=colors_InOut[2], normed=True)
        plt.hist(np.subtract(InputsTargets['PUCorrected'][1,:], Targets[:,1]), bins=nbinsHist, range=[-50, 50], label='PUCorrected, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PUCorrected'][1,:], Targets[:,1])), np.std(np.subtract(InputsTargets['PUCorrected'][1,:]  , Targets[:,1]))), histtype='step', ec=colors_InOut[3], normed=True)
        plt.hist(np.subtract(InputsTargets['PU'][1,:], Targets[:,1]), bins=nbinsHist, range=[-50, 50], label='PU, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PU'][1,:], Targets[:,1])), np.std(np.subtract(InputsTargets['PU'][1,:]  , Targets[:,1]))), histtype='step', ec=colors_InOut[4], normed=True)
        plt.hist(np.subtract(InputsTargets['Puppi'][1,:], Targets[:,1]), bins=nbinsHist, range=[-50, 50], label='Puppi, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Puppi'][1,:], Targets[:,1])), np.std(np.subtract(InputsTargets['Puppi'][1,:]  , Targets[:,1]))), histtype='step', ec=colors_InOut[5], normed=True)
        #plt.hist(np.subtract(pol2pol2kar_x(Outputs['LongZCorrectedRecoil_Pt'],Outputs['LongZCorrectedRecoil_Phi']), Targets[:,1]), bins=nbinsHist, range=[-50, 50], label='Puppi, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['Puppi'][1,:], Targets[:,1])), np.std(np.subtract(InputsTargets['Puppi'][1,:]  , Targets[:,1]))), histtype='step', ec=colors_InOut[5])

        plt.hist(np.subtract(InputsTargets['PF'][1,:], Targets[:,1]), bins=nbinsHist, range=[-50, 50], label='PF, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(InputsTargets['PF'][1,:], Targets[:,1])), np.std(np.subtract(InputsTargets['PF'][1,:]  , Targets[:,1]))), histtype='step', ec=colors_InOut[1], linewidth=1.5, normed=True)
        plt.hist(NN_Diff_y, bins=nbinsHist, range=[-50, 50], label='NN, mean=%.2f$\pm$%.2f'%(np.mean(np.subtract(predictions[:,1], Targets[:,1])), np.std(np.subtract(predictions[:,1]  , Targets[:,1]))), histtype='step', ec=colors_InOut[7], linewidth=1.5, normed=True)

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


    plt.clf()
    plt.figure()
    #plt.suptitle('x-Korrektur PF-Target ')
    plt.xlabel("$ p_{T,x}^Z$")
    plt.ylabel("PF-Target")
    heatmap, xedges, yedges = np.histogram2d(   Targets[:,0], np.subtract(InputsTargets['PF'][0,:],  Targets[:,0]), bins=50,
                                             range=[[np.percentile(Targets[:,0],5),np.percentile(Targets[:,0],95)],
                                                    [np.percentile(np.subtract(InputsTargets['PF'][0,:],  Targets[:,0]),5),
                                                    np.percentile(np.subtract(InputsTargets['PF'][0,:],  Targets[:,0]),95)]])
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
    heatmap, xedges, yedges = np.histogram2d(  Targets[:,1], np.subtract(InputsTargets['PF'][1,:],  Targets[:,1]), bins=50,
                                             range=[[np.percentile(Targets[:,1],5),np.percentile(Targets[:,1],95)],
                                                    [np.percentile(map(np.subtract, InputsTargets['PF'][1,:],  Targets[:,1]),5),
                                                     np.percentile(map(np.subtract, InputsTargets['PF'][1,:],  Targets[:,1]),95)]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    HM= plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm())
    plt.colorbar(HM)
    plt.legend()
    plt.savefig("%sHM_PF-Tar_Tar_y.png"%(plotsD), bbox_inches="tight")

    plt.clf()
    plt.figure()
    #plt.suptitle('x-Korrektur Prediction-Target ')
    plt.xlabel("$ p_{T,x}^Z$")
    plt.ylabel("Prediction-Target")
    heatmap, xedges, yedges = np.histogram2d(  Targets[:,0], NN_Diff_x,  bins=50,
                                             range=[[np.percentile(Targets[:,0],5),np.percentile(Targets[:,0],95)],
                                                    [np.percentile(np.subtract(predictions[:,0],  Targets[:,0]),5),
                                                    np.percentile(np.subtract(predictions[:,0],  Targets[:,0]),95)]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #plt.clf()
    HM =plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm())
    plt.colorbar(HM)

    plt.legend()
    plt.savefig("%sHM_Delta_x_Tar_x.png"%(plotsD), bbox_inches="tight")
    plt.close()






    plt.clf()
    plt.figure()
    #plt.suptitle('x-Korrektur Prediction-Target ')
    plt.xlabel("$ \Delta p_{T,x}^Z$")
    plt.ylabel("$ \Delta p_{T,y}^Z$")
    heatmap, xedges, yedges = np.histogram2d(  NN_Diff_x, NN_Diff_y,  bins=50,
                                             range=[[np.percentile(NN_Diff_x,5),np.percentile(NN_Diff_x,95)],
                                                    [np.percentile(NN_Diff_y,5),np.percentile(NN_Diff_y,95)]])
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
    heatmap, xedges, yedges = np.histogram2d( Targets[:,1], NN_Diff_y, bins=50,
                                             range=[[np.percentile(Targets[:,1],5),np.percentile(Targets[:,1],95)],
                                                    [np.percentile(np.subtract(predictions[:,1], Targets[:,1]),5),
                                                     np.percentile(np.subtract(predictions[:,1], Targets[:,1]),95)]])
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
    heatmap, xedges, yedges = np.histogram2d(  Targets[:,1] , Targets[:,0], bins=50, range=[[-40,40],[-40, 40]])
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
    heatmap, xedges, yedges = np.histogram2d(  predictions[:,1] , predictions[:,0], bins=50,  range=[[-25,25],[-25, 25]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    HM= plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm())
    plt.colorbar(HM)
    plt.legend()
    plt.savefig("%sHM_predictions_Correlation.png"%(plotsD), bbox_inches="tight")
    plt.close()



    print('Summe prediction 0', np.sum(predictions[:,0]))
    print('Summe prediction 1', np.sum(predictions[:,1]))

    print("y-Bias: np.mean(predictions[:,1])-np.mean(Targets[:,1])", np.mean(predictions[:,1])-np.mean(Targets[:,1]))

    if NN_mode =='xyr' or NN_mode =='xyd' or NN_mode =='nr' or NN_mode =='xyd':
        plt.clf()
        plt.figure()
        #plt.suptitle('y-Korrektur: Prediction-Target ')
        plt.xlabel("$|\\vec{MET}|$ in GeV")
        plt.ylabel("Prediction-Target")
        heatmap, xedges, yedges = np.histogram2d(  map(np.subtract, predictions[:,2],  Targets[:,2]) , Targets[:,2], bins=50)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        #plt.clf()
        plt.imshow(heatmap.T, extent=extent, origin='lower')
        #plt.colorbar(heatmap.T)
        plt.legend()
        plt.savefig("%sHM_Pred-Tar_Tar_r.png"%(plotsD), bbox_inches="tight")



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
    print("Korrellationskoeffizient zwischen Targets x, y", np.corrcoef(Targets[:,0],Targets[:,1]))
    print("Korrellationskoeffizient zwischen Predictions x, y", np.corrcoef(predictions[:,0],predictions[:,1]))
    print("Korrellationskoeffizient zwischen Delta x, Delta y", np.corrcoef(predictions[:,0]-Targets[:,0],predictions[:,1]-Targets[:,1]))

    NN_Output.close()
    NN_Output_applied.close()

if __name__ == "__main__":
    outputDir = sys.argv[1]
    optim = str(sys.argv[2])
    loss_fct = str(sys.argv[3])
    NN_mode = sys.argv[4]
    plotsD = sys.argv[5]
    rootOutput = sys.argv[6]
    PhysicsProcess = sys.argv[7]
    rootInput = sys.argv[7]
    if PhysicsProcess == 'Tau':
        Target_Pt = 'genMet_Pt'
        Target_Phi = 'genMet_Phi'
        Outputs = loadData_woutGBRT(rootOutput, rootInput, Target_Pt, Target_Phi)
    else:
        Target_Pt = 'Boson_Pt'
        Target_Phi = 'Boson_Phi'
        Outputs = loadData(rootOutput)

    DFName = Outputs
    print(outputDir)
    plotTraining(outputDir, optim, loss_fct, NN_mode, plotsD, rootOutput, PhysicsProcess)
