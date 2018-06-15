import h5py
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import LogNorm


LegendTitle = '$\mathrm{Summer\ 17\ campaign}$' '\n'  '$\mathrm{Z \  \\rightarrow \ \mu \mu}$'
colors_InOut = cm.brg(np.linspace(0, 1, 8))

def plotTraining(outputD, optim, loss_fct, NN_mode, plotsD):
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
        #predictions=InputsTargets['predictions'].reshape(-1,1)
        #InputsTargets = np.transpose(InputsTargets)
        fig=plt.figure(figsize=(10,6))
        fig.patch.set_facecolor('white')
        ax = plt.subplot(111)
        nbinsHist = 150
        plt.hist(InputsTargets['PF'][0,:], bins=nbinsHist, range=[-50, 50], label='PF, mean=%.2f'%np.mean(InputsTargets['PF'][:,0]), histtype='step', ec=colors_InOut[0])
        plt.hist(InputsTargets['Track'][0,:], bins=nbinsHist, range=[-50, 50], label='Track, mean=%.2f'%np.mean(InputsTargets['Track'][:,0]), histtype='step', ec=colors_InOut[0])
        plt.hist(InputsTargets['NoPU'][0,:], bins=nbinsHist, range=[-50, 50], label='NoPU, mean=%.2f'%np.mean(InputsTargets['NoPU'][:,0]), histtype='step', ec=colors_InOut[2])
        plt.hist(InputsTargets['PUCorrected'][0,:], bins=nbinsHist, range=[-50, 50], label='PUCorrected, mean=%.2f'%np.mean(InputsTargets['PUCorrected'][:,0]), histtype='step', ec=colors_InOut[3])
        plt.hist(InputsTargets['PU'][0,:], bins=nbinsHist, range=[-50, 50], label='PU, mean=%.2f'%np.mean(InputsTargets['PU'][:,0]), histtype='step', ec=colors_InOut[4])
        plt.hist(InputsTargets['Puppi'][0,:], bins=nbinsHist, range=[-50, 50], label='Puppi, mean=%.2f'%np.mean(InputsTargets['Puppi'][:,0]), histtype='step', ec=colors_InOut[5])
        plt.hist(Targets[:,0], bins=nbinsHist, range=[-50, 50], label='Target, mean=%.2f'%np.mean(Targets[:,0]), histtype='step', ec=colors_InOut[6])
        plt.hist(predictions[:,0], bins=nbinsHist, range=[-50, 50], label='Prediction, mean=%.2f'%np.mean(predictions[:,0]), histtype='step', ec=colors_InOut[7])


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
        plt.savefig("%sHist_Input_y_Hist.png"%(plotsD))

        fig=plt.figure(figsize=(10,6))
        fig.patch.set_facecolor('white')
        ax = plt.subplot(111)
        nbinsHist = 150
        plt.hist(InputsTargets['PF'][1,:], bins=nbinsHist, range=[-50, 50], label='PF, mean=%.2f'%np.mean(InputsTargets['PF'][:,1]), histtype='step', ec=colors_InOut[0])
        plt.hist(InputsTargets['Track'][1,:], bins=nbinsHist, range=[-50, 50], label='Track, mean=%.2f'%np.mean(InputsTargets['Track'][:,1]), histtype='step', ec=colors_InOut[0])
        plt.hist(InputsTargets['NoPU'][1,:], bins=nbinsHist, range=[-50, 50], label='NoPU, mean=%.2f'%np.mean(InputsTargets['NoPU'][:,1]), histtype='step', ec=colors_InOut[2])
        plt.hist(InputsTargets['PUCorrected'][1,:], bins=nbinsHist, range=[-50, 50], label='PUCorrected, mean=%.2f'%np.mean(InputsTargets['PUCorrected'][:,1]), histtype='step', ec=colors_InOut[3])
        plt.hist(InputsTargets['PU'][1,:], bins=nbinsHist, range=[-50, 50], label='PU, mean=%.2f'%np.mean(InputsTargets['PU'][:,1]), histtype='step', ec=colors_InOut[4])
        plt.hist(InputsTargets['Puppi'][1,:], bins=nbinsHist, range=[-50, 50], label='Puppi, mean=%.2f'%np.mean(InputsTargets['Puppi'][:,1]), histtype='step', ec=colors_InOut[5])
        plt.hist(Targets[:,1], bins=nbinsHist, range=[-50, 50], label='Target, mean=%.2f'%np.mean(Targets[:,1]), histtype='step', ec=colors_InOut[6])
        plt.hist(predictions[:,1], bins=nbinsHist, range=[-50, 50], label='Prediction, mean=%.2f'%np.mean(predictions[:,1]), histtype='step', ec=colors_InOut[7])


        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        handles, labels = ax.get_legend_handles_labels()
        #handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

        plt.ylabel('Counts')
        plt.xlabel('$ p_{T,y}$ in GeV')
        #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_{T}^Z} \\right) $ in GeV')
        plt.title(' Histogram $ p_{T,y}$')
        #plt.text('$p_T$ range restriction')

        ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
        plt.grid()
        #plt.ylim(ylimResMVAMin, ylimResMax)
        plt.savefig("%sHist_Input_y_Hist.png"%(plotsD))



        fig=plt.figure(figsize=(10,6))
        fig.patch.set_facecolor('white')
        ax = plt.subplot(111)
        nbinsHist = 150
        plt.hist(np.subtract(InputsTargets['PF'][0,:], Targets[:,0]), bins=nbinsHist, range=[-50, 50], label='PF, mean=%.2f'%np.mean(np.subtract(InputsTargets['PF'][0,:], Targets[:,0])), histtype='step', ec=colors_InOut[0])
        plt.hist(np.subtract(InputsTargets['Track'][0,:], Targets[:,0]), bins=nbinsHist, range=[-50, 50], label='Track, mean=%.2f'%np.mean(np.subtract(InputsTargets['Track'][0,:], Targets[:,0])), histtype='step', ec=colors_InOut[0])
        plt.hist(np.subtract(InputsTargets['NoPU'][0,:], Targets[:,0]), bins=nbinsHist, range=[-50, 50], label='NoPU, mean=%.2f'%np.mean(np.subtract(InputsTargets['NoPU'][0,:], Targets[:,0])), histtype='step', ec=colors_InOut[2])
        plt.hist(np.subtract(InputsTargets['PUCorrected'][0,:], Targets[:,0]), bins=nbinsHist, range=[-50, 50], label='PUCorrected, mean=%.2f'%np.mean(np.subtract(InputsTargets['PUCorrected'][0,:], Targets[:,0])), histtype='step', ec=colors_InOut[3])
        plt.hist(np.subtract(InputsTargets['PU'][0,:], Targets[:,0]), bins=nbinsHist, range=[-50, 50], label='PU, mean=%.2f'%np.mean(np.subtract(InputsTargets['PU'][0,:], Targets[:,0])), histtype='step', ec=colors_InOut[4])
        plt.hist(np.subtract(InputsTargets['Puppi'][0,:], Targets[:,0]), bins=nbinsHist, range=[-50, 50], label='Puppi, mean=%.2f'%np.mean(np.subtract(InputsTargets['Puppi'][0,:], Targets[:,0])), histtype='step', ec=colors_InOut[5])
        plt.hist(NN_Diff_x, bins=nbinsHist, range=[-50, 50], label='Prediction, mean=%.2f'%np.mean(np.subtract(predictions[:,0], Targets[:,0])), histtype='step', ec=colors_InOut[7])


        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        handles, labels = ax.get_legend_handles_labels()
        #handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

        plt.ylabel('Counts')
        plt.xlabel('$\\Delta p_{T,x}$ in GeV')
        #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_{T}^Z} \\right) $ in GeV')
        plt.title(' Histogram Deviation to Target  $\\Delta p_{T,x}$')
        #plt.text('$p_T$ range restriction')

        ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
        plt.grid()
        #plt.ylim(ylimResMVAMin, ylimResMax)
        plt.savefig("%sHist_Input_Dev_x_Hist.png"%(plotsD))

        fig=plt.figure(figsize=(10,6))
        fig.patch.set_facecolor('white')
        ax = plt.subplot(111)
        nbinsHist = 150
        plt.hist(np.subtract(InputsTargets['PF'][1,:], Targets[:,1]), bins=nbinsHist, range=[-50, 50], label='PF, mean=%.2f'%np.mean(np.subtract(InputsTargets['PF'][1,:], Targets[:,1])), histtype='step', ec=colors_InOut[0])
        plt.hist(np.subtract(InputsTargets['Track'][1,:], Targets[:,1]), bins=nbinsHist, range=[-50, 50], label='Track, mean=%.2f'%np.mean(np.subtract(InputsTargets['Track'][1,:], Targets[:,1])), histtype='step', ec=colors_InOut[0])
        plt.hist(np.subtract(InputsTargets['NoPU'][1,:], Targets[:,1]), bins=nbinsHist, range=[-50, 50], label='NoPU, mean=%.2f'%np.mean(np.subtract(InputsTargets['NoPU'][1,:], Targets[:,1])), histtype='step', ec=colors_InOut[2])
        plt.hist(np.subtract(InputsTargets['PUCorrected'][1,:], Targets[:,1]), bins=nbinsHist, range=[-50, 50], label='PUCorrected, mean=%.2f'%np.mean(np.subtract(InputsTargets['PUCorrected'][1,:], Targets[:,1])), histtype='step', ec=colors_InOut[3])
        plt.hist(np.subtract(InputsTargets['PU'][1,:], Targets[:,1]), bins=nbinsHist, range=[-50, 50], label='PU, mean=%.2f'%np.mean(np.subtract(InputsTargets['PU'][1,:], Targets[:,1])), histtype='step', ec=colors_InOut[4])
        plt.hist(np.subtract(InputsTargets['Puppi'][1,:], Targets[:,1]), bins=nbinsHist, range=[-50, 50], label='Puppi, mean=%.2f'%np.mean(np.subtract(InputsTargets['Puppi'][1,:], Targets[:,1])), histtype='step', ec=colors_InOut[5])
        plt.hist(NN_Diff_y, bins=nbinsHist, range=[-50, 50], label='Prediction, mean=%.2f'%np.mean(np.subtract(predictions[:,1], Targets[:,1])), histtype='step', ec=colors_InOut[7])


        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        handles, labels = ax.get_legend_handles_labels()
        #handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

        plt.ylabel('Counts')
        plt.xlabel('$\Delta p_{T,y}$ in GeV')
        #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_{T}^Z} \\right) $ in GeV')
        plt.title(' Histogram Deviation to Target  $\Delta p_{T,y}$')
        #plt.text('$p_T$ range restriction')

        ax.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small', title=LegendTitle, numpoints=1	)
        plt.grid()
        #plt.ylim(ylimResMVAMin, ylimResMax)
        plt.savefig("%sHist_Input_Dev_y_Hist.png"%(plotsD))


    plt.clf()
    plt.figure()
    plt.suptitle('x-Korrektur PF-Target ')
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
    plt.savefig("%sHM_PF-Tar_Tar_x.png"%(plotsD))

    plt.clf()
    plt.figure()
    plt.suptitle('y-Korrektur: PF-Target ')
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
    plt.savefig("%sHM_PF-Tar_Tar_y.png"%(plotsD))

    plt.clf()
    plt.figure()
    plt.suptitle('x-Korrektur Prediction-Target ')
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
    plt.savefig("%sHM_Pred-Tar_Tar_x.png"%(plotsD))

    plt.clf()
    plt.figure()
    plt.suptitle('y-Korrektur: Prediction-Target ')
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
    plt.savefig("%sHM_Pred-Tar_Tar_y.png"%(plotsD))

    plt.clf()
    plt.figure()
    plt.suptitle('Targets: x-y-correlation ')
    plt.ylabel("$p_{T,y}^Z$")
    plt.xlabel("$p_{T,x}^Z$")
    heatmap, xedges, yedges = np.histogram2d(  Targets[:,1] , Targets[:,0], bins=50, range=[[np.percentile(Targets[:,0],5),np.percentile(Targets[:,0],95)],[np.percentile( Targets[:,1],5), np.percentile(Targets[:,1],95)]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    HM = plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm() )
    plt.colorbar(HM)
    plt.legend()
    plt.savefig("%sHM_Targets_x_y.png"%(plotsD))

    plt.clf()
    plt.figure()
    plt.suptitle('predictions: x-y-correlation ')
    plt.ylabel("$p_{T,y}^Z$")
    plt.xlabel("$p_{T,x}^Z$")
    heatmap, xedges, yedges = np.histogram2d(  predictions[:,1] , predictions[:,0], bins=50, range=[[np.percentile(predictions[:,0],5),np.percentile(predictions[:,0],95)],[np.percentile( predictions[:,1],5), np.percentile(predictions[:,1],95)]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    HM= plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm())
    plt.colorbar(HM)
    plt.legend()
    plt.savefig("%sHM_predictions_x_y.png"%(plotsD))

    plt.clf()
    plt.figure()
    plt.suptitle('x-Korrektur Prediction-Target ')
    plt.xlabel("$p_{T,x}^Z$")
    plt.hist(map(np.subtract, predictions[:,0],  Targets[:,0]), bins=50, range=[np.percentile(map(np.subtract, predictions[:,0],  Targets[:,0]),5), np.percentile(map(np.subtract, predictions[:,0],  Targets[:,0]),95)])
    plt.legend()
    plt.savefig("%sHist_Pred-Tar_Tar_x.png"%(plotsD))

    plt.clf()
    plt.figure()
    plt.suptitle('y-Korrektur Prediction-Target ')
    plt.xlabel("$p_{T,y}^Z$")
    plt.hist(map(np.subtract, predictions[:,1],  Targets[:,1]), bins=50, range=[np.percentile(map(np.subtract, predictions[:,1],  Targets[:,1]),5), np.percentile(map(np.subtract, predictions[:,1],  Targets[:,1]),95)])
    plt.legend()
    plt.savefig("%sHist_Pred-Tar_Tar_y.png"%(plotsD))



    plt.clf()
    plt.figure()
    plt.suptitle('y-Korrektur Prediction vs. Target ')
    plt.xlabel("$p_{T,y}^Z$")
    plt.hist(predictions[:,1], bins=50, range=[np.percentile(predictions[:,1],5), np.percentile( predictions[:,1],95)], histtype='step' )
    plt.hist(Targets[:,1], bins=50, range=[np.percentile(Targets[:,1],5), np.percentile( Targets[:,1],95)], histtype='step' )
    plt.legend()
    plt.savefig("%sHist_Pred_Tar_y.png"%(plotsD))



    plt.clf()
    plt.figure()
    plt.suptitle('y-Korrektur Prediction vs. Target ')
    plt.xlabel("$p_{T,x}^Z$")
    plt.hist(predictions[:,0], bins=50, range=[np.percentile(predictions[:,0],5), np.percentile( predictions[:,0],95)], histtype='step' , label='prediction')
    plt.hist(Targets[:,0], bins=50, range=[np.percentile(Targets[:,0],5), np.percentile( Targets[:,0],95)], histtype='step',  label='Target')
    plt.legend()
    plt.savefig("%sHist_Pred_Tar_x.png"%(plotsD))

    print('Summe prediction 0', np.sum(predictions[:,0]))
    print('Summe prediction 1', np.sum(predictions[:,1]))

    print("y-Bias: np.mean(predictions[:,1])-np.mean(Targets[:,1])", np.mean(predictions[:,1])-np.mean(Targets[:,1]))

    if NN_mode =='xyr' or NN_mode =='xyd' or NN_mode =='nr' or NN_mode =='xyd':
        plt.clf()
        plt.figure()
        plt.suptitle('y-Korrektur: Prediction-Target ')
        plt.xlabel("${p_T^Z}$")
        plt.ylabel("Prediction-Target")
        heatmap, xedges, yedges = np.histogram2d(  map(np.subtract, predictions[:,2],  Targets[:,2]) , Targets[:,2], bins=50)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        #plt.clf()
        plt.imshow(heatmap.T, extent=extent, origin='lower')
        #plt.colorbar(heatmap.T)
        plt.legend()
        plt.savefig("%sHM_Pred-Tar_Tar_r.png"%(plotsD))



    epochs = range(1, 1+len(loss))
    plt.figure()
    plt.xlabel("Epochs all")
    plt.ylabel("Loss")
    plt.ylabel("Val_loss")
    plt.plot(epochs, loss, 'g^', label="Loss")
    plt.plot(epochs, val_loss, 'r^', label="Validation Loss")
    plt.legend(["Loss","Validation Loss"], loc='upper left')
    plt.yscale('log')
    plt.savefig("%sLoss.png"%(plotsD))

    epochs = range(1, 50)
    plt.figure()
    plt.xlabel("Epochs all")
    plt.ylabel("Loss")
    plt.ylabel("Val_loss")
    plt.plot(epochs, loss[0:49], 'g^', label="Loss")
    plt.plot(epochs, val_loss[0:49], 'g-', label="Validation Loss")
    plt.legend(["Loss","Validation Loss"], loc='upper left')
    plt.yscale('log')
    plt.savefig("%sLoss_CR.png"%(plotsD))

    NN_Output.close()
    NN_Output_applied.close()

if __name__ == "__main__":
    outputDir = sys.argv[1]
    optim = str(sys.argv[2])
    loss_fct = str(sys.argv[3])
    NN_mode = sys.argv[4]
    plotsD = sys.argv[5]
    print(outputDir)
    plotTraining(outputDir, optim, loss_fct, NN_mode, plotsD)
