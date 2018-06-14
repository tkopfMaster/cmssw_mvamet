import h5py
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys
import numpy as np





def plotTraining(outputD, optim, loss_fct, NN_mode, plotsD):
    NN_Output_applied = h5py.File("%sNN_Output_applied_%s.h5"%(outputD,NN_mode), "r")
    predictions = NN_Output_applied["MET_Predictions"]
    Targets = NN_Output_applied["MET_GroundTruth"]
    print("Shape predictions", predictions)
    print("Shape Targets", Targets)
    
    NN_Output = h5py.File("%sNN_Output_%s.h5"%(outputD,NN_mode), "r")
    loss = NN_Output["loss"]
    val_loss = NN_Output["val_loss"]



    plt.clf()
    plt.figure()
    plt.suptitle('x-Korrektur Prediction-Target ')
    plt.xlabel("$p_{T,x}^Z$")
    plt.ylabel("Prediction-Target")
    heatmap, xedges, yedges = np.histogram2d(  map(np.subtract, predictions[:,0],  Targets[:,0]) , Targets[:,0], bins=50, range=[[np.percentile(Targets[:,0],5),np.percentile(Targets[:,0],95)],[np.percentile(map(np.subtract, predictions[:,0],  Targets[:,0]),5), np.percentile(map(np.subtract, predictions[:,0],  Targets[:,0]),95)]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    #plt.colorbar(heatmap.T)
    plt.legend()
    plt.savefig("%sHM_Pred-Tar_Tar_x.png"%(plotsD))

    plt.clf()
    plt.figure()
    plt.suptitle('y-Korrektur: Prediction-Target ')
    plt.xlabel("$p_{T,y}^Z$")
    plt.ylabel("Prediction-Target")
    heatmap, xedges, yedges = np.histogram2d(  map(np.subtract, predictions[:,1],  Targets[:,1]) , Targets[:,1], bins=50, range=[[np.percentile(Targets[:,1],5),np.percentile(Targets[:,1],95)],[np.percentile(map(np.subtract, predictions[:,1],  Targets[:,1]),5), np.percentile(map(np.subtract, predictions[:,1],  Targets[:,1]),95)]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    #plt.colorbar(heatmap.T)
    plt.legend()
    plt.savefig("%sHM_Pred-Tar_Tar_y.png"%(plotsD))

    plt.clf()
    plt.figure()
    plt.suptitle('Targets: x-y-correlation ')
    plt.ylabel("$p_{T,y}^Z$")
    plt.xlabel("$p_{T,x}^Z$")
    heatmap, xedges, yedges = np.histogram2d(  Targets[:,1] , Targets[:,0], bins=50, range=[[np.percentile(Targets[:,0],5),np.percentile(Targets[:,0],95)],[np.percentile( Targets[:,1],5), np.percentile(Targets[:,1],95)]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    #plt.colorbar(heatmap.T)
    plt.legend()
    plt.savefig("%sHM_Targets_x_y.png"%(plotsD))

    plt.clf()
    plt.figure()
    plt.suptitle('predictions: x-y-correlation ')
    plt.ylabel("$p_{T,y}^Z$")
    plt.xlabel("$p_{T,x}^Z$")
    heatmap, xedges, yedges = np.histogram2d(  predictions[:,1] , predictions[:,0], bins=50, range=[[np.percentile(predictions[:,0],5),np.percentile(predictions[:,0],95)],[np.percentile( predictions[:,1],5), np.percentile(predictions[:,1],95)]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    #plt.colorbar(heatmap.T)
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
