#!/bin/bash
source /storage/b/tkopf/jdl_maerz/setup_maerz.sh
echo "trainingname eingeben"
#read trainingname
#trainingname='xyrTargets'
PhysicsProcess="Mu"
optimizer="Adam"
loss="relResponseAsypTPVRange"
NN_mode="xy"
trainingname="TF_flatnet_9pTbins_5PVbins_booleanmask_relu_80ES_Taylor_CrossVal_woutReweight_uniformBatchpTtrainval_wSumEt_woutScale_woutVertexReweight_2000Batch_100000GS_20_200_4HL_${PhysicsProcess}_${NN_mode}_${optimizer}_${loss}_003_Part4"
echo "$trainingname"
if [ -n "$trainingname" ]; then
    echo "$trainingname not empty"
fi
#inputFile=/storage/b/tkopf/mvamet/skim/out.root
#inputFile=/storage/b/tkopf/mvamet/skim/Tau.root
#inputFile=/storage/b/tkopf/mvamet/Gridoutput/data1.root
trainingsFile=/storage/b/tkopf/mvamet/Gridoutput/data1.root
#applyFile=/storage/b/tkopf/mvamet/skim/Tau.root
applyFile=/storage/b/tkopf/mvamet/Gridoutput/data1.root

echo "GBRTFile2 $GBRTFile2"
src_di=$PWD
files_di=/storage/b/tkopf/mvamet/files/
plots_di=/storage/b/tkopf/mvamet/plots/NN/
cd ..
if [ ! -d "trainings" ]; then
	mkdir trainings
fi
if [ ! -d "$files_di$trainingname/" ]; then
	mkdir $files_di$trainingname/
  mkdir $files_di$trainingname/CV/
fi
files_di=$files_di$trainingname/
if [ ! -d "$plots_di/$trainingname/" ]; then
	mkdir $plots_di$trainingname/
  mkdir $plots_di$trainingname/derivates/
  mkdir $plots_di$trainingname/METPOG/
fi
plots_di=$plots_di$trainingname/
if [ ! -d "trainings/$trainingname" ]; then
	mkdir trainings/$trainingname
	echo "trainings/$trainingname"
fi
if [ ! -d "trainings/$trainingname" ]; then
	mkdir trainings/$trainingnamev
	cd trainings/$trainingname/
	echo "files_di"
fi
#spaeter mal: config mit Art des Trainings festlegen
python $src_di/prepareInput_wSumEt.py $trainingsFile $files_di $NN_mode $plots_di $PhysicsProcess $applyFile
#python $src_di/gaussian_1Training_wReweight2.py $files_di $optimizer $loss $NN_mode $plots_di
#python $src_di/gaussian_1Training_wReweight_CV.py $files_di $optimizer $loss $NN_mode $plots_di
##python $src_di/1training_BU1508.py $files_di $optimizer $loss $NN_mode $plots_di
#python $src_di/applyTFmodel.py $applyFile $files_di $optimizer $loss $NN_mode

#python $src_di/prepareOutput_woutScale.py $applyFile $files_di $NN_mode $plots_di $PhysicsProcess
#python $src_di/MiniplotTraining.py $files_di $optimizer $loss $NN_mode $plots_di $PhysicsProcess $applyFile
#python $src_di/plotTrainingclean.py $files_di $optimizer $loss $NN_mode $plots_di $PhysicsProcess $applyFile
#python $src_di/getPlotsOutputclean.py $applyFile $files_di $plots_di $PhysicsProcess $applyFile $NN_mode
#python $src_di/getResponse.py $applyFile $files_di $plots_di $PhysicsProcess $NN_mode
#python $src_di/PlotsMETPog.py $applyFile $files_di $plots_di $PhysicsProcess $NN_mode

cp $src_di/*.py $plots_di
cp $src_di/*Training.sh $plots_di
cp -r $plots_di /usr/users/tkopf/www/METplots/
cp /usr/users/tkopf/www/index.php /usr/users/tkopf/www/METplots/$trainingname/
cp /usr/users/tkopf/www/index.php /usr/users/tkopf/www/METplots/$trainingname/derivates/
cp /usr/users/tkopf/www/index.php /usr/users/tkopf/www/METplots/$trainingname/METPOG/
#python $src_di/getNNModel.py $files_di $optimizer $loss $NN_mode $plots_di
#python $src_di/getPlotsInput.py $inputFile $plots_di $PhysicsProcess
#python $src_di/Test_TF.py
