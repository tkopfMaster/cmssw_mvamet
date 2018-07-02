#!/bin/bash
source /storage/b/tkopf/jdl_maerz/setup_maerz.sh
python -c 'import keras; print(keras.__version__)'
echo "trainingname eingeben"
#read trainingname
#trainingname='xyrTargets'
#trainingname=nrTargets
optimizer="Adam"
loss="mean_squared_error"
NN_mode="xy"
trainingname="Stefan_woutSumET_woutPP_${NN_mode}_${optimizer}_${loss}"
echo "$trainingname"
if [ -n "$trainingname" ]; then
    echo "$trainingname not empty"
else
		trainingname='xyrTargets'
    echo "$trainingname empty"
fi
inputFile=/storage/b/tkopf/mvamet/skim/out.root
GBRTFile=/storage/b/tkopf/mvamet/Gridoutput/data1.root
#cp $GBRTFile /storage/b/tkopf/mvamet/Gridoutput/rootfiles/data_${trainingname}.root
GBRTFile2=/storage/b/tkopf/mvamet/Gridoutput/rootfiles/data_${trainingname}.root
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
fi
files_di=$files_di$trainingname/
if [ ! -d "$plots_di/$trainingname/" ]; then
	mkdir $plots_di$trainingname/
fi
plots_di=$plots_di$trainingname/
if [ ! -d "trainings/$trainingname" ]; then
	mkdir trainings/$trainingname
	echo "trainings/$trainingname"
fi
if [ ! -d "trainings/$trainingname" ]; then
	mkdir trainings/$trainingname
	cd trainings/$trainingname/
	echo "files_di"
fi
#spaeter mal: config mit Art des Trainings festlegen
#python $src_di/prepareInput.py $inputFile $files_di $NN_mode $plots_di
#python $src_di/getNNModel.py $files_di $optimizer $loss $NN_mode $plots_di
#python $src_di/applyNN.py $inputFile $files_di $optimizer $loss $NN_mode
#python $src_di/plotTrainingclean.py $files_di $optimizer $loss $NN_mode $plots_di $GBRTFile2
#python $src_di/prepareOutput.py $GBRTFile2 $files_di $NN_mode $plots_di
#python $src_di/getPlotsInput.py $inputFile $plots_di
#python $src_di/getPlotsOutputclean.py $GBRTFile2 $files_di $plots_di
python $src_di/getResponse.py $GBRTFile2 $files_di $plots_di
cp -r $plots_di /usr/users/tkopf/www/METplots/
cp /usr/users/tkopf/www/index.php /usr/users/tkopf/www/METplots/$trainingname/
