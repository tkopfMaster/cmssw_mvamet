#!/bin/bash
source /storage/b/tkopf/jdl_maerz/setup_maerz.sh
echo "Trainingsname eingeben"
#read Trainingsname
inputFile=/storage/b/tkopf/mvamet/skim/out.root
GBRTFile=/storage/b/tkopf/mvamet/Gridoutput/data1.root

src_di=$PWD
files_di=/storage/b/tkopf/mvamet/files/
plots_di=/storage/b/tkopf/mvamet/plots/NN/
cd ..
if [ ! -d "trainings" ]; then
	mkdir trainings
fi
if [ ! -d "trainings/$trainingname" ]; then
	mkdir trainings/$trainingname
	echo "trainings/$trainingname"
fi
if [ ! -d "$files_di/$trainingname" ]; then
	mkdir trainings/$trainingname
	files_di=$files_di$trainingsname
	plots_di=$plots_di$trainingsname
	cd trainings/$trainingname/
	echo "files_di"
fi
#spaeter mal: config mit Art des Trainings festlegen
#python $src_di/prepareInput.py $inputFile $files_di
#python $src_di/getNNModel.py $files_di
#python $src_di/applyNN.py $inputFile $files_di
#python $src_di/prepareOutput.py $GBRTFile $files_di
#python $src_di/getPlotsInput.py $inputFile $plots_di
python $src_di/getPlotsOutput.py $GBRTFile $files_di $plots_di
