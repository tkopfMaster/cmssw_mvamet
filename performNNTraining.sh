#!/bin/bash
source /storage/b/tkopf/jdl_maerz/setup_maerz.sh
echo "Trainingsname eingeben"
read Trainingsname
inputFile=/storage/b/tkopf/mvamet/skim/out.root


src_di=$PWD
files_di=/storage/b/tkopf/mvamet/files/
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
	cd trainings/$trainingname/
	echo "files_di"
fi
#spaeter mal: config mit Art des Trainings festlegen
python $src_di/prepareInput.py $inputFile $files_di
#getNNModel.py $files_di
#applyNN.py $files_di
#getPlots.py $files_di
