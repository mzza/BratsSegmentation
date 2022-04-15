echo "start unzip"
unzip /data/MICCAI_BraTS_2018_Data_Training.zip -d /input/Brats2018Train
unzip /data/MICCAI_BraTS_2018_Data_Validation.zip -d /input/Brats2018Val
echo "unzip finished"

mkdir /input/Brats2018Base
mkdir /input/Brats2018Base/PreProcess
mkdir /input/Brats2018Base/Raw
mkdir /input/Brats2018Base/Brats2018TrainResults


mkdir /input/Brats2018InputRoot
mkdir /data/Brats2018OutoutRoot
mkdir  /data/Brats2018OutoutRoot/BaseLine
mkdir  /data/Brats2018OutoutRoot/checkpoint



echo "env creating"
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
echo "env finished"

