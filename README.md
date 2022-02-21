
#emotion model train

training gender and age model multi-label use keras

## environment
- keras =2.24 
- tensorflow=1.15
- cuda=10.0.1
- cudnn=7.6.5


## conda install

conda env create -f environment.yml --name env_name

# inference
`python age_gender_inf.py --image_dir  ***folder  --ga_model *****.pb`

- image_dir: test img folder path (none using cam 0)
- ga_model: ga model path
- margin:margin around detected face for age-gender estimation

# train

`python train_onecard.py --appa_dir ??? ` 
- appa_dir: train data dir
- output_dir:model output dir
- batch_size:batch_size(default 32)
- nb_epochs: training epochs
- lr :learning rate(default 0.001)
- opt :optimizer(adam or sgd)

# train step
1.將data資料夾內的train.zip解壓縮

2.python train_onecard.py

3.至checkpoints資料夾取得最好or最後的hdf5 model

4.將keras hdf5 model to tensorflow pb model
    python keras_to_tensorflow.py –input_model_file XXXX.hdf5 –output_model_file XXXX.pb 
    

# error
yaml 安裝錯誤時 請先將該行mark(#) ，重新create一次在自行安裝失敗套件

dlib 錯誤請去下載
https://pypi.org/simple/dlib/ tar.gz解壓縮 
pip install cmake
python setup.py install


