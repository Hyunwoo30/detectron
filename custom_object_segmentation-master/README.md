## Semantic Segmentation with Transfer Learning 

In this project we trained custom detaset using pretrained weights.  

### Dataset annotation 
I have used coco-annotator tool for labeling the custom data. We have stored label details in json file and put them in training and validation folders

3 samples of the dataset


<p float="left">
  <img src="./readme_source/22-05.jpg" width="300" />
  <img src="./readme_source/22-04.jpg" width="300" /> 
  <img src="./readme_source/22-06.jpg" width="300" />
</p>



labelled image 

<p float="left">
  <img src="./readme_source/37-20.jpg" width="300" />

</p>
The model was fine tuned for 100 epochs. 

The segmentation output of the model on a sample image are shown below.


<p float="left">
  <img src="./readme_source/33-22.jpg" width="500" />
  <img src="./readme_source/33-23.jpg" width="500" /> 
  <img src="./readme_source/33-24.jpg" width="500" />
</p>

### Installing dependencies

#### first open your terminal and clone the repository on your specified folder of local machine. 
the project was tested in linux ubuntu OS
```
git clone https://github.com/jakhon37/custom_object_segmentation.git
```

#### Using conda create new enviorenment with python version==3.7
```
conda env create -n c python==3.7

conda activate new_custom_env

pip install -r requirements.txt
```




### Usage of the module

#### training 
 
 run below command on your terminal for training
```
python train.py

```
for custom dataset  open train.py file and specify your data path 

#### evaluation 

for testing the trained model, run below command on your terminal 
```
python eval.py
```
to test differet images, open eval.py file and specify your data path
you can chack test result  in "output" folder. 


#### if you want to test without training, doenload trained weight for this dataset and place it in spesified folder 
link: <a href="https://drive.google.com/file/d/1_MSHem9fZeUW4qPg9F3WhfJHNrwafDYb/view?usp=sharing">model_final.pth</a> 


folder to put the weight: 
```
    --output
    ----segmentation
    ------model_final.pth
```






#### dataset folder strucutre 


    ```
    --dataset
    ----e_motor
    ------train
    ---------Image1
    ---------ImageN
    ---------train_annotation.json
    ------validation
    ---------Image1
    ---------ImageN
    ---------val_annotation.json
    ```

#### ******************
