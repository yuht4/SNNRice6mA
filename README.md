# SNNRice6mA: a deep learning method for predicting DNA N6-methyladenine sites in rice genome

DNA N6-methyladenine (6mA) is an important epigenetic modification, which is involved in many biology regulation processes. An accurate and reliable method for 6mA identification can help us gain a better insight into the regulatory mechanism of the modification. Although, many experimental techniques have been proposed to identify 6mA sites genome-wide, while these techniques are time consuming and laborious. Recently, several machine learning methods have been developed to identify 6mA sites genome-wide. However, there is room for the improvement on their performance for predicting 6mA sites in rice genome.


## Requirements
1. Python2
2. Anaconda 
3. keras
4. numpy, sklearn, pickle, pandas, matplotlib

## 6mA datasets preparation
Please prepare the 6mA dataset containing the positive and negative samples in **CSV** file format.
Each CSV file contains 2 columns : DNA sequence (41bp long), label (0 means negative, 1 means positive), separated by comma.

For the positive CSV file, the example is below:

![pos](https://github.com/yuht4/SNNRice6mA/blob/master/images/csvPos.PNG)


For the negative CSV file, the example is below:

![neg](https://github.com/yuht4/SNNRice6mA/blob/master/images/csv.PNG "neg")


For the convenience, we upload 2 6mA datasets for rice genome proposed by previous studies **(Chen et al., 2019; Lv et al., 2019)**.

The 6mA-rice-Chen dataset is in the folder **6mA_data/Rice_Chen**, while the  6mA-rice-Lv is in the folder **6mA_data/Rice_Lv**.

The 6mA-rice-Chen dataset contains 880 positive samples and 880 negative samples. The 6mA-rice-Lv dataset contains 154,000 positive samples and 154,000 negative samples.DNA sequences in both positive samples and negative samples are 41bp long. For each positive sequence, its center is the 6mA modification site. For each negative sequence, its center contains no 6mA modification site. 

## Train the model on the 6mA dataset
There are several model sciprts in our folder SNNRice6mA/, including SNN6mARice.py, SNN6mARice_feature.py and SNN6mARice_large.py.

- The SNN6mARice.py is the training code for the basic model based on the genomic sequence only.
- The SNN6mARice_feature.py is the training code for the model with feature-based sequence encoding strategy.
- The SNN6mARice_large.py is the training code for the model with more complex architecture.

We will show the method to use these scripts based on the two datasets, 6mA-rice-Chen and 6mA-rice-Lv, we upload.

### Model training and analyzing
Use the model script you like to train the model, one of the three scirpts, (SNN6mARice.py, SNN6mARice_feature.py and SNN6mARice_large.py).

For the SNN6mARice.py, the usage method is in below:
```
python SNN6mARice.py [OPTIONS]
[OPTIONS]:
	--positive     positive 6mA csv file
	--negative     negative 6mA csv file
	--output       output folder
```

For the SNN6mARice_feature.py, the usage method is in below:
```
python SNN6mARice_feature.py [OPTIONS]
[OPTIONS]:
	--positive     positive 6mA csv file
	--negative     negative 6mA csv file
	--output       output folder
```

For the SNN6mARice_large.py, the usage method is in below:
```
python SNN6mARice_large.py [OPTIONS]
[OPTIONS]:
	--positive     positive 6mA csv file
	--negative     negative 6mA csv file
	--output       output folder
```

### Model result
The model result in the output folder, contains the following things.

```
	performance.txt     statistics of metrics of training, validation and testing set
	model(*).h5         model files because of the cross-validation
	ROC.png             ROC cure of training, validation and testing set
```

For example:
![result](https://github.com/yuht4/SNNRice6mA/blob/master/images/S2.PNG "result")

### Model evaluation
Use the script, evaluate.py, in SNNRice6mA/ to evaluate the model

When using the training code to build the model on the dataset, there are multiple models because of cross-validation. These trained models are somewhat similar, and users can choose any of them.

```
python evaluate.py [OPTIONS]
[OPTIONS]:
	--h5File     the model h5 file 
	--csv       the input csv file 
```
 The script will output the accuracy of the model on the input dataset 

## Examples of using the program

### 1 training and anazlying on the 6mA-rice-Chen by SNN6mARice.py

Using the following commands:
```
python SNN6mARice.py --positive 6mA_data/Rice_Chen/Positive.txt --negative 6mA_data/Rice_Chen/Negative.txt --output result/
```

### 2 training and anazlying on the 6mA-rice-Chen by SNN6mARice_feature.py

Using the following commands:
```
python SNN6mARice_feature.py --positive 6mA_data/Rice_Chen/Positive.txt --negative 6mA_data/Rice_Chen/Negative.txt --output result/
```

### 3 training and anazlying on the 6mA-rice-Lv by SNN6mARice.py

Using the following commands:
```
python SNN6mARice.py --positive 6mA_data/Rice_Lv/pos.txt --negative 6mA_data/Rice_Lv/neg.txt --output result/
```
### 4 training and anazlying on the 6mA-rice-Lv by SNN6mARice_large.py

Using the following commands:
```
python SNN6mARice_large.py --positive 6mA_data/Rice_Lv/pos.txt --negative 6mA_data/Rice_Lv/neg.txt --output result/
```

## Reference
1. Chen, W., Lv, H., Nie, F., and Lin, H. (2019). i6mA-Pred: Identifying DNA N6-methyladenine sites in the rice genome. Bioinformatics.
2. Lv, H., Dao, F.-Y., Guan, Z.-X., Zhang, D., Tan, J.-X., Zhang, Y., Chen, W., and Lin, H. (2019). iDNA6mA-Rice: a computational tool for detecting N6-methyladenine sites in rice. Frontiers in Genetics DOI: 10.3389/fgene.2019.00793.
