# SCAG: a 3D branch angle extraction method

Songyin Zhang<sup>1</sup>, Yinmeng Song<sup>1,2</sup>, Ran Ou<sup>1,2</sup>, Yiqiang Liu<sup>1,3</sup>, Shaochen Li<sup>1</sup>, Yanjun Su<sup>4</sup>, Jiang Dong<sup>1,2,3</sup>, Yanfeng Ding<sup>1,2,3</sup>, Junyi Gai<sup>1,2</sup>, Jin Wu<sup>5</sup>, Jiaoping Zhang<sup>1,2</sup>, and Shichao Jin<sup>1,2,3*</sup>

<sup>1</sup>Crop Phenomics Research Centre, Academy for Advanced Interdisciplinary Studies, Collaborative Innovation Centre for Modern Crop Production co-sponsored by Province and Ministry, College of Agriculture, Nanjing Agricultural University, Nanjing 210095, China<br>

<sup>2</sup>State Key Laboratory of Crop Genetics and Germplasm Enhancement, National Center for Soybean Improvement, Key Laboratory for Biology and Genetic Improvement of Soybean (General, Ministry of Agriculture), Nanjing Agricultural University, Nanjing 210095, China<br>

<sup>3</sup>Sanya Research Institute of Nanjing Agriculture University, Sanya 572024, China<br>

<sup>4</sup>State Key Laboratory of Vegetation and Environmental Change, Institute of Botany, Chinese Academy of Sciences, Beijing 100093, China<br>

<sup>5</sup>Division for Ecology and Biodiversity, School of Biological Sciences, The University of Hong Kong, Pokfulam Road, Hong Kong, China<br>

\* Corresponding author: [jinshichao1993@](mailto:jinshichao1993@)gmail.com; [jschaon@](mailto:jschaon@)njau.edu.cn

## Overview

In this release, We uploaded the <font color="#000066">**code**</font> folder, which contains the python code for **SVM**, **DB** and **SCAG** methods, and wrote detailed comments in the code. Each code uses data from the <font color="#000066">**dataset**</font>  folder, and finally outputs the calculated angle results and displays the angle verification scatter plot in the <font color="#000066">**results**</font> folder. The Dataset folder contains the Soybean3D dataset (**Soybean3D**) involved in the article, a dataset of 152 varieties divided into three groups (**3_groups**), maize and tomato datasets (**Pheno4D**), 50 samples for SVM training (**SVM_Train**), and the manual measurement ground truth of 152 varieties angles (**Simples_GT.xlsx**).


## Install Python, Anaconda and Libraries
If you wish to run the SCAG method, you will need to set up Python on your system. 

1. Install Python releases:
   
   •	Read the beginner’s guide to Python if you are new to the language: 
   https://wiki.python.org/moin/BeginnersGuide
   
   •	For Windows users, Python 3 release can be downloaded via: 
   https://www.python.org/downloads/windows/
   
2. Install Anaconda Python distribution:
   
   •	Read the install instruction using the URL: https://docs.continuum.io/anaconda/install
   
   •	For Windows users, a detailed step-by-step installation guide can be found via: 
   https://docs.continuum.io/anaconda/install/windows 
   
   •	An Anaconda Graphical installer can be found via: 
   https://www.continuum.io/downloads

   •	We recommend users install the latest Anaconda Python distribution

3. Install packages:

   •  The SCAG method uses a number of 3rd-party libraries that you may need to add to your conda environment.
   These include, but are not limited to:
   
       Open3d=0.11.2
       Pandas=0.25.1
       Numpy=1.21.5
       Scipy=1.3.1
       Matplotlib=3.1.1
       Seaborn=0.9.0
       Scikit-leatn=0.21.3
       openpyxl=3.1.2
   
## Running <font color="#00dd00">The_SCAG_method.py</font>/<font color="#dd0000">The_DB_method.py</font>/<font color="#0000dd">The_SVM_method.py</font>

After you have configured the above environment, you can directly run **The_SCAG_method.py**. The input dataset and parameter tuning method of DB and SVM are similar to SCAG. The input dataset path is adjusted by **file_pathname**. The calculation results are saved in the results folder.

