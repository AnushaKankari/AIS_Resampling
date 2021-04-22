# **ARTIFICIAL IMMUNE SYSTEMS BASED RESAMPLING FOR DEALING WITH CLASS IMBALANCE**

## **1. Project Goal:**
Many real-world datasets suffer from the class imbalance problem.Frequently, either oversampling or undersampling strategies are applied to balance the dataset. From the past few decades, Artificial Immune Systems (AIS) which are defined as the intelligent computational algorithms inspired by the biological immune systems have gained a lot of attention from reserachers. They aimed to develop various immune-based algorithms to address the complex real-world problems for a wide variety of application domains. Namely, there has been some research works where AIS are applied to solve specific problems that suffer from the class imbalance problem. However, these methods were applied as the learning algorithm to solve the tasks and were not used as a resampling method. Only one work exists that uses some techniques of AIS to address the class imbalance problem [8].

The goal of this project is to explore the use of AIS as a data resampling strategy to tackle the class imbalance problem. In this study, we will test, evaluate and compare different approaches for integrating AIS techniques as resampling methods to address the class imbalance problem. Our experiments will be carried out using a set of the public available imbalanced data sets from Health and Fraud detection fields.

## **2. Project Implementation Details:**
**Programming Language:** Python
**Tools:** Jupyter Notebook
**Cloud-based Platforms:** Kaggle and Google colab
**Downloaded Datasets from:** Kaggle, UCI repository and Keel repository.

## **3. Introduction to Datasets:**
In our study, we considered a set of imbalanced data sets related to the Health and Fraud detection fields that are listed in below table.
![image](https://user-images.githubusercontent.com/58121387/115747984-c0ffa000-a363-11eb-991b-53b837382f12.png)

## **4. Exploratory Data Analysis:**
To implement any solution to a problem, the best initial approach is to analyze, explore, and understand the dataset. This approach is known to be the Exploratory Data Analysis (EDA). This considered to be a crucial step before data modeling in the Machine learning field. EDA supports various techniques that help to visually plot the summarized key characteristics of a dataset within less time. EDA plotting consists of histograms, box plots, scattering plots, and much more. In our study, we considered the below techniques or plotting of EDA to understand each dataset in a detailed manner.
.
1.	Checking the datatypes of features 
2.	Missing values using bar charts
3.	Distribution of the target field using bar charts
4.	Density plots
5.	Outlier Detection using box plots
6.	Heatmap correlation matrix
7.	2D and 3D scatter plots
8.	Feature Importance

The results of the above techniques applied to each dataset in a deeper manner is available in "Exploratory Data Analysis" folder.


## **5. Resampling and Modeling:**
In the Exploratory Data Analysis (EDA) section, a large variety of datasets with an imbalance ratio (IR) ranging between 0.0017 to 0.76 are chosen and various EDA techniques are applied. The visual outcomes of all the EDA techniques helped us gain in-depth insights into each dataset within a short period of time. The imbalanced class distribution of these datasets would have a severe impact on predictive modeling outcomes. It is very important to handle the class imbalance before implementing any predictive data model. In Machine learning, one best approach is to handle the class imbalance by applying suitable resampling strategies before data modeling. As part of this section, various over-sampling and under-sampling algorithms, data preprocessing techniques are applied to see the performance of the data models. All the techniques used are discussed in the list below. 
### **5.1 Performance Evaluation**

**K-fold Cross Validation (CV):** In general a dataset is split into 70% train and 30% test datasets using the train_test_split() function from the sklearn library. Subsequently, the model is trained using the training set and tested using the test dataset. The model performance is evaluated based on various metrics such as accuracy, recall, precision, etc. However, this method is not reliable because the results obtained in one test set can be very different from the results in other test sets. A solution to this problem is provided by the K-fold Cross Validation (CV) where the dataset is divided into folds and each fold at some point is used as a test set. In this study, the value of K is considered to be 5 meaning that each dataset is divided into 5 disjoint folds. In the first iteration, the first fold is used as a test set and the remaining as train sets. In the second iteration, the second fold is used as a test set and the remaining as train sets. This process is continued until each one of the 5 folds is used as a test set. The approach of this method ensures that the model is trained on different scenarios and allows us to obtain a better estimate of the models’ errors.

**Data Modeling:** Algorithms from ensemble classifiers are considered in our study. In ensemble learning, the model combines individual classifier decisions and accordingly makes predictions. Bagging and Boosting are two popular ensemble methods. In bagging, a set of individual classifiers are trained in a parallel manner. Whereas in boosting, a set of individual models are trained in a successive manner. In this sequential manner, the algorithm tries to convert the weak learners from the previous model into strong learners and tends to reduce bias and variance. We selected tree-based ensemble models for our experiments.

a.	Random Forest classifier

b.	Ada Boosting classifier

c.	Light Gradient Boosting classifier
     
**Evaluation metrics:**

The performance of the model is evaluated using various metrics. In our study, since we considered imbalance datasets we selected a set of suitable metrics for this problem.  Geometric mean, precision, recall, AUC, and f-score are considered as evaluation metrics. We also report the accuracy results only for reference. The geometric mean is the square root of the product of the sensitivity and specificity and measures the balance between classification performances on both the majority and minority classes. 

###**5.2 Data Preprocessing and Resampling**

**Data preprocessing:**

a.	Missing values

b.	Standardization

c.	One-Hot encoding

**Oversampling strategies:**

These strategies increase the minority class instances in the original dataset.

a.SMOTE

b.ADASYN

**Under sampling strategies:**

These strategies eliminate the majority class instances and creates a subset of the original dataset.

a. NearMiss

b. Random Under Sampling


The complete experimental setting we implemented, including, the learning algorithms, performance evaluation method and metrics used, as well as the data pre-processing techniques and the imbalanced learning techniques that we considered in this study for each dataset is available in "Data Sampling and Model" folder.


## **6. AIS Resampling and Modeling:**
  ### **6.1 Implementation Details**
  
  In this section, we present the different key functions introduced as part of the AIS algorithm and explain their goal.
  
**1.	Initial Population:** Initialize the population at random depending on the minority class distribution of each feature of a dataset. The population size and feature size are input parameters that generate the initial population. It contains a number of rows equal to population size and a number of columns equivalent to feature size. The output population contains chromosomes/tuples similar to the minority class instances of a given dataset.

**2.	Fitness score:** This function calculates the F-score of each chromosome/tuple of the population. The population, the classifier used to calculate the f-score, the train, and the test sets are given as input parameters. Each chromosome/tuple of a population is appended to the train set, fit to the model using the updated train set, using the trained model results are predicted on the test set, and f-score for that chromosome/tuple is calculated. This way, for each chromosome, their respective f-scores are calculated and stored. A sorted list of chromosomes with their respective scores is the output of this function.

**3.	Selection:** Choose the best-fitting chromosomes as parents in order to pass on genes to the next generation to build a new population. This function requires two inputs one is the sorted population generated by the fitness score function and the other is the parent size that indicates how many chromosomes must be selected as parents for the next population generation. Based on this size, the best chromosomes/tuples from the sorted population are selected as the population for next generation which is the outcome of this function.

**4.	Clone:** This function takes the output from the selection function and clone rate value as input parameters. Based on the clone rate, the chromosomes/tuples of a population are copied and the new population set for the next generation is obtained as an outcome of this function that includes cloned chromosomes/tuples.

**5.	Mutation:** This function alters one or more attributes/genes of each chromosome/tuple of a population and generates a mutated population set. Input parameters include the cloned population and mutation rate. On each gene/attribute of a tuple/chromosome, the mutation rate is applied. The mutated population is considered to be the next generation population set.

Steps 2-5 are repeated until a defined stop condition is reached. After completing its execution, the AIS algorithm produces a new minority class population set. This way, we oversample the minority class, update the training set, and fit the model to calculate various performance evaluation metrics including accuracy, g-mean, f-score, precision, and recall to understand the impact of AIS on the imbalance datasets.  We also tried to apply popular oversampling and undersampling techniques after the use of our proposed AIS system to further enhance the models’ performance and address the class imbalance issue. All the code details and results obtained is available in "AIS_Sampling_and_Modeling" folder.

  ### **6.2 Flow chart for AIS algorithm**
  ![image](https://user-images.githubusercontent.com/58121387/115758300-f6f55200-a36c-11eb-9afe-765b8adc3586.png)

  ### **6.3 Pseudo code for AIS**
  ![image](https://user-images.githubusercontent.com/58121387/115759899-c6161c80-a36e-11eb-8b48-bb787c1b9b2d.png)

## ** 7. Conclusion:**
AIS are defined as intelligent computational algorithms inspired by the biological immune systems and gained a lot of attention from the past decades. We carried out a study to implement an AIS inspired algorithm as a sampling strategy to solve the class imbalance issue. We used a set of predictive models with the AIS algorithm to generate synthetic minority class samples and observed the results of several performance evaluation metrics. We found that the main drawback of the AIS is the generation of a very small number of synthetic samples compared to other sampling algorithms. The reasons for this could include the use of f-scores in stop condition, the size of the datasets, or the AIS parameters. However, this drawback has no bearing on the AIS algorithm results. In the majority of the datasets tested in our study, the AIS algorithm’s performance alone is equivalent to that of other oversampling/undersampling algorithms applied on the original dataset. This study helped us understand that, despite the small size of the generated sample set, they are the best samples. On the other hand, this also becomes an advantage of the AIS system proposed as a good performance is achieved by using a smaller dataset. To further explore the impact of the imbalance on the AIS sampled training set, we also experimented the application of suitable sampling algorithm to allow balancing the problem classes.

Overall, we find our AIS proposal to be competitive against other popular oversampling and undersampling algorithms. Moreover, a similar or better performance is achieved with a reduced number of synthetic cases. 

As future work, we would like to further apply our solution to other problem domains. Moreover, the proposed AIS algorithm could also be extended to the majority class cases for which we could generate good representatives, and this way we could perform undersampling. Another possible direction regards the exploration of other metrics for being included in the algorithm stopping criteria.



## **8. References**
1.	https://archive.ics.uci.edu/ml/datasets/ILPD+(Indian+Liver+Patient+Dataset). Last Accessed: 2020-02-01.   

2.	https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29 . Last Accessed: 2020-02-01.  

3.	https://sci2s.ugr.es/keel/dataset.php?cod=155 . Last Accessed: 2020-02-01.  

4.	https://archive.ics.uci.edu/ml/datasets/EEG+Eye+State . Last Accessed: 2020-02-01.  

5.	https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients . Last Accessed: 2020-02-01.  
 
6.	https://www.kaggle.com/mlg-ulb/creditcardfraud/ . Last Accessed: 2020-02-01.  

7.	https://www.kaggle.com/uciml/german-credit . Last Accessed: 2020-02-01.  

8.	Duangjai Jitkongchuen and Warattha Sukpongthai.  Handling imbalanced data classification problem using artificial immune system with mahalanobis distance. In 2019   20th IEEE/ACIS International Conference on  Software  Engineering,  Artificial  Intelligence,  Networking, and  Parallel/Distributed Computing (SNPD), pages 67–71. IEEE, 2019.

