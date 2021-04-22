# **ARTIFICIAL IMMUNE SYSTEMS BASED RESAMPLING FOR DEALING WITH CLASS IMBALANCE**

## **Project Goal:**
Many real-world datasets suffer from the class imbalance problem.Frequently, either oversampling or undersampling strategies are applied to balance the dataset. From the past few decades, Artificial Immune Systems (AIS) defined as the intelligent computational algorithms inspired by the biological immune systems and gained a lot of attention.Researchers aimed to develop various immune-based algorithms to address the complex real-world problems for a wide variety of application domains. Namely, there has been some research works where AIS are applied to solve specific problems that suffer from the class imbalance problem. However, these methods were applied as the learning algorithm to solve the tasks and were not used as a resampling method. Only one work exists that uses some techniques of AIS to address the class imbalance problem [8].

The goal of this project is to explore the use of AIS as a data resampling strategy to tackle the class imbalance problem. In this study, we will test, evaluate and compare different approaches for integrating AIS techniques as resampling methods to address the class imbalance problem. Our experiments will be carried out using a set of the public available imbalanced data sets from Health and Fraud detection fields.

## **Project Implementation Details:**
**Programming Language:** Python
**Tools:** Jupyter Notebook
**Cloud-based Platforms:** Kaggle and Google colab
**Downloaded Datasets from:** Kaggle, UCI repository and Keel repository.

## **Introduction to Datasets:**
In our study, we considered a set of imbalanced data sets related to the Health and Fraud detection fields that are listed in below table.
![image](https://user-images.githubusercontent.com/58121387/115747984-c0ffa000-a363-11eb-991b-53b837382f12.png)

## **Exploratory Data Analysis:**

## **Resampling and Modeling:**

## **AIS Resampling and Modeling:**
  ### **Flow chart for AIS algorithm**
  ![image](https://user-images.githubusercontent.com/58121387/115758300-f6f55200-a36c-11eb-9afe-765b8adc3586.png)

  ### **Pseudo code for AIS**
  ![image](https://user-images.githubusercontent.com/58121387/115759899-c6161c80-a36e-11eb-8b48-bb787c1b9b2d.png)

## **Conclusion:**
AIS are defined as intelligent computational algorithms inspired by the biological immune systems and gained a lot of attention from the past decades. We carried out a study to implement an AIS inspired algorithm as a sampling strategy to solve the class imbalance issue. We used a set of predictive models with the AIS algorithm to generate synthetic minority class samples and observed the results of several performance evaluation metrics. We found that the main drawback of the AIS is the generation of a very small number of synthetic samples compared to other sampling algorithms. The reasons for this could include the use of f-scores in stop condition, the size of the datasets, or the AIS parameters. However, this drawback has no bearing on the AIS algorithm results. In the majority of the datasets tested in our study, the AIS algorithm’s performance alone is equivalent to that of other oversampling/undersampling algorithms applied on the original dataset. This study helped us understand that, despite the small size of the generated sample set, they are the best samples. On the other hand, this also becomes an advantage of the AIS system proposed as a good performance is achieved by using a smaller dataset. To further explore the impact of the imbalance on the AIS sampled training set, we also experimented the application of suitable sampling algorithm to allow balancing the problem classes.

Overall, we find our AIS proposal to be competitive against other popular oversampling and undersampling algorithms. Moreover, a similar or better performance is achieved with a reduced number of synthetic cases. 

As future work, we would like to further apply our solution to other problem domains. Moreover, the proposed AIS algorithm could also be extended to the majority class cases for which we could generate good representatives, and this way we could perform undersampling. Another possible direction regards the exploration of other metrics for being included in the algorithm stopping criteria.



## **References**
1.	https://archive.ics.uci.edu/ml/datasets/ILPD+(Indian+Liver+Patient+Dataset). Last Accessed: 2020-02-01.   

2.	https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29 . Last Accessed: 2020-02-01.  

3.	https://sci2s.ugr.es/keel/dataset.php?cod=155 . Last Accessed: 2020-02-01.  

4.	https://archive.ics.uci.edu/ml/datasets/EEG+Eye+State . Last Accessed: 2020-02-01.  

5.	https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients . Last Accessed: 2020-02-01.  
 
6.	https://www.kaggle.com/mlg-ulb/creditcardfraud/ . Last Accessed: 2020-02-01.  

7.	https://www.kaggle.com/uciml/german-credit . Last Accessed: 2020-02-01.  

8.	Duangjai Jitkongchuen and Warattha Sukpongthai.  Handling imbalanced data classification problem using artificial immune system with mahalanobis distance. In 2019   20th IEEE/ACIS International Conference on  Software  Engineering,  Artificial  Intelligence,  Networking, and  Parallel/Distributed Computing (SNPD), pages 67–71. IEEE, 2019.

