# Affluency-Classification-Using-Energy-Time-Series
A project that demonstrates how a household's affluency class can be predicted from its energy usage profile.

## Introduction

The purpose of this project is to apply a Machine learning technique to assess whether SmartMeter energy consumption readings
for a given customer can be used to predict information about the customer's economic status. To this end, two different
Support Vector Machines are trained using a subset of daily energy consumption readings as training vectors and a corresponing subset of Acorn classifications as target
values. The models are then assessed to see how well each can retrieve the known Acorn classification for a set of test vectors that are not contained in the training set.

For the reader who wishes to jump right in, the Python code that was used for this project is contained in the Jupyter notebook ```Affluency_Classification.ipynb``` contained
in the top level of this repository.

The project was initiated by and undertaken with the help of Michael Blackmon (https://www.linkedin.com/in/michael-blackmon-b4431263).


## Data Source

The SmartMeter data has been made available on the following site.

https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households

This site allows the viewer to download a zip file of size 783 MB that contains an 11.3 GB CSV(comma seperated values) file with 167 million rows of data. 
The CSV file contains measurements of energy consumption in kW-h taken every half hour for each customer over a period of about two years. 
Furthermore, each customer is assigned an 'Acorn' designation together with a prosperity category based on the 'Acorn' designation. 
Acorn (developed by CACI Ltd in the UK) segments UK postcodes and neighborhoods into 6 Categories, 18 Groups and 62 types, three of which are not private households. 
By analyzing significant social factors and population behavior, it provides precise information and in-depth understanding 
of the different types of people (https://en.wikipedia.org/wiki/Acorn_(demographics)).

For the purpose of this project we use SmartMeter readings recorded during 2013 only, owing to the fact that 2013 provides the most complete set of readings.
The data itself contains a number of NaN values and zero values. The dataset used for this project is limited to readings from customers with less than 10% of their 
data containing NaN or zero values. NaN values are replaced with zeros. Additional cleaning was required to eliminate a number of duplicate records.

Additional data processing was required to create daily total energy usage for each customer. This was achieved using the groupby functionality provided by the Pandas Python library, 
and resulted in a dataset with 365 numerical features - one feature representing total energy consumption for a particular day in 2013.

Not all customers have a perfect record of energy consumption for 2013 - a perfect record being one for which a meaningful SmartMeter reading is available for the
entirity of 2013. Comments contained in the accompanying source code as well as those contained in this document refer to the subset of customers
for whom a perfect half-hourly record of energy consumption is available as the 'perfect customers' dataset. The complete set of 
customers (perfect or otherwise) is referred to as the 'entire customers' dataset.

The result of the data processing is a set of four .csv files, compressed versions of which are contained in the ```./data folder``` of this repository.

For the 'entire customers', we have the following two datsets:
<br>(1) ```dailyTotals.csv``` - containing total daily energy consumption for every day of 2013 for all customers 
<br>(2) ```entireCustomers.csv``` - containing the Acorn designation for all customers

For the 'perfect customers', we have the following two datsets:
<br>(3) ```perfectDailyTotals.csv``` - containing total daily energy consumption for the 'perfect customers'. 
<br>(4) ```perfectCustomers.csv``` - containing the Acorn designation for the 'perfect customers'.

Each row of the above four datsets is referenced by a unqiue customer 'id'. It is therefore possible to match an energy consumption record contained
in the 'dailyTotals' datset to an Acorn classification record contained in the 'Customers' dataset.

## Data Inspection

The first five rows of daily total energy consumption are displayed for the first few days of 2013 alongside the associated Acorn classification for the relevant customer.
A glimpse at the daily energy consumption reveals that a higher affluency ranking does not necessarily indicate a higher energy consumption.

<p align="center">
    <img src="https://raw.githubusercontent.com/JerryGreenough/Affluency-Classification-Using-Energy-Time-Series/master/img/affluency.png" width="550" height="183">  
</p>

## Machine Learning

The objective of the project is to use a customer's energy consumption profile (taken throughout the year 2013) in order to determine whether the customer
is "Affluent" or "Non-Affluent". This amounts to a binary classiciation problem using the 365 numerical features comprising the daily energy consumption figures. 
In order to accomplish this, it was decided to use the Support Vector Machine (SVM) functionality provided by the ```scikit-learn``` package. In particular, two
kernel functions were employed; the linear kernal function and the radial basis function. 

Both kernel functions were run for a variety of soft margin parameters (C) 
and in the case of the radial basis function influence parameter (gamma). This can be done quite readily using the ```GridSearchCV``` function provided by the scikit-learn 
package. The following code snippet illustrates the fact that 3-fold cross-validation was used when assessing each parameter combination. Furthermore, the ```njobs=-1``` specification
ensured that all available processors were used in running the job on parallel processors.

    grid = GridSearchCV(model, param_grid, verbose=3, n_jobs=-1, cv=3)
    grid.fit(X_train, y_train)

In addition to searching for the optimal parameter combination using ```GridSearchCV```, training took place on datasets with a variety of ratios for Non-Affluency count to Affluency count (
henceforth referred to as the 'training ratio').

All training was done using vectors taken from the 'perfect customer' dataset. Once a dataset with the correct training ratio had been identified, a test-train split of 0.25:0.75 was employed.
Once training had taken place, the resulting model was assessed against (i) the test portion taken from the 'perfect customer' dataset and thereafter (ii) the entire dataset in order to see how well
the model would generalize.

The exact implementation details are given in the accompanying source code.

The result of training the scikit-learn linear SVM using the 'perfect customer' data are shown below for a number of training ratios. For each training ratio, the parameter combination
[C, gamms] that GridSearchCV calculated as optimal is shown in tandem with four results for each generalization case, 'test' and 'entire' (described previously). The four results are
taken directly from the classification report (implemented in scikit-learn as ```classification_report```). The assessment results concerned are the 'precision', 'recall', 'accuracy' and 'f1-score' values
associated with the "Affluent" class.

<p align="center">
    <img src="https://raw.githubusercontent.com/JerryGreenough/Affluency-Classification-Using-Energy-Time-Series/master/img/linear.png" width="650" height="255">  
</p>

The result of training the scikit-learn radial basis function SVM using the 'perfect customer' data are shown below for a number of training ratios.

<p align="center">
    <img src="https://raw.githubusercontent.com/JerryGreenough/Affluency-Classification-Using-Energy-Time-Series/master/img/rbf.png" width="650" height="255">  
</p>



## Conclusion

An inspection of the f1-scores for the "Affluent" category suggests that the radial basis function SVM generally out-performs the linear SVM when generalized to the entire dataset.
The top f1 score occurs with the rbf SVM for a training ratio of 0.9 (C = 1.0, gamma = 0.000562 = 10^-3.25). This is not an unexpected result when considering the observation from earlier
that a higher affluency ranking is not necessarily an indicator of high energy usage.

Not surprisingly, the optimal models for each training ratio generalize far better to the test set taken from the
'perfect customer' data set. For instance, for a training ratio of 0.7, the Affluency precision is 0.64 and the recall is 0.84. 
Moreover, the performance of the model increases as the training ratio reduces. 

The overall conclusion is that the SVM approach, when used with an rbf kernel function, provides a reasonable means of predicting customer affluence - particularly for the case in 
which we attempt to predict affluency for customers with a perfect set of energy consumption readings. 
A recall figure of over 0.8 (in tandem with a precision figure of 0.63) for the case in which a training ratio of 0.7 is employed suggests that
the method can capture most of the Affluent customers, withoutaccumulating a prohibitive number of non-Affluent customers along the way.





