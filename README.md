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

![Average monthly energy consumption during a typical day](https://raw.githubusercontent.com/JerryGreenough/Affluency-Classification-Using-Energy-Time-Series/master/img/affluency.png)

## Conclusion



