![DataMexPT2021](imgs/datascience.jpg)

# Project: "DATAMEXPT2021 Competencia en clase!

## Kaggle Change of Job predictions

## Overview
A company which is active in Big Data and Data Science wants to hire data scientists among people
who successfully pass some courses which conduct by the company.

In this competition, you’ll predict if a candidate will work for the company based on the current credentials,
demographics,experience. 

You'll help reduce the cost and time as well as the quality of training or planning the courses and categorization of candidates.

## Context and Content
A company which is active in Big Data and Data Science wants to hire data scientists among people who successfully pass some courses
which conduct by the company. Many people signup for their training. Company wants to know which of these candidates are really wants
to work for the company after training or looking for a new employment because it helps to reduce the cost and time as well as 
the quality of training or planning the courses and categorization of candidates. Information related to demographics, education, 
experience are in hands from candidates signup and enrollment.

This dataset designed to understand the factors that lead a person to leave current job for HR researches too. 
By model(s) that uses the current credentials,demographics,experience data you will predict the probability of a candidate 
to look for a new job or will work for the company, as well as interpreting affected factors on employee decision.

The whole data divided to train and test . Target isn't included in test but the test target values data file is in hands 
for related tasks. A sample submission correspond to enrollee_id of test set provided too with columns : enrollee _id , target

## Note:
The dataset is imbalanced.
Most features are categorical (Nominal, Ordinal, Binary), some with high cardinality.
Missing imputation can be a part of your pipeline as well.
Features

## Dictionary
* enrollee_id             : Unique ID for candidate
* city                    : City code
* city_development_index  : Developement index of the city (scaled)
* gender                  : Gender of candidate
* relevent_experience     : Relevant experience of candidate
* enrolled_university     : Type of University course enrolled if any
* education_level         : Education level of candidate
* major_discipline        : Education major discipline of candidate
* experience              : Candidate total experience in years
* company_size            : No of employees in current employer's company
* company_type            : Type of current employer
* lastnewjob              : Difference in years between previous job and current job
* training_hours          : training hours completed
* target                  : 0–Not looking for job change, 1–Looking for a job change

## Inspiration
Predict the target of a candidate will work for the company
Interpret model(s) such a way that illustrate which features affect candidate decision

## Technical Requirements
* The dataset shark-attacks.
* Pandas library.
* A Jupyter Notebook ** kaggle-change-of-job.ipynb ** with code.
* A clean model_dataset.csv file.

## Development procedure
* 1- Import libraries we'll use ``Pandas``, ``Numpy``, ``digists`` and ``DateTime``.
* 2- Import z_train dataset into a pandas dataframe.
* 3- Analize data with pandas methods like **info(), head() and datacolums**.
* 4- Develop functions to treat data as follow:
    * 1 ``clean_headers()`` - Clean white spaces in headers.
    * 2 ``select_useful_columns_1()`` - Chosing relevants columns from main data.
    * 3 ``rename_columns()`` - Change some titles names to best undertanding.
    * 4 ``remove_empty_rows()`` - Delete empty rows from data.
    * 5 ``get_date()`` - Extract date from field with date in.
    * 6 ``nulls_treatment()`` - null's treatment.
    * 7 ``categorized_type()`` - categorize type of shark attak.
    * 8 ``to_numeric()`` - return just numeric values in string.
    * 9 ``get_numeric()`` - get numeric values from data.
    * 10 ``replace_specific_value()`` - replace specific_values from column.
    * 11 ``select_useful_columns_2()`` - Select final columns.
    * 12 ``re_order_columns()`` - to arrange columns appropriately.
* 5-**Main Pipeline** transform data from a function to another.
* 6-Export final dataframe to a **model_dataset.csv** file.

## Useful Resources
* [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)
* [Pandas Tutorials](https://pandas.pydata.org/pandas-docs/stable/tutorials.html)
* [StackOverflow Pandas Questions](https://stackoverflow.com/questions/tagged/pandas)
* [Awesome Public Data Sets](https://github.com/awesomedata/awesome-public-datasets)
* [Kaggle Data Sets](https://www.kaggle.com/datasets)
