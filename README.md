# PIMA_DiabetesPrediction
This project uses the PIMA Diabetes dataset obtained from Kaggle.The objective of the project is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.
We import all the necessary libraries and then read the dataset using pandas.read_csv() library and get a look of the data using data.shape() and data.head().
The datasets consists of several medical predictor variables and one target variable, Outcome. Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.
In preprocessing part we check whether there are any missing values or categorical values that we need to convert to numerical data. Using data.isnull().values.any() we see that there are no missing values also the data consists of only numerical values so we don't have to make any changes to the data.
Next, we get correlations of each features in dataset and get diabetes_true_count and diabetes_false_count using 
diabetes_true_count = len(data.loc[data['Outcome'] == 1])
diabetes_false_count = len(data.loc[data['Outcome'] == 0])
outcome=1 suggests that the person has diabetes and 0 suggests that the person does not have diabetes.
Next, we create independent and Dependent Features,  X contains all the columns except "Outcome" which is our target variable and Y contains the "Outcome" variable.
Using the train_test_split model we split our data into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=10)
This means 70% is train data on which we train our model and 30% is the test data on which we check our accuracy.
The dataset does not contain any missing values however, the value of some variables is zero which cannot be, 
