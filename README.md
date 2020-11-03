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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=10)
This means 80% is train data on which we train our model and 20% is the test data on which we check our accuracy.
The dataset does not contain any missing values however, the value of some variables is zero which cannot be, So we mark zero values as missing or NaN

from numpy import nan
X_train[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']] = X_train[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']].replace(0, nan)
X_test[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']] = 
X_test[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']].replace(0, nan)

and fill these missing values with mean column values
X_train.fillna(X_train.mean(), inplace=True)
X_test.fillna(X_test.mean(), inplace=True)

We use Random Forest for classification. A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. The sub-sample size is controlled with the max_samples parameter if bootstrap=True (default), otherwise the whole dataset is used to build each tree.

We fit our model on X_train and y_train and predict on X_test.

scikit-learn has a handy function we can use to calculate accuracy: metrics.accuracy_score(). The function accepts two parameters,the actual values and our predicted values respectively, and returns our accuracy score.

Our model gives an accuracy of 75.974% when tested against our 20% test set. 
