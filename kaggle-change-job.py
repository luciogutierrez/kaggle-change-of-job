# --------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------
import pandas as pd
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# --------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------
df_train = pd.read_csv('./datasets/z_train_change_job.csv')
df_target = pd.read_csv('./datasets/z_test_change_job.csv')

# --------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------

# Train Set
# --------------------------------------------------------------------
# df_train.head()
# df_train.info()
# df_train.isna().sum()
# df_train.target.value_counts()
# df_train.columns
# df_train.target.value_counts(normalize=True)

# Target Set
# --------------------------------------------------------------------
# df_target.head()
# df_target.info()
# df_target.isna().sum()
# df_target.target.value_counts()
# df_target.columns

# --------------------------------------------------------------------
# Transformation
# --------------------------------------------------------------------

# Selecting variables to include in the model
# --------------------------------------------------------------------
def selectingVariablesToModel(df, dataset):
    """
    ['enrollee_id', 'city', 'city_development_index', 'gender',
       'relevent_experience', 'enrolled_university', 'education_level',
       'major_discipline', 'experience', 'company_size', 'company_type',
       'last_new_job', 'training_hours', 'target']    
    """
    df = df.copy()
    model_cols = ['enrollee_id','city','city_development_index','gender',
                    'relevent_experience','enrolled_university','education_level',
                    'major_discipline', 'experience', 'company_size', 'company_type',
                    'last_new_job','training_hours']
    if dataset == 0:
        model_cols.append('target')
    df = df[model_cols]
    return df

# Droping citys items that not exist in target dataset
# --------------------------------------------------------------------
def depuraTrainSet(df, df2):
    df = df.copy()
    df2_city_list = df2.city.value_counts().index.tolist()
    df.drop(df[~df.city.isin(df2_city_list)].index, inplace = True)
    return df

# Change type of numerics variables that should be categorical
# --------------------------------------------------------------------
def changeTypeToCagory(df, cols):
    df = df.copy()
    for col in cols:
        df[col] = df[col].astype('category')
    return df

# balance train set with over-sampling
# --------------------------------------------------------------------
def balanceTrainSet(df):
    df = df.copy()
    df_0 = df[df.target == 0]
    df_1 = df[df.target == 1]
    df_1_resample = resample(df_1, replace=True, n_samples=df_0.shape[0])
    df_bal = pd.concat([df_1_resample, df_0])
    df_bal = df_bal.sample(frac=1)
    return df_bal

# Scale numerical variables
# --------------------------------------------------------------------
def ScaleNumericVariables(df):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=['int64']).columns.tolist()
    scaler = StandardScaler()
    scaler.fit(df[numeric_cols])
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    return df

# Generate dummy variables
# --------------------------------------------------------------------
def generateDummySet(df):
    df = df.copy()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df

# Train Pipeline
# --------------------------------------------------------------------
df = depuraTrainSet(df_train, df_target)
df = selectingVariablesToModel(df, 0)
df = changeTypeToCagory(df, ['enrollee_id','target'])    
df = balanceTrainSet(df)
df = ScaleNumericVariables(df)
df = generateDummySet(df)

# Making X and y train/test sets
# --------------------------------------------------------------------
X = df.drop(['enrollee_id','target'], axis=1).to_numpy()
y = df.target.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15, stratify=y)

# Training Logistic Regresion model
# --------------------------------------------------------------------
logreg = LogisticRegression(solver='lbfgs', max_iter=1000)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print(classification_report(y_test, y_pred))

# # Target Pipeline
# # --------------------------------------------------------------------
# dft = changeTypeToCagory(df_target, ['enrollee_id'])    
# dft = selectingVariablesToModel(dft, 1)
# dft = ScaleNumericVariables(dft)
# dft = generateDummySet(dft)

# # Making predictions on target dataset
# # --------------------------------------------------------------------
# X_t = dft.drop(['enrollee_id'], axis=1).to_numpy()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15, stratify=y)
# y_pred_t = logreg.predict(X_t)
# df_target['target'] = y_pred_t
# df_submission = df_target[['enrollee_id','target']]
# print(df_submission.target.value_counts())

# # Save submission csv file to upload in kaggle
# # --------------------------------------------------------------------
# df_submission.to_csv('./outputs/submission.csv', index=False)