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
df_train = pd.read_csv('./datasets/z_train.csv')
# df_train.head()
# df_train.info()
# df_train.shape
# df_train.target.value_counts()

# --------------------------------------------------------------------
# Transformation
# --------------------------------------------------------------------
def changeTypeToCagory(df, cols):
    # change numerics variables that should be categorical
    df = df.copy()
    for col in cols:
        df[col] = df[col].astype('category')
    return df
# --------------------------------------------------------------------
def balanceTrainSet(df):
    # balance df_train and over-sampling 1
    df = df.copy()
    df_0 = df[df['target'] == 0]
    df_1 = df[df['target'] == 1]
    df_1_resample = resample(df_1, replace=True, n_samples=df_0.shape[0], random_state=123)
    df_bal = pd.concat([df_1_resample, df_0])
    df_bal = df_bal.sample(frac=1)
    return df_bal
# --------------------------------------------------------------------
def ScaleNumericVariables(df):
    # Scale numerical variables
    df = df.copy()
    numeric_cols = df.select_dtypes(include=['int64']).columns.tolist()
    scaler = StandardScaler()
    scaler.fit(df[numeric_cols])
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    return df
# --------------------------------------------------------------------
def generateDummySet(df):
    # Generate dummy variables
    df = df.copy()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df

# Main Pipeline
df1 = changeTypeToCagory(df_train, ['enrollee_id','target'])    
df2 = balanceTrainSet(df1)
df3 = ScaleNumericVariables(df2)
df4 = generateDummySet(df3)
# df4.info()

# creamos sets de entrenamiento
# --------------------------------------------------------------------
X = df4.drop(['enrollee_id','target'], axis=1).to_numpy()
y = df4.target.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15, stratify=y)

# Modelo de Regresion Logistica
# --------------------------------------------------------------------
logreg = LogisticRegression(solver='lbfgs', max_iter=1000)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print(classification_report(y_test, y_pred))