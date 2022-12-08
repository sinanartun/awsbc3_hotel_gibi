import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

import xgboost as xgb
from xgboost import XGBClassifier

from sklearn.ensemble import VotingClassifier
pd.options.mode.chained_assignment = None


from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report


df = pd.read_csv(r'hotel_bookings.csv')


df['arrival_date_month'].replace({'January' : '1',
        'February' : '2',
        'March' : '3',
        'April' : '4',
        'May' : '5',
        'June' : '6',
        'July' : '7',
        'August' : '8',
        'September' : '9',
        'October' : '10',
        'November' : '11',
        'December' : '12'}, inplace=True)


def week_function(f1, f2, df):
    df['weekend_or_weekday'] = 0
    for i in range(0, len(df)):
        if f2.iloc[i] == 0 and f1.iloc[i] > 0:
            df['weekend_or_weekday'].iloc[i] = 'haftasonu'
        if f2.iloc[i] > 0 and f1.iloc[i] == 0:
            df['weekend_or_weekday'].iloc[i] = 'haftaici'
        if f2.iloc[i] > 0 and f1.iloc[i] > 0:
            df['weekend_or_weekday'].iloc[i] = 'ikiside'
        if f2.iloc[i] == 0 and f1.iloc[i] == 0:
            df['weekend_or_weekday'].iloc[i] = 'tanımsız'


week_function(df['stays_in_weekend_nights'], df['stays_in_week_nights'], df)


df['adr'] = df['adr'].astype(float)

df['all_children'] = df['children'] + df['babies']

df['children'] =  df['children'].fillna(0)
df['all_children'] =  df['all_children'].fillna(0)
df['country'] = df['country'].fillna(df['country'].mode().index[0])
df['agent']= df['agent'].fillna('0')
df = df.drop(['company'], axis =1)

df['agent']= df['agent'].astype(int)
df['country']= df['country'].astype(str)

cat_cols = df[['hotel','is_canceled','arrival_date_month','meal',
                                     'country','market_segment','distribution_channel',
                                     'is_repeated_guest', 'reserved_room_type',
                                     'assigned_room_type','deposit_type','agent',
                                     'customer_type','reservation_status',
                                     'weekend_or_weekday']]

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

for col in cat_cols:
    label_encoder(df, col)

ohe_cols = [col for col in df.columns if 60 >= df[col].nunique() > 2]
one_hot_encoder(df, ohe_cols)

cat_cols.info()

num_cols= df.drop(['hotel','is_canceled', 'arrival_date_month','meal',
                                       'country','market_segment','distribution_channel',
                                       'is_repeated_guest', 'reserved_room_type',
                                       'assigned_room_type','deposit_type','agent',
                                       'customer_type','reservation_status',
                                       'weekend_or_weekday'], axis = 1)
num_cols.info()

df = df.drop(['reservation_status', 'children', 'reservation_status_date'], axis=1)

df_model = df
df_tunning = df

## Finding parameters for XGBoost model

# model = XGBClassifier()
# parameters = {
# 'n_estimators' : [100,250,500],
# 'learning_rate' : [0.01, 0.1],
# 'subsample' :[0.5, 1.0],
# 'max_depth' : [3,5,7],
# 'criterion' : ['giny','entropy'],
# 'objective':['binary:logistic'],
# }

# grid_search = GridSearchCV(estimator=model, param_grid=parameters,
#                           cv=5, scoring='f1', verbose=True, n_jobs=-1)
# grid_search.fit(X, y)
# print(grid_search.best_score_)
# print(grid_search.best_params_)

y_model = df.iloc[:,1]
X_model = pd.concat([df_tunning.iloc[:,0],df_tunning.iloc[:,2:30]], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_model, y_model, test_size=0.3, random_state=42)

standardScalerX = StandardScaler()
X_train = standardScalerX.fit_transform(X_train)
X_test = standardScalerX.fit_transform(X_test)

kfold_cv=StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

for train_index, test_index in kfold_cv.split(X_model,y_model):
    X_train, X_test = X_model.iloc[train_index], X_model.iloc[test_index]
    y_train, y_test = y_model.iloc[train_index], y_model.iloc[test_index]

xgb_model = XGBClassifier( learning_rate = 0.01, max_depth = 5, n_estimators = 500,
                          objective ='binary:logistic', subsample = 1.0)
# fit the model
xgb_model.fit(X_train, y_train)
#Predict Model
predict_xgb = xgb_model.predict(X_test)

print("XGB", classification_report(y_test, predict_xgb))

#Model Kaydetme

joblib.dump(xgb_model, "xgb_model.pkl")


