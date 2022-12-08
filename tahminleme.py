import pandas as pd
import numpy as np
import joblib

import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

#Tahmin edilecek değerleri dataframe'e dönüştürür, df_ile_birleştirme isimli fonksiyonun içinde çalışır
def deger_olusturma():
    data = ['Resort Hotel', np.nan, 342, 2017, 'July', 27, 1, 0, 0, 2, 0, 0, 'BB', 'PRT', 'Direct', 'Direct',
            0, 0, 0, 'C', 'C', 3, 'No Deposit', np.nan, np.nan, 0, 'Transient', 0, 0, 0, 'Check-Out', '2017/07/15']

    data_columns = ['hotel', 'is_canceled', 'lead_time', 'arrival_date_year',
                    'arrival_date_month', 'arrival_date_week_number',
                    'arrival_date_day_of_month', 'stays_in_weekend_nights',
                    'stays_in_week_nights', 'adults', 'children', 'babies', 'meal',
                    'country', 'market_segment', 'distribution_channel',
                    'is_repeated_guest', 'previous_cancellations',
                    'previous_bookings_not_canceled', 'reserved_room_type',
                    'assigned_room_type', 'booking_changes', 'deposit_type', 'agent',
                    'company', 'days_in_waiting_list', 'customer_type', 'adr',
                    'required_car_parking_spaces', 'total_of_special_requests',
                    'reservation_status', 'reservation_status_date']

    deger = pd.DataFrame(data, data_columns).T

    deger['reservation_status_date'] = pd.to_datetime(deger['reservation_status_date'], infer_datetime_format=True)

    return deger #ta #

#Tahmin edilecek değerlerle ana veriyi birleştirir, bu işlem feature engineering işlemlerinin birlikte çalışması için yapılır
def df_ile_birlestirme():
    df_1 = pd.read_csv(
        r'hotel_bookings.csv')

    deger = deger_olusturma()
    df = pd.concat([deger, df_1.loc[:]]).reset_index(drop=True)
    return  df

df = df_ile_birlestirme()

#feature_enginering içinde çalışır
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

#feature_enginering içinde çalışır
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

#feature_enginering içinde çalışır
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

#Tahmin edilecek değerlerle birleşmiş veriyi temizler, encode eder, veriyi tahmine hazır hale getirir.
def feature_engineering(df):

    df['arrival_date_month'].replace({'January': '1',
                                      'February': '2',
                                      'March': '3',
                                      'April': '4',
                                      'May': '5',
                                      'June': '6',
                                     'July': '7',
                                      'August': '8',
                                      'September': '9',
                                      'October': '10',
                                      'November': '11',
                                      'December': '12'}, inplace=True)

    week_function(df['stays_in_weekend_nights'], df['stays_in_week_nights'], df)

    df['adr'] = df['adr'].astype(float)
    df['all_children'] = df['children'] + df['babies']
    df['children'] = df['children'].fillna(0)
    df['all_children'] = df['all_children'].fillna(0)
    df['country'] = df['country'].fillna(df['country'].mode().index[0])
    df['agent'] = df['agent'].fillna('0')
    df = df.drop(['company'], axis=1)
    df['agent'] = df['agent'].astype(int)
    df['country'] = df['country'].astype(str)
    cat_cols = df[['hotel', 'is_canceled', 'arrival_date_month', 'meal',
                   'country', 'market_segment', 'distribution_channel',
                   'is_repeated_guest', 'reserved_room_type',
                   'assigned_room_type', 'deposit_type', 'agent',
                   'customer_type', 'reservation_status',
                   'weekend_or_weekday']]

    for col in cat_cols:
        label_encoder(df, col)

    ohe_cols = [col for col in df.columns if 60 >= df[col].nunique() > 2]
    one_hot_encoder(df, ohe_cols)

    num_cols = df.drop(['hotel', 'is_canceled', 'arrival_date_month', 'meal',
                        'country', 'market_segment', 'distribution_channel',
                        'is_repeated_guest', 'reserved_room_type',
                        'assigned_room_type', 'deposit_type', 'agent',
                        'customer_type', 'reservation_status',
                        ], axis=1)

    df = df.drop(['is_canceled','reservation_status', 'children', 'reservation_status_date'], axis=1)

    standardScalerX = StandardScaler()
    x_model = standardScalerX.fit_transform(df)

    return x_model

def tahminleme():
    x_model = feature_engineering(df)
    xgb_model_from_disc = joblib.load(r"xgb_model.pkl")
    sonuc = xgb_model_from_disc.predict(x_model)
    print('Tahmin Edilen Değer:', sonuc[0])
    return sonuc[0]

sonuc = tahminleme()
sonuc






