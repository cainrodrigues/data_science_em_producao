import pickle
import inflection
import pandas as pd
import numpy as np
import math
import time
import datetime

class Rossmann(object):
    def __init__(self):
        state = 1
        
        self.home_path = 'C:/Users/Caio/Desktop/Caio/repos/data_science_em_producao/'
        self.competition_distance_scaler   = pickle.load(open(self.home_path + 'parameters/competition_distance_scaler.pkl','rb'))
        self.competition_time_month_scaler = pickle.load(open(self.home_path + 'parameters/competition_time_month_scaler.pkl','rb'))
        self.promo_time_week_scaler        = pickle.load(open(self.home_path + 'parameters/promo_time_week_scaler.pkl','rb'))
        self.year_scaler                   = pickle.load(open(self.home_path + 'parameters/year_scaler.pkl','rb'))
        self.store_type_scaler             = pickle.load(open(self.home_path + 'parameters/store_type_scaler.pkl','rb'))
        
        
    def data_cleaning(self, df1,):

        ## 1.1. Rename Columns
        #The idea here is to get agility on development through easy names on the columns

        cols_old = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo',
               'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',
               'CompetitionDistance', 'CompetitionOpenSinceMonth',
               'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',
               'Promo2SinceYear', 'PromoInterval']

        snakecase = lambda x: inflection.underscore(x)

        cols_new = list(map(snakecase, cols_old))

        #rename
        df1.columns = cols_new

        ## 1.3. Data Types

        df1['date'] = pd.to_datetime(df1['date'])

        ## 1.5. Fillout NA

        # competition_distance      
        # Here i'm fillin the NA's with a value that is much higher than the max value for competitor distance on the dataset
        df1['competition_distance'] = df1['competition_distance'].apply(lambda x: 200000 if math.isnan(x) else x)

        # competition_open_since_month
        # Here, I'm assuming that is important to have this information filled (M02_V02_9min)
        df1['competition_open_since_month'] = df1.apply(lambda x: x['date'].month if math.isnan(x['competition_open_since_month'])
                                                                                        else x['competition_open_since_month'], axis=1)

        # competition_open_since_year 
        df1['competition_open_since_year'] = df1.apply(lambda x: x['date'].year if math.isnan(x['competition_open_since_year'])
                                                                                        else x['competition_open_since_year'], axis=1)

        # promo2_since_week      

        df1['promo2_since_week'] = df1.apply(lambda x: x['date'].week if math.isnan(x['promo2_since_week']) else x['promo2_since_week'], axis=1)

        # promo2_since_year    
        df1['promo2_since_year'] = df1.apply(lambda x: x['date'].year if math.isnan(x['promo2_since_year']) else x['promo2_since_year'], axis=1)

        month_map = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}

        # fill na's with 0 to avoid the comparison using 'isnan'
        df1['promo_interval'].fillna(0,inplace=True)

        # extract the month of the 'date' column and apply the dictionary created above to use as future comparison.
        df1['month_map'] = df1['date'].dt.month.map(month_map)

        # verifying if the store is participating in the promo, based on column 'date', represented by 'month_map'
        df1['is_promo'] = df1[['promo_interval','month_map']].apply(lambda x: 0 if x['promo_interval'] == 0 
                                                                    else 1 if x['month_map'] in x['promo_interval'].split(',')
                                                                    else 0, axis=1)

        ## 1.6. Change Types
        #It's important to verify if the types are correct after many modifications on the variables

        # These variables were float64.
        # competition
        df1['competition_open_since_month'] = df1['competition_open_since_month'].astype('int64')
        df1['competition_open_since_year'] = df1['competition_open_since_year'].astype('int64')
        
        # promo2
        df1['promo2_since_week'] = df1['promo2_since_week'].astype('int64')
        df1['promo2_since_year'] = df1['promo2_since_year'].astype('int64')

        return df1

    def feature_engineering(self, df2):

        ## 2.4. Feature Engineering

        # year
        df2['year'] = df2['date'].dt.year

        # month
        df2['month'] = df2['date'].dt.month

        # day
        df2['day'] = df2['date'].dt.day

        # week of year
        df2['week_of_year'] = df2['date'].dt.isocalendar().week

        # year week
        df2['year_week'] = df2['date'].dt.strftime('%Y-%W')

        # competition since
        # gather 'competition_open_since_year' and 'competition_open_since_month' together and then
        # subtracting it to 'date' so we can obtain how many months have passed for each store since
        # competitions opened
        df2['competition_since'] = df2.apply(lambda x: datetime.datetime(year=x['competition_open_since_year'],
                          month=x['competition_open_since_month'],
                         day=1), axis=1)

        # dividing by 30 so we can obtain the result as months
        df2['competition_time_month'] = ( (df2['date'] - df2['competition_since'])/30 ).apply(lambda x: x.days).astype(int)

        # promo since
        df2['promo_since'] = df2['promo2_since_year'].astype(str) + '-' + df2['promo2_since_week'].astype(str)

        # now we have to convert the 'promo_since' to datetime. This method is explained
        # on the bonus video, that is not launched at the moment (26/03) 
        df2['promo_since'] = df2['promo_since'].apply(lambda x: datetime.datetime.strptime(x + '-1', '%Y-%W-%w' ) - datetime.timedelta(days = 7) )
        df2['promo_time_week'] = ( (df2['date'] - df2['promo_since'] )/7 ).apply(lambda x: x.days ).astype(int)

        # assortment
        df2['assortment'] = df2['assortment'].apply(lambda x: 'basic' if x == 'a' else 'extra' if x == 'b'
                                                  else 'extended')

        # state holiday
        df2['state_holiday'] = df2['state_holiday'].apply(lambda x: 'public_holiday' if x == 'a' else 'easter_holiday' if x == 'b' 
              else 'christmas' if x == 'c' else 'regular_day')

        # 3.0. Filtragem de Variáveis

        ## 3.1. Filtragem das Linhas

        # stores open only
        df2 = df2[df2['open'] != 0]

        ## 3.2. Seleção das colunas

        #'customers' é uma restrição do negócio, pois não teremos o input no momento da análise de quantas pessoas
        # estarão nas lojas nas próximas 6 semanas
        # a coluna 'open' não será mais necessária pois estará preenchida totalmente apenas com o valor '1'
        # 'promo_interval' e 'month_map' são variáveis auxiliares, e não mais necessárias
        cols_drop = ['open','promo_interval','month_map']
        df2 = df2.drop(cols_drop, axis=1)
        
        return df2
    
    def data_preparation(self, df5):

        ## 5.2. Rescaling

        # competition distance
        # We inserted the outliers on 'fillot NA'. So, we use the 'Robust Scaler' method.
        # In this case, we are 'calling' the pickle archive inside the function we are creating.
        df5['competition_distance'] = self.competition_distance_scaler.fit_transform(df5[['competition_distance']].values)
        
        # competition time month - Robust Scaler method (a lot of outliers)
        df5['competition_time_month'] = self.competition_time_month_scaler.fit_transform(df5[['competition_time_month']].values)
        
        # promo time week - 'MinMaxScaler'
        df5['promo_time_week'] = self.promo_time_week_scaler.fit_transform(df5[['promo_time_week']].values)
        
        # year - MinMaxScaler
        df5['year'] = self.year_scaler.fit_transform(df5[['year']].values)


        ### 5.3.1. Encoding

        # state holiday - One Hot Encoding
        df5 = pd.get_dummies(df5, prefix=['state_holiday'], columns=['state_holiday'])

        # store type - Label Encoding
        df5['store_type'] = self.store_type_scaler.fit_transform(df5['store_type'])

        # assortment - Ordinal Encoding
        assortment_dict = {'basic':1,
                          'extra':2,
                          'extended':3}
        df5['assortment'] = df5['assortment'].map(assortment_dict)

        ### 5.3.3. Nature Transformation

        # day of week
        df5['day_of_week_sin'] = df5['day_of_week'].apply(lambda x: np.sin(x*(2*np.pi/7)))
        df5['day_of_week_cos'] = df5['day_of_week'].apply(lambda x: np.cos(x*(2*np.pi/7)))

        # day
        df5['day_sin'] = df5['day'].apply(lambda x: np.sin(x*(2*np.pi/30)))
        df5['day_cos'] = df5['day'].apply(lambda x: np.cos(x*(2*np.pi/30)))

        # month
        df5['month_sin'] = df5['month'].apply(lambda x: np.sin(x*(2*np.pi/12)))
        df5['month_cos'] = df5['month'].apply(lambda x: np.cos(x*(2*np.pi/12)))

        # week of year
        df5['week_of_year_sin'] = df5['week_of_year'].apply(lambda x: np.sin(x*(2*np.pi/52)))
        df5['week_of_year_cos'] = df5['week_of_year'].apply(lambda x: np.cos(x*(2*np.pi/52)))
        
        cols_selected_boruta = ['store','promo','store_type','assortment','competition_distance',
                                'competition_open_since_month','competition_open_since_year','promo2','promo2_since_week',
                                'promo2_since_year','competition_time_month','promo_time_week','day_of_week_sin','day_of_week_cos',
                                'month_sin','month_cos','day_sin','day_cos','week_of_year_sin','week_of_year_cos']

        return df5[cols_selected_boruta]
    
    def get_prediction(self, model, original_data, test_data):
        # prediction
        pred = model.predict(test_data)
        
        # join pred into the original data so the users can view all data
        original_data['prediction'] = np.expm1(pred)
        
        return original_data.to_json(orient = 'records', date_format = 'iso')
