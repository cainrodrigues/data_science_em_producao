import pandas as pd
import pickle
from flask           import Flask, request, Response
from rossmann.Rossmann import Rossmann

# Loading model
model = pickle.load(open('C:/Users/Caio/Desktop/Caio/repos/data_science_em_producao/models/model_rossmann.pkl','rb') )

# Initialize API
#methods = ['Post'] means that it only sends data
app = Flask(__name__)

@app.route('/rossmann/predict', methods=['POST'])
def rossmann_predict():
    test_json = request.get_json() # gets the json that will be sent via API
    
    if test_json: # if there's data
        if isinstance(test_json, dict):  # if only one data
            test_raw = pd.DataFrame(test_json, index=[0])
            
        else: # if multiple json's
            test_raw = pd.DataFrame(test_json, columns = test_json[0].keys()) 
            
        # Instantiate Rossmann Class
        pipeline = Rossmann()
        
        # data cleaning
        df1 = pipeline.data_cleaning(test_raw)
        
        # feature engineering
        df2 = pipeline.feature_engineering(df1)
        
        # data preparation
        df3 = pipeline.data_preparation(df2)
        
        # prediction
        df_response = pipeline.get_prediction(model, test_raw, df3)
        
        return df_response
    
    else:
        return Response('{}', status = 200, mimetype = 'application/json') # return empty if there's no data

if __name__ == '__main__':
    app.run('127.0.0.1') # run on local host
