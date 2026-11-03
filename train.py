import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(filepath='hour.csv'):
    """載入資料並刪除無用/洩漏特徵"""
    df = pd.read_csv(filepath)
    cols_to_drop = ['casual', 'registered', 'atemp', 'instant', 'dteday']
    df = df.drop(columns=cols_to_drop)
    return df

def build_features(df, version='v3'):
    """特徵工程：這裡我們直接使用表現最好的 v3 (Sin/Cos) 編碼"""
    features = df.copy()
    
    if version == 'v3':
        features['hr_sin'] = np.sin(2 * np.pi * features['hr'] / 24)
        features['hr_cos'] = np.cos(2 * np.pi * features['hr'] / 24)
        features = features.drop(columns=['hr'])
        
    cat_cols = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']
    for col in cat_cols:
        if col in features.columns:
            features[col] = features[col].astype(str)
            
    return features.to_dict(orient='records')

def train_model():
    print("Loading data...")
    df = load_and_preprocess_data('hour.csv')
    
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    
    y_train = df_train['cnt'].values
    y_test = df_test['cnt'].values
    
    del df_train['cnt']
    del df_test['cnt']
    
    print("Building features (v3 encoding)...")
    train_dicts = build_features(df_train, version='v3')
    test_dicts = build_features(df_test, version='v3')
    
    print("Vectorizing...")
    dv = DictVectorizer(sparse=True)
    X_train = dv.fit_transform(train_dicts)
    X_test = dv.transform(test_dicts)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    print("Training XGBoost model...")
    xgb_params = {
        'eta': 0.1,
        'max_depth': 6,
        'min_child_weight': 1,
        'objective': 'reg:squarederror',
        'nthread': 8,
        'seed': 42,
        'verbosity': 0,
    }
    
    model = xgb.train(xgb_params, dtrain, num_boost_round=150)
    
    y_pred = model.predict(dtest)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Test RMSE: {rmse:.2f}")
    
    print("Saving model and vectorizer...")
    model.save_model('model.json')  
    with open('dv.pkl', 'wb') as f:
        pickle.dump(dv, f)          
    print("Done!")

if __name__ == '__main__':
    train_model()