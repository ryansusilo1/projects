from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE
import pandas as pd
import numpy as np

class UserPredictor:

    def __init__(self):
        self.model = None

    def _features_train(self, train_users, train_logs, train_y):

        # Total and avg time feature
        train_data = train_users[['user_id', 'past_purchase_amt', 'age']].merge(train_logs[['user_id', 'seconds']], on='user_id', how='left')
        #print(train_data.shape)
        train_data['seconds'] = train_data['seconds'].fillna(0)
        #print(train_data.shape)
        train_data = train_data[['user_id', 'past_purchase_amt', 'age']].merge(train_data.groupby('user_id')['seconds'].sum(), on='user_id').drop_duplicates().merge(train_data.groupby('user_id')['seconds'].mean(), on='user_id').drop_duplicates()
        train_data = train_data.rename({'seconds_x': 'total_time', 'seconds_y': 'avg_time'}, axis=1)
        
        # Laptop visits feature
        train_data['laptop_visits'] = train_logs[train_logs['url']=='/laptop.html'].groupby('user_id')['user_id'].count()
        train_data['non_laptop_visits'] = train_logs[~train_logs['url'].eq('/laptop.html')].groupby('user_id')['user_id'].count()
        
        train_data = train_data.fillna(0)
        train_data['no_visits'] = np.where(train_data['laptop_visits'] + train_data['non_laptop_visits'] == 0, 1, 0)
        #print(train_data.shape)
        
        # Badges feature
        badge_dummies_train = pd.get_dummies(train_users['badge'], dtype=float)
        train_users = pd.concat([train_users, badge_dummies_train], axis=1)
        train_data['bronze'] = train_users['bronze']
        train_data['silver'] = train_users['bronze']
        train_data['gold'] = train_users['gold']
        train_data = train_data.fillna(0)
        train_data['total_badges'] = train_data['gold'] + train_data['silver'] + train_data['bronze']
        #print(train_data.shape)
        
        train_data = train_data.merge(train_y, on='user_id')
        #print(train_data.shape)
        
        # Total time per user who visited laptop.html
        laptop_train = train_logs[train_logs['url'] == '/laptop.html']
        train_data = train_data.merge(laptop_train.groupby('user_id')['seconds'].sum(), on='user_id', how='left').fillna(0).merge(laptop_train.groupby('user_id')['seconds'].mean(), on='user_id', how='left').fillna(0).rename({'seconds_x': 'total_time_laptop', 'seconds_y': 'avg_time_laptop'}, axis=1)
        
        return train_data

    def _features_test(self, test_users, test_logs):
        
        # Total and avg time feature
        test_data = test_users[['user_id', 'past_purchase_amt', 'age']].merge(test_logs[['user_id', 'seconds']], on='user_id', how='left')
        test_data['seconds'] = test_data['seconds'].fillna(0)
        test_data = test_data[['user_id', 'past_purchase_amt', 'age']].merge(test_data.groupby('user_id')['seconds'].sum(), on='user_id').drop_duplicates().merge(test_data.groupby('user_id')['seconds'].mean(), on='user_id').drop_duplicates()
        test_data = test_data.rename({'seconds_x': 'total_time', 'seconds_y': 'avg_time'}, axis=1)
        
        # Laptop visits feature
        test_data['laptop_visits'] = test_logs[test_logs['url']=='/laptop.html'].groupby('user_id')['user_id'].count()
        test_data['non_laptop_visits'] = test_logs[~test_logs['url'].eq('/laptop.html')].groupby('user_id')['user_id'].count()
        
        test_data = test_data.fillna(0)
        test_data['no_visits'] = np.where(test_data['laptop_visits'] + test_data['non_laptop_visits'] == 0, 1, 0)
        
        # Badges feature
        badge_dummies_test = pd.get_dummies(test_users['badge'], dtype=float)
        test_users = pd.concat([test_users, badge_dummies_test], axis=1)
        test_data['bronze'] = test_users['bronze']
        test_data['silver'] = test_users['silver']
        test_data['gold'] = test_users['gold']

        test_data = test_data.fillna(0)
        test_data['total_badges'] = test_data['gold'] + test_data['silver'] + test_data['bronze']
        
        # Total and avg time per user who visited laptop.html
        laptop_test = test_logs[test_logs['url'] == '/laptop.html']
        test_data = test_data.merge(laptop_test.groupby('user_id')['seconds'].sum(), on='user_id', how='left').fillna(0).merge(laptop_test.groupby('user_id')['seconds'].mean(), on='user_id', how='left').fillna(0).rename({'seconds_x': 'total_time_laptop', 'seconds_y': 'avg_time_laptop'}, axis=1)
        
        return test_data

    def fit(self, train_users, train_logs, train_y):
        feat = ['past_purchase_amt', 'age', 'total_time', 'avg_time', 'total_time_laptop', 'avg_time_laptop', 'total_badges']#,'laptop_visits', 'non_laptop_visits', 'no_visits']
        self.feat = feat
        print("Features:", feat)

        # Create features from training data
        train_data = self._features_train(train_users, train_logs, train_y) 
        #print("train_data shape: ", train_data.shape)        
        
        # Polynomial features transformer to handle non-linear relationships
        poly_features = PolynomialFeatures(degree=2, include_bias=False)
        self.poly_features = poly_features
        poly_features.fit(train_data[feat])
        train_data_poly = poly_features.transform(train_data[feat])
        
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression())   
        ])
        
        # Train the model
        pipe.fit(train_data_poly, train_data['y'])
        self.model = pipe 
        
        # Cross-validation
        scores = cross_val_score(self.model, train_data_poly, train_data["y"], cv=5) 
        print(f"Cross-Validation Scores: {scores}")
        print(f"AVG: {scores.mean()}, STD: {scores.std()}\n")

    def predict(self, test_users, test_logs):
    
        # Create features from test data
        test_data = self._features_test(test_users, test_logs)
        #print("test_data shape: ",test_data.shape)
        test_data_poly = self.poly_features.transform(test_data[self.feat])
        
        # Ensure model is fitted
        if self.model is None:
            raise ValueError("Model not fitted yet. Call 'fit' before prediction.")

        # Make predictions
        return self.model.predict(test_data_poly)