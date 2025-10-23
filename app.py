# app.py
from flask import Flask, render_template, jsonify
import fastf1
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import os
from tqdm import tqdm
import logging

# Suppress FastF1 logging
logging.getLogger('fastf1').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Create cache directory for FastF1 data
cache_dir = 'cache/'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# Enable FastF1 cache to speed up data loading
fastf1.Cache.enable_cache(cache_dir)

class F1AdvancedFeatureEngineer:
    def __init__(self, season=None):
        self.season = season if season else datetime.now().year
        self.features = {}
    
    def calculate_track_performance(self, driver_abbr, historical_df):
        features = {}
        
        driver_data = historical_df[
            (historical_df['driver'] == driver_abbr) & 
            (historical_df['season'] == self.season)
        ]
        
        if len(driver_data) == 0:
            return features
        
        race_data = driver_data[driver_data['race_type'] == 'race']
        sprint_data = driver_data[driver_data['race_type'] == 'sprint']
        
        features[f'{driver_abbr}_avg_finish_pos'] = race_data['position'].mean() if len(race_data) > 0 else 99
        features[f'{driver_abbr}_avg_grid_pos'] = race_data['grid'].mean() if len(race_data) > 0 else 99
        features[f'{driver_abbr}_total_points'] = race_data['points'].sum() if len(race_data) > 0 else 0
        features[f'{driver_abbr}_wins'] = len(race_data[race_data['position'] == 1]) if len(race_data) > 0 else 0
        features[f'{driver_abbr}_podiums'] = len(race_data[race_data['position'] <= 3]) if len(race_data) > 0 else 0
        features[f'{driver_abbr}_top10'] = len(race_data[race_data['position'] <= 10]) if len(race_data) > 0 else 0
        features[f'{driver_abbr}_top5'] = len(race_data[race_data['position'] <= 5]) if len(race_data) > 0 else 0
        features[f'{driver_abbr}_dnf_rate'] = len(race_data[race_data['status'] == 'DSQ']) / len(race_data) if len(race_data) > 0 else 0
        
        if len(race_data) > 1:
            features[f'{driver_abbr}_pos_std'] = race_data['position'].std()
            features[f'{driver_abbr}_grid_to_finish_improvement'] = (race_data['grid'] - race_data['position']).mean()
        else:
            features[f'{driver_abbr}_pos_std'] = 0
            features[f'{driver_abbr}_grid_to_finish_improvement'] = 0
        
        if len(sprint_data) > 0:
            features[f'{driver_abbr}_sprint_avg_pos'] = sprint_data['position'].mean()
            features[f'{driver_abbr}_sprint_total_points'] = sprint_data['points'].sum()
            features[f'{driver_abbr}_sprint_wins'] = len(sprint_data[sprint_data['position'] == 1])
            features[f'{driver_abbr}_sprint_podiums'] = len(sprint_data[sprint_data['position'] <= 3])
        else:
            features[f'{driver_abbr}_sprint_avg_pos'] = 99
            features[f'{driver_abbr}_sprint_total_points'] = 0
            features[f'{driver_abbr}_sprint_wins'] = 0
            features[f'{driver_abbr}_sprint_podiums'] = 0
        
        if len(race_data) >= 5:
            recent_data = race_data.tail(5)
            features[f'{driver_abbr}_recent_avg_pos'] = recent_data['position'].mean()
            features[f'{driver_abbr}_recent_points'] = recent_data['points'].sum()
        else:
            features[f'{driver_abbr}_recent_avg_pos'] = features[f'{driver_abbr}_avg_finish_pos']
            features[f'{driver_abbr}_recent_points'] = features[f'{driver_abbr}_total_points']
        
        return features

    def calculate_championship_dynamics(self, historical_df):
        features = {}
        
        current_season = historical_df[historical_df['season'] == self.season]
        
        driver_points = current_season.groupby('driver')['points'].sum()
        driver_positions = current_season.groupby('driver')['position'].mean()
        driver_wins = current_season[current_season['position'] == 1].groupby('driver').size()
        
        if len(driver_points) > 0:
            leader = driver_points.idxmax()
            leader_points = driver_points.max()
            
            features['championship_leader'] = leader
            features['championship_leader_points'] = leader_points
            
            for driver in driver_points.index:
                gap = leader_points - driver_points[driver]
                features[f'{driver}_points_gap_to_leader'] = gap
                features[f'{driver}_total_points'] = driver_points[driver]
                
                if driver_positions[driver] > 0:
                    features[f'{driver}_avg_finish_pos'] = driver_positions[driver]
                
                features[f'{driver}_wins'] = driver_wins.get(driver, 0)
        
        return features

    def extract_all_features(self, historical_df):
        all_features = {}
        
        current_season_drivers = historical_df[
            historical_df['season'] == self.season
        ]['driver'].unique()
        
        for driver in current_season_drivers:
            driver_features = self.calculate_track_performance(driver, historical_df)
            all_features.update(driver_features)
        
        championship_features = self.calculate_championship_dynamics(historical_df)
        all_features.update(championship_features)
        
        return all_features

class F1AdvancedPredictor:
    def __init__(self):
        self.models = {
            'rf': RandomForestClassifier(n_estimators=300, random_state=42, max_depth=15),
            'gb': GradientBoostingClassifier(random_state=42, n_estimators=200),
            'lr': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        self.ensemble = VotingClassifier(
            estimators=[
                ('rf', self.models['rf']),
                ('gb', self.models['gb']),
                ('lr', self.models['lr'])
            ],
            voting='soft'
        )
        self.scaler = StandardScaler()
        self.trained = False
        
    def prepare_training_data(self, historical_df):
        season_champions = {}
        current_year = datetime.now().year
        if datetime.now().month > 11:
            training_seasons = [current_year - 1, current_year - 2]
        else:
            training_seasons = [current_year - 1, current_year - 2]
        
        X, y = [], []
        
        for season in training_seasons:
            season_data = historical_df[historical_df['season'] == season]
            if len(season_data) == 0:
                continue
                
            drivers = season_data['driver'].unique()
            
            features_for_season = {}
            for driver in drivers:
                driver_data = season_data[season_data['driver'] == driver]
                
                race_data = driver_data[driver_data['race_type'] == 'race']
                sprint_data = driver_data[driver_data['race_type'] == 'sprint']
                
                driver_features = {
                    'avg_finish_pos': race_data['position'].mean() if len(race_data) > 0 else 99,
                    'avg_grid_pos': race_data['grid'].mean() if len(race_data) > 0 else 99,
                    'total_points': race_data['points'].sum() if len(race_data) > 0 else 0,
                    'wins': len(race_data[race_data['position'] == 1]) if len(race_data) > 0 else 0,
                    'podiums': len(race_data[race_data['position'] <= 3]) if len(race_data) > 0 else 0,
                    'top5': len(race_data[race_data['position'] <= 5]) if len(race_data) > 0 else 0,
                    'top10': len(race_data[race_data['position'] <= 10]) if len(race_data) > 0 else 0,
                    'dnf_count': len(race_data[race_data['status'] == 'DSQ']) if len(race_data) > 0 else 0,
                    'pos_std': race_data['position'].std() if len(race_data) > 1 else 0,
                    'grid_to_finish_improvement': (race_data['grid'] - race_data['position']).mean() if len(race_data) > 0 else 0,
                    'sprint_avg_pos': sprint_data['position'].mean() if len(sprint_data) > 0 else 99,
                    'sprint_total_points': sprint_data['points'].sum() if len(sprint_data) > 0 else 0,
                    'sprint_wins': len(sprint_data[sprint_data['position'] == 1]) if len(sprint_data) > 0 else 0,
                    'recent_avg_pos': race_data.tail(3)['position'].mean() if len(race_data) >= 3 else race_data['position'].mean() if len(race_data) > 0 else 99,
                    'recent_points': race_data.tail(3)['points'].sum() if len(race_data) >= 3 else race_data['points'].sum() if len(race_data) > 0 else 0,
                    'total_races': len(driver_data)
                }
                
                features_for_season[driver] = driver_features
            
            for driver, feats in features_for_season.items():
                feature_vector = list(feats.values())
                X.append(feature_vector)
                
                champion = season_data.groupby('driver')['points'].sum().idxmax()
                label = 1 if driver == champion else 0
                y.append(label)
        
        if len(X) == 0:
            return None, None, None
        
        X = np.array(X)
        y = np.array(y)
        
        return X, y, list(features_for_season[list(features_for_season.keys())[0]].keys())

    def train_model(self, historical_df):
        X, y, feature_names = self.prepare_training_data(historical_df)
        
        if X is None or len(X) == 0:
            self.trained = False
            return
        
        if sum(y) == 0:
            self.trained = False
            return
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.ensemble.fit(X_train_scaled, y_train)
        
        y_pred = self.ensemble.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        if accuracy >= 0.75:
            self.trained = True
        else:
            self.trained = False

    def predict_champion(self, current_features, current_year):
        if not self.trained:
            return self.advanced_rule_based_prediction(current_features)
        
        drivers = []
        for key in current_features.keys():
            if key.endswith('_total_points'):
                driver = key.replace('_total_points', '')
                if '_' not in driver and len(driver) <= 3:
                    drivers.append(driver)
        
        drivers = list(set(drivers))
        
        predictions = {}
        for driver in drivers:
            points = current_features.get(f"{driver}_total_points", 0)
            avg_pos = current_features.get(f"{driver}_avg_finish_pos", 99)
            wins = current_features.get(f"{driver}_wins", 0)
            podiums = current_features.get(f"{driver}_podiums", 0)
            pos_std = current_features.get(f"{driver}_pos_std", 0)
            grid_improvement = current_features.get(f"{driver}_grid_to_finish_improvement", 0)
            recent_avg_pos = current_features.get(f"{driver}_recent_avg_pos", avg_pos)
            recent_points = current_features.get(f"{driver}_recent_points", 0)
            top10 = current_features.get(f"{driver}_top10", 0)
            
            base_score = points
            pos_bonus = max(0, (25 - min(avg_pos, 25)) * 0.8)
            consistency_bonus = max(0, (10 - min(pos_std, 10)) * 0.2)
            perf_bonus = wins * 5 + podiums * 2 + top10 * 0.5
            grid_bonus = max(0, grid_improvement * 0.5)
            recent_bonus = max(0, (25 - min(recent_avg_pos, 25)) * 0.3) + recent_points * 0.1
            
            total_score = base_score + pos_bonus + consistency_bonus + perf_bonus + grid_bonus + recent_bonus
            predictions[driver] = total_score
        
        total_score = sum(predictions.values())
        if total_score > 0:
            for driver in predictions:
                predictions[driver] /= total_score
        
        return predictions

    def advanced_rule_based_prediction(self, current_features):
        driver_scores = {}
        
        for key, value in current_features.items():
            if key.endswith('_total_points') and isinstance(value, (int, float)) and not pd.isna(value):
                driver = key.replace('_total_points', '')
                if '_' not in driver and len(driver) <= 3:
                    points = value
                    avg_pos = current_features.get(f"{driver}_avg_finish_pos", 99)
                    wins = current_features.get(f"{driver}_wins", 0)
                    podiums = current_features.get(f"{driver}_podiums", 0)
                    pos_std = current_features.get(f"{driver}_pos_std", 0)
                    grid_improvement = current_features.get(f"{driver}_grid_to_finish_improvement", 0)
                    recent_avg_pos = current_features.get(f"{driver}_recent_avg_pos", avg_pos)
                    recent_points = current_features.get(f"{driver}_recent_points", 0)
                    top10 = current_features.get(f"{driver}_top10", 0)
                    
                    base_score = points
                    pos_bonus = max(0, (25 - min(avg_pos, 25)) * 0.8)
                    consistency_bonus = max(0, (10 - min(pos_std, 10)) * 0.2)
                    perf_bonus = wins * 5 + podiums * 2 + top10 * 0.5
                    grid_bonus = max(0, grid_improvement * 0.5)
                    recent_bonus = max(0, (25 - min(recent_avg_pos, 25)) * 0.3) + recent_points * 0.1
                    
                    total_score = base_score + pos_bonus + consistency_bonus + perf_bonus + grid_bonus + recent_bonus
                    driver_scores[driver] = total_score
        
        leader_points = 0
        for key, value in current_features.items():
            if key.endswith('_total_points') and isinstance(value, (int, float)):
                if value > leader_points:
                    leader_points = value
        
        current_month = datetime.now().month
        if current_month >= 11:
            remaining_races = 3
            remaining_sprints = 1
        elif current_month >= 10:
            remaining_races = 5
            remaining_sprints = 2
        elif current_month >= 9:
            remaining_races = 7
            remaining_sprints = 2
        elif current_month >= 7:
            remaining_races = 10
            remaining_sprints = 4
        elif current_month >= 5:
            remaining_races = 16
            remaining_sprints = 6
        else:
            remaining_races = 22
            remaining_sprints = 8
        
        max_remaining_points = (remaining_races * 26) + (remaining_sprints * 8)
        
        for driver in driver_scores:
            current_points = current_features.get(f"{driver}_total_points", 0)
            theoretical_max = current_points + max_remaining_points
            
            if theoretical_max <= leader_points:
                driver_scores[driver] = 0.0
        
        total_score = sum(driver_scores.values())
        if total_score == 0:
            for driver in driver_scores:
                driver_scores[driver] = 1.0 / len(driver_scores) if len(driver_scores) > 0 else 0
        else:
            for driver in driver_scores:
                driver_scores[driver] /= total_score
        
        return driver_scores

class F1HistoricalDataCollector:
    def __init__(self):
        self.historical_data = []
    
    def collect_historical_data_current_year(self):
        current_year = datetime.now().year
        start_year = current_year - 2
        
        seasons = [start_year, start_year + 1, current_year]
        
        all_features = []
        
        for season in seasons:
            try:
                schedule = fastf1.get_event_schedule(season)
            except:
                continue
            
            for i in range(len(schedule)):
                try:
                    event = schedule.iloc[i]
                    if event['EventDate'] < datetime.now():
                        try:
                            race_session = fastf1.get_session(season, i+1, 'R')
                            race_session.load(laps=False, telemetry=False, weather=False, messages=False)
                            
                            for idx, driver_result in race_session.results.iterrows():
                                if pd.notna(driver_result['Position']):
                                    feature_row = {
                                        'season': season,
                                        'round': i+1,
                                        'driver': driver_result['Abbreviation'],
                                        'position': driver_result['Position'],
                                        'points': driver_result['Points'],
                                        'grid': driver_result['GridPosition'],
                                        'status': driver_result['Status'],
                                        'race_type': 'race'
                                    }
                                    all_features.append(feature_row)
                        except:
                            pass
                        
                        try:
                            sprint_session = fastf1.get_session(season, i+1, 'S')
                            sprint_session.load(laps=False, telemetry=False, weather=False, messages=False)
                            
                            for idx, driver_result in sprint_session.results.iterrows():
                                if pd.notna(driver_result['Position']):
                                    feature_row = {
                                        'season': season,
                                        'round': i+1,
                                        'driver': driver_result['Abbreviation'],
                                        'position': driver_result['Position'],
                                        'points': driver_result['Points'],
                                        'grid': driver_result['GridPosition'],
                                        'status': driver_result['Status'],
                                        'race_type': 'sprint'
                                    }
                                    all_features.append(feature_row)
                        except:
                            pass
                                
                except:
                    pass
        
        return pd.DataFrame(all_features)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict')
def get_predictions():
    current_year = datetime.now().year
    
    collector = F1HistoricalDataCollector()
    historical_df = collector.collect_historical_data_current_year()
    
    feature_engineer = F1AdvancedFeatureEngineer(season=current_year)
    current_features = feature_engineer.extract_all_features(historical_df)
    
    predictor = F1AdvancedPredictor()
    predictor.train_model(historical_df)
    
    champion_predictions = predictor.predict_champion(current_features, current_year)
    
    sorted_predictions = sorted(champion_predictions.items(), key=lambda x: x[1], reverse=True)
    
    all_drivers = ['VER', 'PER', 'HAM', 'RUS', 'LEC', 'SAI', 'NOR', 'PIA', 
                   'ALO', 'STR', 'OCO', 'GAS', 'BOT', 'ZHO', 'MAG', 'HUL', 
                   'TSU', 'DEV', 'ALB', 'SAR']
    
    existing_drivers = {driver for driver, prob in sorted_predictions}
    for driver in all_drivers:
        if driver not in existing_drivers:
            sorted_predictions.append((driver, 0.0))
    
    sorted_predictions = sorted(sorted_predictions, key=lambda x: x[1], reverse=True)[:20]
    
    active_drivers = [(driver, prob) for driver, prob in sorted_predictions if prob > 0]
    eliminated_drivers = [(driver, prob) for driver, prob in sorted_predictions if prob == 0]
    
    eliminated_drivers_sorted = sorted(eliminated_drivers, 
                                     key=lambda x: current_features.get(f"{x[0]}_total_points", 0), 
                                     reverse=True)
    
    # Get leader information
    leader_points = 0
    leader = ""
    for key, value in current_features.items():
        if key.endswith('_total_points') and isinstance(value, (int, float)):
            if value > leader_points:
                leader_points = value
                leader = key.replace('_total_points', '')
    
    result = {
        'current_year': current_year,
        'active_drivers': [
            {
                'driver': driver,
                'probability': round(prob * 100, 2),
                'points': int(current_features.get(f"{driver}_total_points", 0)),
                'avg_pos': round(current_features.get(f"{driver}_avg_finish_pos", 99), 1),
                'wins': int(current_features.get(f"{driver}_wins", 0)),
                'podiums': int(current_features.get(f"{driver}_podiums", 0))
            } for driver, prob in active_drivers
        ],
        'eliminated_drivers': [
            {
                'driver': driver,
                'probability': round(prob * 100, 2),
                'points': int(current_features.get(f"{driver}_total_points", 0)),
                'avg_pos': round(current_features.get(f"{driver}_avg_finish_pos", 99), 1),
                'wins': int(current_features.get(f"{driver}_wins", 0)),
                'podiums': int(current_features.get(f"{driver}_podiums", 0))
            } for driver, prob in eliminated_drivers_sorted
        ],
        'leader': {
            'driver': leader,
            'points': int(leader_points)
        },
        'total_active': len(active_drivers),
        'total_eliminated': len(eliminated_drivers_sorted)
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)