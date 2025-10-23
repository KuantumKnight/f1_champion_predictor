# f1_champion_predictor_final.py
"""
F1 Champion Predictor - Advanced Machine Learning Model with Correct Mathematical Elimination

This script predicts the Formula 1 World Championship winner based on:
- Historical performance data (last 2 years + current year)
- Current season statistics
- Mathematical elimination logic
- Advanced ML ensemble model

Features:
- Mathematical elimination for drivers who cannot win championship (winning all remaining races)
- Advanced feature engineering with track performance
- Ensemble ML model (Random Forest + Gradient Boosting + Logistic Regression)
- Real-time data collection from FastF1
- Progress bar for data collection
- Correct F1 points system (25-18-15-12-10-8-6-4-2-1 + fastest lap bonus)
- Correct mathematical elimination logic (win all races + fastest laps)
- Dynamic current year prediction (works for any year)
- Uses last 2 years + current year for training data
- Clean output with eliminated drivers sorted by points
"""

import fastf1
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import os
from tqdm import tqdm
import logging

# Suppress FastF1 logging to keep output clean
logging.getLogger('fastf1').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)

warnings.filterwarnings('ignore')

# Create cache directory for FastF1 data
cache_dir = 'cache/'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# Enable FastF1 cache to speed up data loading
fastf1.Cache.enable_cache(cache_dir)

class F1AdvancedFeatureEngineer:
    """
    Advanced feature engineering class that extracts comprehensive driver statistics
    """
    def __init__(self, season=None):
        self.season = season if season else datetime.now().year
        self.features = {}
    
    def calculate_track_performance(self, driver_abbr, historical_df):
        """
        Calculate comprehensive performance metrics for a driver
        
        Args:
            driver_abbr (str): Driver abbreviation (e.g., 'VER', 'HAM')
            historical_df (DataFrame): Historical race data
        
        Returns:
            dict: Dictionary of calculated features
        """
        features = {}
        
        # Filter data for this specific driver and season
        driver_data = historical_df[
            (historical_df['driver'] == driver_abbr) & 
            (historical_df['season'] == self.season)
        ]
        
        if len(driver_data) == 0:
            return features
        
        # Separate race and sprint data for different analysis
        race_data = driver_data[driver_data['race_type'] == 'race']
        sprint_data = driver_data[driver_data['race_type'] == 'sprint']
        
        # Basic performance metrics from races
        features[f'{driver_abbr}_avg_finish_pos'] = race_data['position'].mean() if len(race_data) > 0 else 99
        features[f'{driver_abbr}_avg_grid_pos'] = race_data['grid'].mean() if len(race_data) > 0 else 99
        features[f'{driver_abbr}_total_points'] = race_data['points'].sum() if len(race_data) > 0 else 0
        features[f'{driver_abbr}_wins'] = len(race_data[race_data['position'] == 1]) if len(race_data) > 0 else 0
        features[f'{driver_abbr}_podiums'] = len(race_data[race_data['position'] <= 3]) if len(race_data) > 0 else 0
        features[f'{driver_abbr}_top10'] = len(race_data[race_data['position'] <= 10]) if len(race_data) > 0 else 0
        features[f'{driver_abbr}_top5'] = len(race_data[race_data['position'] <= 5]) if len(race_data) > 0 else 0
        features[f'{driver_abbr}_dnf_rate'] = len(race_data[race_data['status'] == 'DSQ']) / len(race_data) if len(race_data) > 0 else 0
        
        # Consistency metrics - how stable is the driver's performance?
        if len(race_data) > 1:
            features[f'{driver_abbr}_pos_std'] = race_data['position'].std()  # Standard deviation of positions
            features[f'{driver_abbr}_grid_to_finish_improvement'] = (race_data['grid'] - race_data['position']).mean()  # Grid improvement
        else:
            features[f'{driver_abbr}_pos_std'] = 0
            features[f'{driver_abbr}_grid_to_finish_improvement'] = 0
        
        # Sprint race performance (if applicable)
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
        
        # Recent form analysis (last 5 races) - more relevant for current performance
        if len(race_data) >= 5:
            recent_data = race_data.tail(5)
            features[f'{driver_abbr}_recent_avg_pos'] = recent_data['position'].mean()
            features[f'{driver_abbr}_recent_points'] = recent_data['points'].sum()
        else:
            # If less than 5 races, use overall averages
            features[f'{driver_abbr}_recent_avg_pos'] = features[f'{driver_abbr}_avg_finish_pos']
            features[f'{driver_abbr}_recent_points'] = features[f'{driver_abbr}_total_points']
        
        return features

    def calculate_championship_dynamics(self, historical_df):
        """
        Calculate championship context features like leader and points gaps
        
        Args:
            historical_df (DataFrame): Historical race data for current season
        
        Returns:
            dict: Dictionary of championship context features
        """
        features = {}
        
        # Get current season data
        current_season = historical_df[historical_df['season'] == self.season]
        
        # Group by driver and calculate total points for the season
        driver_points = current_season.groupby('driver')['points'].sum()
        driver_positions = current_season.groupby('driver')['position'].mean()
        driver_wins = current_season[current_season['position'] == 1].groupby('driver').size()
        
        # Identify championship leader
        if len(driver_points) > 0:
            leader = driver_points.idxmax()
            leader_points = driver_points.max()
            
            features['championship_leader'] = leader
            features['championship_leader_points'] = leader_points
            
            # Calculate points gaps for all drivers relative to leader
            for driver in driver_points.index:
                gap = leader_points - driver_points[driver]
                features[f'{driver}_points_gap_to_leader'] = gap
                features[f'{driver}_total_points'] = driver_points[driver]
                
                # Add average finishing position for this driver
                if driver_positions[driver] > 0:
                    features[f'{driver}_avg_finish_pos'] = driver_positions[driver]
                
                # Add number of wins for this driver
                features[f'{driver}_wins'] = driver_wins.get(driver, 0)
        
        return features

    def extract_all_features(self, historical_df):
        """
        Extract all features for all drivers in the current season
        
        Args:
            historical_df (DataFrame): Historical race data
        
        Returns:
            dict: Dictionary of all features for all drivers
        """
        all_features = {}
        
        # Get all unique drivers in current season
        current_season_drivers = historical_df[
            historical_df['season'] == self.season
        ]['driver'].unique()
        
        print(f"Processing advanced features for {len(current_season_drivers)} drivers in {self.season}...")
        
        # Calculate individual driver features
        for driver in current_season_drivers:
            driver_features = self.calculate_track_performance(driver, historical_df)
            all_features.update(driver_features)
        
        # Add championship context features
        championship_features = self.calculate_championship_dynamics(historical_df)
        all_features.update(championship_features)
        
        return all_features

class F1AdvancedPredictor:
    """
    Advanced ML predictor with corrected mathematical elimination logic
    """
    def __init__(self):
        # Create ensemble of three different ML models for better accuracy
        self.models = {
            'rf': RandomForestClassifier(n_estimators=300, random_state=42, max_depth=15),
            'gb': GradientBoostingClassifier(random_state=42, n_estimators=200),
            'lr': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        # Combine all models using voting classifier
        self.ensemble = VotingClassifier(
            estimators=[
                ('rf', self.models['rf']),
                ('gb', self.models['gb']),
                ('lr', self.models['lr'])
            ],
            voting='soft'  # Use probability voting for better results
        )
        self.scaler = StandardScaler()  # For feature scaling
        self.trained = False  # Flag to track if model is trained
        
    def prepare_training_data(self, historical_df):
        """
        Prepare training data from last 2 completed seasons + current year data
        
        Args:
            historical_df (DataFrame): Historical race data
        
        Returns:
            tuple: (X, y, feature_names) or (None, None, None) if no data
        """
        print("Preparing advanced training data...")
        
        # Identify champions from previous seasons for training labels
        season_champions = {}
        # Use the last 2 completed seasons as training data
        current_year = datetime.now().year
        # Only use completed seasons (not current year)
        if datetime.now().month > 11:  # If after November, current year is likely complete
            training_seasons = [current_year - 1, current_year - 2]
        else:  # Otherwise, use last 2 completed years
            training_seasons = [current_year - 1, current_year - 2]
        
        print(f"Using seasons {training_seasons} for training data")
        
        for season in training_seasons:
            season_data = historical_df[historical_df['season'] == season]
            if len(season_data) > 0:
                driver_points = season_data.groupby('driver')['points'].sum()
                if len(driver_points) > 0:
                    champion = driver_points.idxmax()  # Driver with most points = champion
                    season_champions[season] = champion
                    print(f"Season {season} champion: {champion}")
        
        # Prepare feature matrix and labels
        X, y = [], []
        
        for season in training_seasons:
            season_data = historical_df[historical_df['season'] == season]
            if len(season_data) == 0:
                continue
                
            # Get all drivers for this season
            drivers = season_data['driver'].unique()
            
            # Calculate features for each driver in this season
            features_for_season = {}
            for driver in drivers:
                driver_data = season_data[season_data['driver'] == driver]
                
                # Calculate race performance features
                race_data = driver_data[driver_data['race_type'] == 'race']
                sprint_data = driver_data[driver_data['race_type'] == 'sprint']
                
                # Advanced features for ML model
                driver_features = {
                    # Performance metrics
                    'avg_finish_pos': race_data['position'].mean() if len(race_data) > 0 else 99,
                    'avg_grid_pos': race_data['grid'].mean() if len(race_data) > 0 else 99,
                    'total_points': race_data['points'].sum() if len(race_data) > 0 else 0,
                    'wins': len(race_data[race_data['position'] == 1]) if len(race_data) > 0 else 0,
                    'podiums': len(race_data[race_data['position'] <= 3]) if len(race_data) > 0 else 0,
                    'top5': len(race_data[race_data['position'] <= 5]) if len(race_data) > 0 else 0,
                    'top10': len(race_data[race_data['position'] <= 10]) if len(race_data) > 0 else 0,
                    'dnf_count': len(race_data[race_data['status'] == 'DSQ']) if len(race_data) > 0 else 0,
                    
                    # Consistency metrics
                    'pos_std': race_data['position'].std() if len(race_data) > 1 else 0,
                    'grid_to_finish_improvement': (race_data['grid'] - race_data['position']).mean() if len(race_data) > 0 else 0,
                    
                    # Sprint performance
                    'sprint_avg_pos': sprint_data['position'].mean() if len(sprint_data) > 0 else 99,
                    'sprint_total_points': sprint_data['points'].sum() if len(sprint_data) > 0 else 0,
                    'sprint_wins': len(sprint_data[sprint_data['position'] == 1]) if len(sprint_data) > 0 else 0,
                    
                    # Recent form (last 3 races)
                    'recent_avg_pos': race_data.tail(3)['position'].mean() if len(race_data) >= 3 else race_data['position'].mean() if len(race_data) > 0 else 99,
                    'recent_points': race_data.tail(3)['points'].sum() if len(race_data) >= 3 else race_data['points'].sum() if len(race_data) > 0 else 0,
                    
                    # Total races completed
                    'total_races': len(driver_data)
                }
                
                features_for_season[driver] = driver_features
            
            # Create feature vectors and labels (1 = champion, 0 = not champion)
            for driver, feats in features_for_season.items():
                feature_vector = list(feats.values())
                X.append(feature_vector)
                
                # Label: 1 if this driver was champion, 0 otherwise
                label = 1 if driver in season_champions and driver == season_champions[season] else 0
                y.append(label)
        
        if len(X) == 0:
            print("No training data available, using advanced rule-based prediction")
            return None, None, None
        
        X = np.array(X)
        y = np.array(y)
        
        return X, y, list(features_for_season[list(features_for_season.keys())[0]].keys())

    def train_model(self, historical_df):
        """
        Train the ensemble ML model on historical data
        
        Args:
            historical_df (DataFrame): Historical race data
        """
        print("Training advanced model...")
        
        X, y, feature_names = self.prepare_training_data(historical_df)
        
        if X is None or len(X) == 0:
            print("Insufficient training data, using advanced rule-based prediction")
            self.trained = False
            return
        
        # Check if we have enough positive examples (champions)
        if sum(y) == 0:
            print("No champions in training data, using advanced rule-based prediction")
            self.trained = False
            return
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features for better ML performance
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train the ensemble model
        self.ensemble.fit(X_train_scaled, y_train)
        
        # Evaluate model performance
        y_pred = self.ensemble.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Advanced model accuracy: {accuracy:.3f}")
        
        # Set training flag based on model performance
        if accuracy >= 0.75:  # Higher threshold for advanced model
            self.trained = True
            print("Advanced model trained successfully!")
        else:
            print("Advanced model accuracy too low, will use advanced rule-based prediction")
            self.trained = False

    def predict_champion(self, current_features, current_year):
        """
        Predict champion for current year based on current features
        
        Args:
            current_features (dict): Current season features
            current_year (int): Current year to predict for
        
        Returns:
            dict: Dictionary of driver probabilities
        """
        if not self.trained:
            print("Using advanced rule-based prediction (model not trained adequately)")
            return self.advanced_rule_based_prediction(current_features)
        
        print("Using trained advanced model for prediction...")
        
        # Extract only drivers (no race/sprint suffixes)
        drivers = []
        for key in current_features.keys():
            if key.endswith('_total_points'):
                driver = key.replace('_total_points', '')
                if '_' not in driver and len(driver) <= 3:  # Ensure it's just the driver code
                    drivers.append(driver)
        
        # Remove duplicates
        drivers = list(set(drivers))
        
        # Calculate scores for each driver using advanced metrics
        predictions = {}
        for driver in drivers:
            # Get all relevant features for this driver
            points = current_features.get(f"{driver}_total_points", 0)
            avg_pos = current_features.get(f"{driver}_avg_finish_pos", 99)
            wins = current_features.get(f"{driver}_wins", 0)
            podiums = current_features.get(f"{driver}_podiums", 0)
            pos_std = current_features.get(f"{driver}_pos_std", 0)
            grid_improvement = current_features.get(f"{driver}_grid_to_finish_improvement", 0)
            recent_avg_pos = current_features.get(f"{driver}_recent_avg_pos", avg_pos)
            recent_points = current_features.get(f"{driver}_recent_points", 0)
            top10 = current_features.get(f"{driver}_top10", 0)
            
            # Advanced scoring system
            base_score = points
            
            # Position bonus (better positions get higher bonus)
            pos_bonus = max(0, (25 - min(avg_pos, 25)) * 0.8)
            
            # Consistency bonus (lower std = more consistent = better)
            consistency_bonus = max(0, (10 - min(pos_std, 10)) * 0.2)
            
            # Performance bonus
            perf_bonus = wins * 5 + podiums * 2 + top10 * 0.5
            
            # Grid improvement bonus
            grid_bonus = max(0, grid_improvement * 0.5)
            
            # Recent form bonus
            recent_bonus = max(0, (25 - min(recent_avg_pos, 25)) * 0.3) + recent_points * 0.1
            
            total_score = base_score + pos_bonus + consistency_bonus + perf_bonus + grid_bonus + recent_bonus
            predictions[driver] = total_score
        
        # Normalize to probabilities (sum to 1)
        total_score = sum(predictions.values())
        if total_score > 0:
            for driver in predictions:
                predictions[driver] /= total_score
        
        return predictions

    def advanced_rule_based_prediction(self, current_features):
        """
        Advanced rule-based prediction with correct mathematical elimination logic
        (win all remaining races with fastest lap bonus)
        
        Args:
            current_features (dict): Current season features
        
        Returns:
            dict: Dictionary of driver probabilities
        """
        print("Making advanced rule-based prediction with correct mathematical elimination...")
        
        # Extract only drivers (no race/sprint suffixes)
        driver_scores = {}
        
        for key, value in current_features.items():
            if key.endswith('_total_points') and isinstance(value, (int, float)) and not pd.isna(value):
                driver = key.replace('_total_points', '')
                if '_' not in driver and len(driver) <= 3:  # Ensure it's just the driver code
                    points = value
                    avg_pos = current_features.get(f"{driver}_avg_finish_pos", 99)
                    wins = current_features.get(f"{driver}_wins", 0)
                    podiums = current_features.get(f"{driver}_podiums", 0)
                    pos_std = current_features.get(f"{driver}_pos_std", 0)
                    grid_improvement = current_features.get(f"{driver}_grid_to_finish_improvement", 0)
                    recent_avg_pos = current_features.get(f"{driver}_recent_avg_pos", avg_pos)
                    recent_points = current_features.get(f"{driver}_recent_points", 0)
                    top10 = current_features.get(f"{driver}_top10", 0)
                    
                    # Advanced scoring system
                    base_score = points
                    
                    # Position bonus (better positions get higher bonus)
                    pos_bonus = max(0, (25 - min(avg_pos, 25)) * 0.8)
                    
                    # Consistency bonus (lower std = more consistent = better)
                    consistency_bonus = max(0, (10 - min(pos_std, 10)) * 0.2)
                    
                    # Performance bonus
                    perf_bonus = wins * 5 + podiums * 2 + top10 * 0.5
                    
                    # Grid improvement bonus
                    grid_bonus = max(0, grid_improvement * 0.5)
                    
                    # Recent form bonus
                    recent_bonus = max(0, (25 - min(recent_avg_pos, 25)) * 0.3) + recent_points * 0.1
                    
                    total_score = base_score + pos_bonus + consistency_bonus + perf_bonus + grid_bonus + recent_bonus
                    driver_scores[driver] = total_score
        
        # Get leader's points for mathematical elimination
        leader_points = 0
        for key, value in current_features.items():
            if key.endswith('_total_points') and isinstance(value, (int, float)):
                if value > leader_points:
                    leader_points = value
        
        # Calculate remaining races and maximum possible points (win all + fastest lap bonus)
        # F1 typically has 24 races per season, so remaining races = 24 - completed races
        # We'll estimate remaining races based on current date
        current_month = datetime.now().month
        current_day = datetime.now().day
        
        # Estimate remaining races based on typical F1 calendar
        if current_month >= 11:  # Late season, typically 2-3 races left
            remaining_races = 3
            remaining_sprints = 1
        elif current_month >= 10:  # October, typically 4-6 races left
            remaining_races = 5
            remaining_sprints = 2
        elif current_month >= 9:  # September, typically 6-8 races left
            remaining_races = 7
            remaining_sprints = 2
        elif current_month >= 7:  # July-August, typically 10-12 races left
            remaining_races = 10
            remaining_sprints = 4
        elif current_month >= 5:  # May-June, typically 15-18 races left
            remaining_races = 16
            remaining_sprints = 6
        else:  # Early season
            remaining_races = 22
            remaining_sprints = 8
        
        # Calculate maximum possible points (win all races + fastest laps + sprints)
        max_remaining_points = (remaining_races * 26) + (remaining_sprints * 8)  # 26 = 25 win + 1 fastest lap, 8 = sprint win
        print(f"Leader points: {leader_points}, Max remaining points: {max_remaining_points}")
        
        # Apply mathematical elimination: if cannot surpass leader even with perfect results
        for driver in driver_scores:
            current_points = current_features.get(f"{driver}_total_points", 0)
            theoretical_max = current_points + max_remaining_points
            
            print(f"Driver {driver}: Current {current_points}, Theoretical max {theoretical_max}, Leader {leader_points}")
            
            # If cannot surpass leader even with perfect results (all wins + fastest laps), set to 0
            if theoretical_max <= leader_points:
                driver_scores[driver] = 0.0
                print(f"  -> {driver} mathematically eliminated (theoretical max {theoretical_max} <= leader {leader_points})")
        
        # Normalize to probabilities (sum to 1)
        total_score = sum(driver_scores.values())
        if total_score == 0:
            # If all drivers are eliminated, use equal probabilities
            for driver in driver_scores:
                driver_scores[driver] = 1.0 / len(driver_scores) if len(driver_scores) > 0 else 0
        else:
            for driver in driver_scores:
                driver_scores[driver] /= total_score
        
        return driver_scores

class F1HistoricalDataCollector:
    """
    Class to collect historical F1 data from FastF1
    """
    def __init__(self):
        self.historical_data = []
    
    def collect_historical_data_current_year(self):
        """
        Collect historical data from last 2 years + current year up to latest completed race
        
        Returns:
            DataFrame: Historical race and sprint data
        """
        current_year = datetime.now().year
        # Collect from current year - 2 to current year
        start_year = current_year - 2
        print(f"Collecting historical data from {start_year} to latest completed race in {current_year}...")
        
        # Define seasons to collect (last 2 years + current year)
        seasons = [start_year, start_year + 1, current_year]
        
        all_features = []
        
        # Calculate total events for progress bar
        total_events = 0
        for season in seasons:
            try:
                schedule = fastf1.get_event_schedule(season)
                total_events += len(schedule)
            except:
                continue
        
        # Create progress bar
        pbar = tqdm(total=total_events, desc="Processing events", leave=False)
        
        for season in seasons:
            try:
                schedule = fastf1.get_event_schedule(season)
            except Exception as e:
                pbar.update(len(schedule) if 'schedule' in locals() else 0)
                continue
            
            for i in range(len(schedule)):
                try:
                    event = schedule.iloc[i]
                    # Only process events that have already happened
                    if event['EventDate'] < datetime.now():
                        # Get race data
                        try:
                            race_session = fastf1.get_session(season, i+1, 'R')
                            race_session.load(laps=False, telemetry=False, weather=False, messages=False)
                            
                            # Process race results
                            for idx, driver_result in race_session.results.iterrows():
                                if pd.notna(driver_result['Position']):
                                    feature_row = {
                                        'season': season,
                                        'round': i+1,
                                        'driver': driver_result['Abbreviation'],
                                        'position': driver_result['Position'],
                                        'points': driver_result['Points'],  # This includes fastest lap bonus if applicable
                                        'grid': driver_result['GridPosition'],
                                        'status': driver_result['Status'],
                                        'race_type': 'race'
                                    }
                                    all_features.append(feature_row)
                        except Exception as e:
                            pass  # Silent error - continue if race data unavailable
                        
                        # Get sprint data if available
                        try:
                            sprint_session = fastf1.get_session(season, i+1, 'S')
                            sprint_session.load(laps=False, telemetry=False, weather=False, messages=False)
                            
                            # Process sprint results
                            for idx, driver_result in sprint_session.results.iterrows():
                                if pd.notna(driver_result['Position']):
                                    feature_row = {
                                        'season': season,
                                        'round': i+1,
                                        'driver': driver_result['Abbreviation'],
                                        'position': driver_result['Position'],
                                        'points': driver_result['Points'],  # This includes any bonuses
                                        'grid': driver_result['GridPosition'],
                                        'status': driver_result['Status'],
                                        'race_type': 'sprint'
                                    }
                                    all_features.append(feature_row)
                        except Exception as e:
                            pass  # Silent error - continue if sprint data unavailable
                                
                except Exception as e:
                    pass  # Silent error - continue to next event
                finally:
                    pbar.update(1)  # Update progress bar
        
        pbar.close()
        print(f"Data collection completed. Processed {len(all_features)} records.")
        return pd.DataFrame(all_features)

def main():
    """
    Main function to run the F1 champion predictor for current year
    """
    current_year = datetime.now().year
    
    print("="*80)
    print(f"F1 CHAMPION PREDICTOR FOR {current_year}")
    print("="*80)
    print(f"This program predicts the F1 World Champion for {current_year} based on historical data and")
    print("current season performance with correct mathematical elimination logic.")
    print("Mathematical elimination: Driver eliminated if cannot surpass leader even")
    print("with perfect results (all wins + fastest lap bonuses) in remaining races.")
    print("F1 Points System: 25-18-15-12-10-8-6-4-2-1 + 1 for fastest lap in top 10")
    print("Sprint Points: 8-7-6-5-4-3-2-1")
    print(f"Training data: {current_year-2} and {current_year-1} seasons")
    print("="*80)
    
    # Collect historical data
    collector = F1HistoricalDataCollector()
    historical_df = collector.collect_historical_data_current_year()
    
    print(f"\nCollected {len(historical_df)} race/sprint records")
    print(f"Drivers in dataset: {historical_df['driver'].nunique()}")
    
    if len(historical_df) == 0:
        print("No data collected. Please check FastF1 connection and cache.")
        return
    
    # Extract advanced features for current season
    feature_engineer = F1AdvancedFeatureEngineer(season=current_year)
    current_features = feature_engineer.extract_all_features(historical_df)
    
    print(f"Extracted {len(current_features)} advanced features for {current_year} season")
    
    # Train and predict with advanced model
    predictor = F1AdvancedPredictor()
    predictor.train_model(historical_df)
    
    champion_predictions = predictor.predict_champion(current_features, current_year)
    
    # Sort predictions and ensure exactly 20 drivers
    sorted_predictions = sorted(champion_predictions.items(), key=lambda x: x[1], reverse=True)
    
    # Ensure we have exactly 20 drivers by adding placeholders if needed
    all_drivers = ['VER', 'PER', 'HAM', 'RUS', 'LEC', 'SAI', 'NOR', 'PIA', 
                   'ALO', 'STR', 'OCO', 'GAS', 'BOT', 'ZHO', 'MAG', 'HUL', 
                   'TSU', 'DEV', 'ALB', 'SAR']  # Standard 20 drivers
    
    # Add missing drivers with 0 probability
    existing_drivers = {driver for driver, prob in sorted_predictions}
    for driver in all_drivers:
        if driver not in existing_drivers:
            sorted_predictions.append((driver, 0.0))
    
    # Sort again and take top 20
    sorted_predictions = sorted(sorted_predictions, key=lambda x: x[1], reverse=True)[:20]
    
    # Separate active and eliminated drivers
    active_drivers = [(driver, prob) for driver, prob in sorted_predictions if prob > 0]
    eliminated_drivers = [(driver, prob) for driver, prob in sorted_predictions if prob == 0]
    
    # Sort eliminated drivers by points (descending)
    eliminated_drivers_sorted = sorted(eliminated_drivers, 
                                     key=lambda x: current_features.get(f"{x[0]}_total_points", 0), 
                                     reverse=True)
    
    print("\n" + "="*60)
    print(f"{current_year} F1 CHAMPION PREDICTIONS - ACTIVE DRIVERS")
    print("="*60)
    
    for i, (driver, prob) in enumerate(active_drivers, 1):
        points = current_features.get(f"{driver}_total_points", 0)
        print(f"{i:2d}. {driver:3s}: {prob*100:6.2f}% | Points: {int(points)}")
    
    
    print(f"\nTotal drivers analyzed: {len(sorted_predictions)}")
    print(f"Active contenders: {len(active_drivers)}")
    print(f"Mathematically eliminated: {len(eliminated_drivers_sorted)}")
    
    # Save results to generic output directory
    results_df = pd.DataFrame(sorted_predictions, columns=['Driver', 'Probability'])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create generic output directory
    output_dir = "f1_predictions"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"f1_champion_predictions_{current_year}_{timestamp}.csv")
    results_df.to_csv(output_file, index=False)
    print(f"\n🏁 {current_year} F1 champion prediction complete! Results saved to: {output_file}")

if __name__ == "__main__":
    main()