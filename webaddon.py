from flask import Flask, render_template, jsonify
import subprocess
import sys
import threading
import queue
import time
import json
import re

app = Flask(__name__)

# Global variables for simulation state
simulation_output = []
simulation_progress = 0
simulation_running = False
output_queue = queue.Queue()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/start_simulation')
def start_simulation():
    global simulation_output, simulation_progress, simulation_running, output_queue
    
    # Reset state
    simulation_output = []
    simulation_progress = 0
    simulation_running = True
    output_queue = queue.Queue()
    
    # Start simulation in a separate thread
    simulation_thread = threading.Thread(target=run_simulation)
    simulation_thread.daemon = True
    simulation_thread.start()
    
    return jsonify({"status": "started"})

@app.route('/api/simulation_progress')
def get_simulation_progress():
    global simulation_progress, simulation_running
    return jsonify({
        "progress": simulation_progress,
        "running": simulation_running
    })

@app.route('/api/simulation_output')
def get_simulation_output():
    global simulation_output
    # Return only the latest lines to avoid overwhelming the frontend
    return jsonify({"output": simulation_output[-50:]})  # Last 50 lines

@app.route('/api/simulation_result')
def get_simulation_result():
    global simulation_output, simulation_running
    
    if not simulation_running and simulation_output:
        # Parse the final output to extract predictions
        output_text = "\n".join(simulation_output)
        
        # Extract current year
        year_match = re.search(r'F1 CHAMPION PREDICTOR FOR (\d{4})', output_text)
        current_year = int(year_match.group(1)) if year_match else 2025
        
        # Extract active drivers
        active_drivers = []
        active_pattern = r'(\d+)\.\s+([A-Z]{3}):\s+([\d.]+)%\s+\|\s+Points:\s+(\d+)'
        active_matches = re.findall(active_pattern, output_text)
        
        for match in active_matches:
            position = int(match[0])
            driver = match[1]
            probability = float(match[2])
            points = int(match[3])
            
            # Extract additional stats for this driver
            avg_pos = 0
            wins = 0
            podiums = 0
            
            # Try to find stats in output
            stats_pattern = rf'{driver}:.*?Avg:\s*([\d.]+).*?W:\s*(\d+).*?P:\s*(\d+)'
            stats_match = re.search(stats_pattern, output_text)
            if stats_match:
                avg_pos = float(stats_match.group(1))
                wins = int(stats_match.group(2))
                podiums = int(stats_match.group(3))
            
            active_drivers.append({
                'driver': driver,
                'probability': probability,
                'points': points,
                'avg_pos': avg_pos,
                'wins': wins,
                'podiums': podiums
            })
        
        # Extract eliminated drivers
        eliminated_drivers = []
        eliminated_pattern = r'->\s+([A-Z]{3})\s+mathematically eliminated'
        eliminated_matches = re.findall(eliminated_pattern, output_text)
        
        # Get points for eliminated drivers
        for driver in eliminated_matches:
            points_pattern = rf'{driver}:\s+Current\s+([\d.]+)'
            points_match = re.search(points_pattern, output_text)
            points = float(points_match.group(1)) if points_match else 0
            
            # Extract additional stats for this driver
            avg_pos = 0
            wins = 0
            podiums = 0
            
            # Try to find stats in output
            stats_pattern = rf'{driver}:.*?Avg:\s*([\d.]+).*?W:\s*(\d+).*?P:\s*(\d+)'
            stats_match = re.search(stats_pattern, output_text)
            if stats_match:
                avg_pos = float(stats_match.group(1))
                wins = int(stats_match.group(2))
                podiums = int(stats_match.group(3))
            
            eliminated_drivers.append({
                'driver': driver,
                'probability': 0,
                'points': int(points),
                'avg_pos': avg_pos,
                'wins': wins,
                'podiums': podiums
            })
        
        # Get leader information
        leader_match = re.search(r'Leader points: ([\d.]+)', output_text)
        leader_points = float(leader_match.group(1)) if leader_match else 0
        
        # Find leader driver (highest points)
        leader_driver = ""
        if active_drivers:
            leader_driver = max(active_drivers, key=lambda x: x['points'])['driver']
        elif eliminated_drivers:
            leader_driver = max(eliminated_drivers, key=lambda x: x['points'])['driver']
        
        # Count totals
        total_active = len(active_drivers)
        total_eliminated = len(eliminated_drivers)
        
        data = {
            'current_year': current_year,
            'active_drivers': active_drivers,
            'eliminated_drivers': eliminated_drivers,
            'leader': {
                'driver': leader_driver,
                'points': int(leader_points)
            },
            'total_active': total_active,
            'total_eliminated': total_eliminated
        }
        
        return jsonify(data)
    
    return jsonify({"status": "waiting"})

def run_simulation():
    global simulation_output, simulation_progress, simulation_running
    
    try:
        # Run the F1 predictor and capture output
        process = subprocess.Popen(
            [sys.executable, 'f1_champion_predictor_final.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Read output line by line
        for line in process.stdout:
            line = line.rstrip()
            simulation_output.append(line)
            
            # Update progress based on keywords
            if "Processing events:" in line:
                # Extract progress percentage from line like "Processing events:  88%|#########1| 67/73 [00:38<00:03,  1.78it/s]"
                progress_match = re.search(r'(\d+)%', line)
                if progress_match:
                    simulation_progress = int(progress_match.group(1))
            elif "Data collection completed" in line:
                simulation_progress = 90
            elif "Training advanced model" in line:
                simulation_progress = 95
            elif "F1 champion prediction complete!" in line:
                simulation_progress = 100
        
        process.wait()
        
    except Exception as e:
        simulation_output.append(f"ERROR: {str(e)}")
    finally:
        simulation_running = False

if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=True)