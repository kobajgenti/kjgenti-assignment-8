from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from logistic_regression import do_experiments

app = Flask(__name__, static_url_path='/static', static_folder='static')

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_experiment', methods=['POST'])
def run_experiment():
    try:
        start = float(request.json['start'])
        end = float(request.json['end'])
        step_num = int(request.json['step_num'])

        # Run the experiment
        do_experiments(start, end, step_num)

        # Check if result images exist
        dataset_img = "results/dataset.png"
        parameters_img = "results/parameters_vs_shift_distance.png"
        
        return jsonify({
            "dataset_img": dataset_img if os.path.exists(dataset_img) else None,
            "parameters_img": parameters_img if os.path.exists(parameters_img) else None,
            "success": True
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/results/<filename>')
def results(filename):
    return send_from_directory('results', filename)

if __name__ == '__main__':
    app.run(debug=True, port=3000)