import pickle
from flask import Flask, request, jsonify, render_template
import logging

application = Flask(__name__)
app = application

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load the model and scaler
try:
    logger.debug("Loading ridge model...")
    ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
    logger.debug("Ridge model loaded successfully.")

    logger.debug("Loading scaler...")
    standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))
    logger.debug("Scaler loaded successfully.")

    # Debug print for types
    print(f"Type of standard_scaler: {type(standard_scaler)}")
    print(f"Type of ridge_model: {type(ridge_model)}")
except Exception as e:
    logger.error(f"Error loading model or scaler: {e}")
    raise


@app.route("/")
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            # Collect the input data from the form
            temperature = float(request.form.get('Temperature'))
            rh = float(request.form.get('RH'))
            wind_speed = float(request.form.get('Ws'))
            rain = float(request.form.get('Rain'))
            ffmc = float(request.form.get('FFMC'))
            dmc = float(request.form.get('DMC'))
            isi = float(request.form.get('ISI'))
            classes = int(request.form.get('Classes'))
            region = int(request.form.get('Region'))

            # Combine inputs into a list
            input_data = [temperature, rh, wind_speed, rain, ffmc, dmc, isi, classes, region]

            # Log the input data
            app.logger.info(f"Input data: {input_data}")

            print(f"Type of standard_scaler: {type(standard_scaler)}")


            # Scale the input data
            scaled_data = standard_scaler.transform([input_data])  # Transform data for prediction

            # Make a prediction
            prediction = ridge_model.predict(scaled_data)
            

            # Log the prediction
            app.logger.info(f"Prediction: {prediction}")

            return render_template('result.html', prediction=round(prediction[0], 4))


        except Exception as e:
            # Log and return the error if something goes wrong
            app.logger.error(f"Error during prediction: {str(e)}")
            return jsonify({"error": str(e)})
    else:
        # Render the home page for GET requests
        return render_template('home.html')


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
