import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
house = pd.read_csv('Cleaned_Bengaluru_hpp.csv')
pipe = pickle.load(open('Linearmodel.pkl', 'rb'))

@app.route('/')
def index():
        locations = sorted(house['location'].unique())
        return render_template('index.html', locations=locations)


@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = float(request.form.get('bhk'))
    bath = float(request.form.get('bath'))
    sqft = float(request.form.get('total_sqft'))
    print(location, bhk, bath, sqft)
    input = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
    prediction = pipe.predict(input)[0] * 1e5
    
    return str(np.round(prediction, 2))
    

if __name__ == '__main__' :
    app.run(debug = True)