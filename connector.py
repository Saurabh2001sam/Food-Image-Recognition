from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
from predictor import predict_cal

app = Flask(__name__,template_folder=".")



@app.route('/')
def index():
    return render_template('home_page.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'result': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'result': 'No selected file'})

    result = predict_cal(file)

    return jsonify({'result': result})

@app.route('/cal.png')
def serve_image(filename = "cal.png"):
    return send_from_directory('', filename)

if __name__ == '__main__':
    app.run()
