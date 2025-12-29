from flask import Flask, render_template
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_resnet50', methods=['POST'])
def detect_resnet50():
    result = subprocess.run(["python", "predict_vitiligo50.py"], capture_output=True, text=True)
    prediction = result.stdout.strip()
    return render_template('result.html', prediction=prediction, model="ResNet50")

@app.route('/detect_resnetrs50', methods=['POST'])
def detect_resnetrs50():
    result = subprocess.run(["python", "predict_vitiligoRS.py"], capture_output=True, text=True)
    prediction = result.stdout.strip()
    return render_template('result.html', prediction=prediction, model="ResNet-RS-50")


@app.route('/detect_efficientnetb6', methods=['POST'])
def detect_efficientnetb6():
    result = subprocess.run(["python", "predict_vitiligoRS.py"], capture_output=True, text=True)
    prediction = result.stdout.strip()
    return render_template('result.html', prediction=prediction, model="efficientnetb6")


@app.route('/detect_efficientnetv2', methods=['POST'])
def detect_efficientnetv2():
    result = subprocess.run(["python", "predict_vitiligoefv2.py"], capture_output=True, text=True)
    prediction = result.stdout.strip()
    return render_template('result.html', prediction=prediction, model="efficientnetv2")



@app.route('/detect_efficientnetb0', methods=['POST'])
def detect_efficientnetb0():
    result = subprocess.run(["python", "predict_vitiligoefb0.py"], capture_output=True, text=True)
    prediction = result.stdout.strip()
    return render_template('result.html', prediction=prediction, model="efficientnetb0")




@app.route('/detect_resnet101', methods=['POST'])
def detect_resnet101():
    result = subprocess.run(["python", "predict_vitiligoresnet101.py"], capture_output=True, text=True)
    prediction = result.stdout.strip()
    return render_template('result.html', prediction=prediction, model="resnet101")

if __name__ == '__main__':
    app.run(debug=True)
