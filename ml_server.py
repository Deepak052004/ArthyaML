import os
import urllib.request
from flask import Flask, request, jsonify
import fasttext

model_path = "lid.176.bin"
model_url = "https://drive.google.com/uc?export=download&id=1m4TQ6Op-p5CQRLiPRwStJaPYRl5fm_aq"

if not os.path.exists(model_path):
    print("Downloading FastText model...")
    urllib.request.urlretrieve(model_url, model_path)
    print("Download complete.")

model = fasttext.load_model(model_path)

app = Flask(__name__)

@app.route('/parse', methods=['POST'])
def parse():
    data = request.get_json()
    text = data.get("text", "")
    prediction = model.predict(text)
    return jsonify({"prediction": prediction[0][0]})

if __name__ == "__main__":
    app.run(debug=True)
