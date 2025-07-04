import os
import fasttext
from flask import Flask, request, jsonify

model_path = "lid.176.bin"
gdrive_file_id = "1m4TQ6Op-p5CQRLiPRwStJaPYRl5fm_aq"

# Download the model from Google Drive using gdown if not already present
if not os.path.exists(model_path):
    print("Downloading model via gdown...")
    os.system(f"pip install gdown && gdown --id {gdrive_file_id} -O {model_path}")
    print("Download complete.")

# Load the FastText model
model = fasttext.load_model(model_path)

# Initialize Flask app
app = Flask(__name__)

@app.route('/parse', methods=['POST'])
def parse():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    prediction = model.predict(text)
    language = prediction[0][0].replace("__label__", "")
    confidence = prediction[1][0]
    
    return jsonify({
        "language": language,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
