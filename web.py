from flask import Flask, request, jsonify
from PIL import Image
import requests
from io import BytesIO
from minio import Minio
from minio.error import S3Error
import logging

from unet import Unet

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

minio_client = Minio(
    endpoint="localhost:9000",
    access_key="admin",
    secret_key="admin123",
    secure=False
)
bucket_name = "root"


def upload_to_minio(file_data, file_name):
    try:
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)
        file_data.seek(0, 2)
        file_length = file_data.tell()
        file_data.seek(0)
        minio_client.put_object(bucket_name, file_name, file_data, length=file_length, content_type="image/jpeg")
        return f"http://localhost:9000/{bucket_name}/{file_name}"
    except S3Error as exc:
        app.logger.error("Error uploading to MinIO: %s", exc)
        return None


@app.route('/predict', methods=['POST'])
def predict():
    app.logger.info("Processing new prediction request")
    file_url = request.form['fileUrl']
    img_type = request.form['type']

    if img_type == "1":
        model_type = "cell"
    elif img_type == "2":
        model_type = "root"
    else:
        return jsonify({"error": "Invalid type provided"}), 400

    app.logger.info("Downloading image from URL")
    try:
        response = requests.get(file_url)
        image = Image.open(BytesIO(response.content))
    except Exception as e:
        app.logger.error("Error downloading or opening image: %s", e)
        return jsonify({"error": "Error processing image"}), 500

    app.logger.info("Predicting image using Unet model")
    try:
        unet = Unet(model_type=model_type)
        predicted_image = unet.detect_image(image)
    except Exception as e:
        app.logger.error("Error during image prediction: %s", e)
        return jsonify({"error": "Error during prediction"}), 500

    app.logger.info("Uploading predicted image to MinIO")
    img_byte_arr = BytesIO()
    predicted_image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    predicted_file_name = f"predicted/{model_type}_{file_url.split('/')[-1]}"
    predicted_url = upload_to_minio(BytesIO(img_byte_arr), predicted_file_name)

    if predicted_url:
        app.logger.info("Prediction and upload successful")
        return jsonify({"predictedUrl": predicted_url})
    else:
        app.logger.error("Failed to upload predicted image")
        return jsonify({"error": "Failed to upload predicted image"}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
