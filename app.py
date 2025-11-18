# from flask import Flask, request, render_template
# import os
# from predict import predict_image

# app = Flask(__name__)
# UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# #app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# @app.route("/", methods=["GET", "POST"])
# def index():
#     prediction = ""
#     image_path = ""
#     if request.method == "POST":
#         file = request.files["image"]
#         if file:
#             image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
#             file.save(image_path)
#             prediction = predict_image(image_path)
#     return render_template("index.html", prediction=prediction, image_path=image_path)

# if __name__ == "__main__":
#     app.run(debug=True)
from flask import Flask, request, render_template, send_from_directory, url_for
import os
from werkzeug.utils import secure_filename
from predict import predict_image

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Route to serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    filename = ""
    if request.method == "POST":
        file = request.files["image"]
        if file:
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(image_path)
            prediction = predict_image(image_path)
    return render_template("index.html", prediction=prediction, filename=filename)

if __name__ == "__main__":
    app.run(debug=True)
