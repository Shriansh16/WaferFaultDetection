from flask import Flask,request,render_template,jsonify,send_file
import os
import sys
from pathlib import Path 
sys.path.insert(0, 'D:\WaferFaultDetection\src\pipelines')
from prediction_pipeline import PredictionPipeline


application=Flask(__name__)
app=application


@app.route('/')
def home():
    return "Welcome to my application"
@app.route('/predict',methods=['GET','POST'])
def upload():
    try:
        if request.method=='POST':
            prediction_pipeline=PredictionPipeline(request)
            prediction_file_detail = prediction_pipeline.run_pipeline()

            #logging.info("prediction completed. Downloading prediction file.")
            return send_file(prediction_file_detail.prediction_file_path,
                            download_name= prediction_file_detail.prediction_file_name,
                            as_attachment= True)
        else:
            return render_template('upload_file.html')
    except Exception as e:
        raise CustomException(e,sys)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug= True)