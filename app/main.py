from flask import Flask, render_template, Response
from scipy.spatial import distance
import numpy as np
import cv2
from model.model import predict_pipeline

app = Flask(__name__)


font = cv2.FONT_HERSHEY_DUPLEX
color_identify = (255, 255, 255)
color_unknown = (0, 0, 255)

def gen_frame():
    camera = cv2.VideoCapture(0)
    print(f'Camera runing status: {camera.isOpened()}')
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            predict_results = predict_pipeline(frame)
            for idx, (predicted_name, confidence) in enumerate(predict_results):
                if predicted_name == "Waiting":
                    if idx == 0:
                        cv2.putText(frame, f'{predicted_name} - {np.round(confidence, 2)}', (6, 40), font, 1.0, color_unknown, 1)
                    else:
                        cv2.putText(frame, f'{predicted_name} - {np.round(confidence, 2)}', (6, 40+(40*idx)), font, 1.0, color_unknown, 1)
                else:
                    # img[y:h, x:w]
                    if idx == 0:
                        cv2.putText(frame, f'{predicted_name} - {np.round(confidence, 2)}', (6, 40), font, 1.0, color_identify, 1)
                    else:
                        cv2.putText(frame, f'{predicted_name} - {np.round(confidence, 2)}', (6, 40+(40*idx)), font, 1.0, color_identify, 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == "__main__":
    app.run(debug=True)
