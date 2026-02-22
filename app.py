from flask import Flask, render_template, Response
import main as gesture_main
import cv2
import threading

app = Flask(__name__)

def gen_frames():
    while True:
        if gesture_main.latest_frame is not None:
            ret, buffer = cv2.imencode('.jpg', gesture_main.latest_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Start the gesture logic in a separate thread
    threading.Thread(target=gesture_main.main, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)