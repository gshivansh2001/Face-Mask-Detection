from flask import Flask, render_template, Response
import cv2
from tensorflow.keras.models import load_model
import datetime
from keras.preprocessing import image
app=Flask(__name__)
camera = cv2.VideoCapture(0)
model = load_model('Mask_Detection.h5')

def gen_frames():  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        
            faces=detector.detectMultiScale(frame,1.1,7)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
             #Draw the rectangle around each face
            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                cv2.imwrite('temp.jpg',face_img)
                test_image = cv2.imread('temp.jpg')
                test_image = cv2.resize(test_image,(160,160))
                test_image = test_image.reshape(1,160,160,3)
                confidence = model.predict(test_image)
                if confidence>=0.5:
                    label = "Masked"+ ' Probability={}'.format(confidence)
                    #cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
                    #cv2.putText(img,'NO MASK',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 2)
                    #Drawing a GREEN coloured rectangle around the face
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), thickness=2)
                else:
                    label = "Not Masked"+ ' Probability={}'.format(confidence)
                    #Putting Not Masked text in the images
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 2)
                    #Drawing a RED coloured rectangle around the face
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), thickness=2)
                    #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
                    #cv2.putText(img,'MASK',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
                datet=str(datetime.datetime.now())
                cv2.putText(frame,datet,(400,450),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
                ### cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                ### roi_gray = gray[y:y+h, x:x+w]
                ### roi_color = frame[y:y+h, x:x+w]

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    app.run(debug=True)