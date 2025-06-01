
import cv2 
from flask import Flask, render_template, request, Response
import keras.utils as image 
import numpy as np
from keras.models import model_from_json, Sequential  
import keras.utils as image  

#load model  
with open("models/fer_model.json", "r") as json_file:
    loaded_model_json = json_file.read()

model = model_from_json(loaded_model_json, custom_objects={'Sequential': Sequential})  # Now this will work
  
#load weights  
model.load_weights('models/fer_model.h5')  

  
face_haar_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml') 
app = Flask(__name__)



cap = cv2.VideoCapture(0)

def video_stream():
# loop runs if capturing has been initialized.
	while True:
		ret,test_img=cap.read()# captures frame and returns boolean value and captured image  
		if not ret:  
			continue  
		gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)  
	
		faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)  
	
	
		for (x,y,w,h) in faces_detected:  
			cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)  
			roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image  
			roi_gray=cv2.resize(roi_gray,(48,48))  
			img_pixels = image.img_to_array(roi_gray)  
			img_pixels = np.expand_dims(img_pixels, axis = 0)  
			img_pixels /= 255  
	
			predictions = model.predict(img_pixels)  
			print(predictions)
			#find max indexed array  
			max_index = np.argmax(predictions[0])  
	
			emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')  
			predicted_emotion = emotions[max_index]  
	
			cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)   
				
		ret, buffer = cv2.imencode('.jpg', test_img)
		test_img = buffer.tobytes()
		yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + test_img + b'\r\n')

 
@app.route("/")
@app.route("/first")
def first():
	return render_template('first.html')
      
@app.route("/livevideo")
def livevideo():
	return render_template('livevideo.html')
@app.route("/index", methods=['GET', 'POST'])
def index():
	return render_template("index.html")
@ app.route('/video-feed', methods=['GET'])
def video_feed():
    return Response(video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/image-prediction", methods=['GET', 'POST'])
def image_prediction():
	try:
		file = request.files.get('image')
		img =  cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		face = face_haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
		#Draw the rectangle around each face
		for (x, y, w, h) in face:
			#cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
			roi_gray=gray[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image  
			roi_gray=cv2.resize(roi_gray,(48,48))  
			img_pixels = image.img_to_array(roi_gray)  
			img_pixels = np.expand_dims(img_pixels, axis = 0)  
			img_pixels /= 255  
  
			predictions = model.predict(img_pixels)  
			print(predictions)
			#find max indexed array  
			max_index = np.argmax(predictions[0])  
	
			emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')  
			predicted_emotion = emotions[max_index]
			
		return render_template('prediction.html', prediction=predicted_emotion)
	except:
		pass
		
@app.route('/requests',methods=['POST'])
def video_operation():
    global cap
    if  request.form.get('stop') == 'Stop':
            cap.release()
            return render_template('livevideo.html')
            #cv2.destroyAllWindows()
            
    else:
        cap = cv2.VideoCapture(0)
        return render_template('livevideo.html')
    


	
if __name__ =='__main__':
	app.run(debug = True)


	

	


