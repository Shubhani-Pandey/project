from django.shortcuts import render
from django.http import HttpResponse
import tensorflow as tf
from PIL import Image
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.http import HttpResponse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np
import cv2
import json
import base64
import pickle
from PIL import Image
from io import BytesIO

#bp_model = pickle.load(open("bp_linear_model.sav", 'rb'))


@csrf_exempt
def home(request):
    return HttpResponse("home working")
# Create your views here.

@csrf_exempt
def face_feature_extraction(request):
    received_json = json.loads(request.body)
    url = received_json['url']
    encoded_data = url.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades +'/haarcascade_frontalface_default.xml')
    eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades +'/haarcascade_eye.xml')
    #img = cv2.imread('lalala.PNG')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.2, 5)
    # When no faces detected, face_classifier returns and empty tuple
    if faces is ():
        print("No Face Found")
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_classifier.detectMultiScale(gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(255,255,0),2)
    #cv2.imshow(img) 
    encoded_string = base64.b64encode(img ).decode()
    #return HttpResponse(img, content_type="image/png")
    return HttpResponse(encoded_string)


@csrf_exempt
def bp_prediction(request):
    values=[]
    received_json = json.loads(request.body)
    age = received_json['user_age']
    sex_val = received_json['user_sex']
    weight = received_json['user_weight']
    height = received_json['user_height']
    smoking_val = received_json['user_smoking_or_not']
    cholesterol = received_json['cholesterol']

    if sex_val =='F' or 'f':
      sex = 1.0
    else:
      sex = 0.0

    if smoking_val == 'Y' or 'y':
      smoking = 1.0
    else:
      smoking = 0.0

    race = 1.0

    values.extend((age, sex, height,weight, race, smoking, cholesterol))
    data = [values]

    # res=bp_model.predict(data)
    res=0

    if res==1:
      response = 'high_bp'
    else:
      response = 'normal_bp'    

    return HttpResponse(response)


@csrf_exempt
def eye_disease_pred(request):
    received_json = json.loads(request.body)
    return HttpResponse('Glaucoma')
