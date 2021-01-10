from django.urls import path

from .views import home
from .views import face_feature_extraction
from .views import bp_prediction
from .views import eye_disease_pred
urlpatterns = [
    path('home/', home),
    path('face_feature_extraction/',face_feature_extraction),
    path('bp_prediction/',bp_prediction),
    path('eye_disease_prediction/',eye_disease_pred),
]
