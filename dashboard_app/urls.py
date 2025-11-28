# from django.urls import path
# from . import views

# # urlpatterns = [
# #     path('', views.predict_view, name='predict_view'),
    
# # ]

# urlpatterns = [
#     path('', views.dashboard, name='dashboard'),
#     # path('nauval_case', views.predict_view, name='predict_view'),
#     path('nauval_case/', views.predict_view, name='nauval_case'),
#     # path('api/students/', views.get_students, name='get_students'),
#     # path('api/predict/', views.predict_grade, name='predict_grade'),
# ]

from django.urls import path
from . import views

from django.conf import settings
from django.conf.urls.static import static



urlpatterns = [
    # URL untuk menampilkan halaman home
    path('', views.dashboard, name='dashboard'),
    # path('prediction/', views.prediction, name='prediction'),

    #nauval_url
    path('prediction/nauval_case/', views.predict_view, name='predict_view'),
    path('prediction/nauval-case/api/', views.predict_view, name='nauval_case'),

    #habibi_url
    path('prediction/habibi_case/', views.student_prediction_view, name='prediction_habibi'),
    path('prediction/habibi-case/api/', views.predict_student, name='habibi_case'),

    #kalvin_url
    path('prediction/kalvin_case/', views.predict_activity_duration, name='prediction_kalvin'),

    #syafira_url
    path('prediction/syafira_case/', views.grade_predictor, name='grade_predictor'),
    path('api/students/', views.get_student_data, name='get_students'),
    path('api/predict/', views.predict_grade, name='predict_grade'),
]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)