from django.shortcuts import render
from .forms import PredictForm, StudentPredictionForm, PredictionForm
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import json
import joblib
import pickle
import os
import numpy as np
import pandas as pd
from .etl import StudentDataETL
from .ml_model import GradePredictionModel
from django.db import connection
from .models import TeamMember, PredictionFeature


# Create your views here.
def dashboard(request):
    return render(request, 'dashboard_app/dashboard.html')

#nauval_views
MODEL_PATH = 'rf_activity_success_model.pkl'

# Load model hanya jika file ada
model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

def predict_view(request):
    prediction = None
    probability = None

    if request.method == 'POST':
        form = PredictForm(request.POST)
        if form.is_valid() and model:
            gender = int(form.cleaned_data['gender'])
            age = form.cleaned_data['age']
            activity_type = int(form.cleaned_data['activity_type'])
            duration = form.cleaned_data['total_duration_minutes']

            # Siapkan input untuk prediksi
            X_input = np.array([[gender, age, activity_type, duration]])
            prediction = model.predict(X_input)[0]
            probability = model.predict_proba(X_input)[0][prediction]

    else:
        form = PredictForm()

    return render(request, 'dashboard_app/prediction/nauval_case.html', {
        'form': form,
        'prediction': prediction,
        'probability': probability,
    })

#habibi_views

def get_model_metrics():
    """
    Returns hardcoded model metrics for demonstration.
    In a real application, these would come from model evaluation.
    """
    return {
        "accuracy": 0.41,
        "classification_report": {
            "1": {"precision": 1.00, "recall": 1.00, "f1": 1.00, "support": 22},
            "2": {"precision": 0.38, "recall": 0.29, "f1": 0.32, "support": 21},
            "3": {"precision": 0.16, "recall": 0.22, "f1": 0.19, "support": 18},
            "4": {"precision": 0.24, "recall": 0.21, "f1": 0.22, "support": 19},
            "5": {"precision": 0.25, "recall": 0.25, "f1": 0.25, "support": 20},
        },
        "macro_avg": {"precision": 0.41, "recall": 0.39, "f1": 0.40},
        "weighted_avg": {"precision": 0.42, "recall": 0.41, "f1": 0.41}
    }

@csrf_exempt
def predict_student(request):
    if request.method == "POST":
        try:
            # Parse incoming JSON data
            data = json.loads(request.body)
            print(f"Data received: {data}")

            # Validate input data types if necessary (though the form uses number inputs)
            age = data.get("age")
            avg_prior_grade = data.get("avg_prior_grade")
            # target_course_id = data.get("target_course_id") # <-- REMOVE THIS LINE IF IT WAS HERE
            # The JS will stop sending it, so `data.get("target_course_id")` would be None anyway.

            # Ensure 'age' and 'avg_prior_grade' are present
            if not all([age, avg_prior_grade]):
                raise ValueError("Missing one or more required input fields: age, avg_prior_grade")

            # Prepare feature array - NOW ONLY WITH 2 FEATURES
            features = np.array([
                age,
                avg_prior_grade,
                # Remove target_course_id from here
            ]).reshape(1, -1)
            print(f"Features prepared: {features}")

            # Load model
            model_path = os.path.join(settings.BASE_DIR, 'final_student_model.pkl') # Corrected model filename
            print(f"Looking for model at: {model_path}")

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")

            model = joblib.load(model_path)
            print("Model loaded successfully")

            # Make prediction
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0].tolist()
            print(f"Prediction: {prediction}, Probability: {probability}")

            # Prepare response with prediction and metrics
            result = {
                "prediction": float(prediction),
                "probability": probability,
                "metrics": get_model_metrics()
            }
            return JsonResponse(result)

        except json.JSONDecodeError:
            print("Error: Invalid JSON received")
            return JsonResponse({"error": "Invalid JSON payload."}, status=400)
        except ValueError as ve:
            print(f"Prediction input error: {str(ve)}")
            return JsonResponse({"error": str(ve)}, status=400)
        except FileNotFoundError as fnfe:
            print(f"Model file error: {str(fnfe)}")
            return JsonResponse({"error": "Model file not found. Please check server configuration."}, status=500)
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return JsonResponse({"error": f"An unexpected error occurred: {str(e)}"}, status=500)
    else:
        return JsonResponse({"error": "Only POST requests are allowed."}, status=405)

def student_prediction_view(request):
    form = StudentPredictionForm()
    return render(request, 'dashboard_app/prediction/habibi_case.html', {'form': form})

#kalvin_views
def load_prediction_model():
    model_path = os.path.join(settings.BASE_DIR, 'activity_duration_predictor.pkl')

    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Model file not found. Please train the model first by running: "
            "'python manage.py student_activity_duration'"
        )
    
    with open(model_path, 'rb') as f:
        return pickle.load(f)

try:
    model_bundle = load_prediction_model()
    MODEL = model_bundle['model']
    PREPROCESSOR = model_bundle['preprocessor']
except Exception as e:
    MODEL = None
    PREPROCESSOR = None
    print(f"Model loading warning: {str(e)}")

def predict_activity_duration(request):
    form = PredictionForm()
    prediction = None
    error = None
    input_data = None

    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            try:
                if MODEL is None or PREPROCESSOR is None:
                    raise ValueError("Prediction model is not available.")
                
                input_data = {
                    'activity_type_id': form.cleaned_data['activity_type_id'],
                    'course_id': form.cleaned_data['course_id'],
                    'stu_id': form.cleaned_data['stu_id'],
                }

                df_input = pd.DataFrame([input_data])
                processed_data = PREPROCESSOR.transform(df_input)
                predicted_minutes = MODEL.predict(processed_data)[0]

                prediction = round(predicted_minutes, 2)

            except Exception as e:
                error = f"Prediction failed: {str(e)}"
        else:
            error = "Invalid form data."

    context = {
        'form': form,
        'prediction': prediction,
        'error': error,
        'input_data': input_data
    }

    return render(request, 'dashboard_app/prediction/kalvin_case.html', context)

#syafira_views

def grade_predictor(request):
    """Main page view"""
    return render(request, 'dashboard_app/prediction/syafira_case.html')


def get_student_data(request):
    """Get list of all students"""
    with connection.cursor() as cursor:
        cursor.execute("SELECT DISTINCT name FROM student ORDER BY name")
        students = [row[0] for row in cursor.fetchall()]
    
    return JsonResponse({'students': students})

@csrf_exempt
def predict_grade(request):
    """Predict grade for selected student"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            student_name = data.get('student_name')
            
            if not student_name:
                return JsonResponse({'error': 'Student name is required'}, status=400)
            
            # ETL Process
            etl = StudentDataETL()
            student_profile = etl.load_student_profile(student_name)
            
            if not student_profile:
                return JsonResponse({'error': 'Student not found'}, status=404)
            
            # ML Prediction
            ml_model = GradePredictionModel()
            prediction_result = ml_model.predict_grade(student_profile)
            
            # Combine student profile with prediction
            response_data = {
                'student_info': student_profile,
                'prediction': prediction_result
            }
            
            return JsonResponse(response_data)
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)