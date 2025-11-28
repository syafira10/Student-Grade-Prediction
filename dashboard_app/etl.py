import pandas as pd
from django.db import connection
from datetime import datetime, timedelta
import numpy as np

class StudentDataETL:
    """ETL Pipeline for Student Data Processing"""
    
    def __init__(self):
        self.raw_data = None
        self.processed_data = None
    
    def extract_student_data(self, student_name=None):
        """Extract data from PostgreSQL database"""
        query = """
        SELECT 
            s.stu_id,
            s.name,
            s.email,
            s.gender,
            s.dob,
            c.course_name,
            e.grade,
            ca.activity_name,
            at.type_name as activity_type,
            sal.activity_start,
            sal.activity_end,
            EXTRACT(EPOCH FROM (sal.activity_end - sal.activity_start))/3600 as hours_spent
        FROM student s
        LEFT JOIN enrollment e ON s.stu_id = e.stu_id
        LEFT JOIN course c ON e.course_id = c.course_id
        LEFT JOIN student_activity_log sal ON s.stu_id = sal.stu_id
        LEFT JOIN course_activity ca ON sal.activity_id = ca.activity_id
        LEFT JOIN activity_type at ON ca.type_id = at.type_id
        """
        
        if student_name:
            query += f" WHERE s.name = '{student_name}'"
            
        with connection.cursor() as cursor:
            cursor.execute(query)
            columns = [col[0] for col in cursor.description]
            data = cursor.fetchall()
            
        self.raw_data = pd.DataFrame(data, columns=columns)
        return self.raw_data
    
    def transform_data(self):
        """Transform and clean the extracted data"""
        if self.raw_data is None or self.raw_data.empty:
            return pd.DataFrame()
        
        df = self.raw_data.copy()
        
                
        # Aggregate student data
        student_data = []
        
        for student_id in df['stu_id'].unique():
            student_df = df[df['stu_id'] == student_id]
            
            # Basic student info
            student_info = student_df.iloc[0]
            
            # Calculate activity metrics
            total_activities = len(student_df.dropna(subset=['activity_name']))
            total_hours = float(student_df['hours_spent'].fillna(0).sum())  # Force float type
            
            # Calculate course grades and averages
            course_grades = student_df.dropna(subset=['grade'])
            avg_grade = course_grades['grade'].mean() if not course_grades.empty else 0
            
            # Get enrolled courses with grades
            courses = {}
            for _, row in course_grades.iterrows():
                if pd.notna(row['course_name']) and pd.notna(row['grade']):
                    courses[row['course_name']] = row['grade']
            
            student_data.append({
                'stu_id': student_id,
                'name': student_info['name'],
                'email': student_info['email'],
                'gender': student_info['gender'],
                'avg_past_grade': avg_grade,
                'total_activities': total_activities,
                'total_hours': total_hours,
                'courses': courses
            })
        
        self.processed_data = pd.DataFrame(student_data)
        return self.processed_data
    
    def load_student_profile(self, student_name):
        """Complete ETL process for a specific student"""
        self.extract_student_data(student_name)
        transformed_data = self.transform_data()
        
        if transformed_data.empty:
            return None
            
        return transformed_data.iloc[0].to_dict()
