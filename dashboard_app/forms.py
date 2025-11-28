from django import forms

#nauval_forms
class PredictForm(forms.Form):
    nama = forms.CharField(label='Nama', max_length=100) 
    gender = forms.ChoiceField(
        choices=[('0', 'Male'), ('1', 'Female')],
        label='Gender'
    )
    age = forms.IntegerField(label='Age')
    activity_type = forms.ChoiceField(
        choices=[
            ('0', 'Forum'),
            ('1', 'Group Assignment'),
            ('2', 'Individual Assignment'),
            ('3', 'Quiz')
        ],
        label='Activity Type'
    )
    total_duration_minutes = forms.FloatField(label='Total Duration (Minutes)')

#habibi_forms
class StudentPredictionForm(forms.Form):
    age = forms.IntegerField(label='Age', min_value=15, max_value=30)
    avg_prior_grade = forms.FloatField(label='Average Prior Grade', min_value=0, max_value=100)
    target_course_id = forms.IntegerField(label='Target Course ID', min_value=1, max_value=5)

#kalvin_forms
class PredictionForm(forms.Form):
    activity_type_id = forms.IntegerField(label='Activity Type ID')
    course_id = forms.IntegerField(label='Course ID')
    stu_id = forms.IntegerField(label='Student ID')

#syafira_forms
class GradePredictionForm(forms.Form):
    gender = forms.ChoiceField(
        label="Gender",
        choices=[('1', 'Male'), ('0', 'Female')],
        widget=forms.RadioSelect
    )
    course_id = forms.ChoiceField(
        label="Course",
        choices=[('1', 'Course 1'), ('2', 'Course 2'), ('3', 'Course 3'), ('4', 'Course 4'), ('5', 'Course 5')]
    )
    activity_count = forms.IntegerField(label="Number of Activities Participated", min_value=0)
    total_activity_time = forms.FloatField(label="Total Activity Time (seconds)", min_value=0)
    avg_past_grade = forms.FloatField(label="Average Past Grade", min_value=0, max_value=100)
