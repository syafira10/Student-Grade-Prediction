from django.db import models
from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator

class TeamMember(models.Model):
    name = models.CharField(max_length=100)
    role = models.CharField(max_length=100)
    description = models.TextField()
    prediction_type = models.CharField(max_length=200)
    status_choices = [
        ('active', 'Active'),
        ('development', 'In Development'),
        ('completed', 'Completed'),
    ]
    status = models.CharField(max_length=20, choices=status_choices, default='active')
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.name} - {self.prediction_type}"

class PredictionFeature(models.Model):
    title = models.CharField(max_length=200)
    description = models.TextField()
    member = models.ForeignKey(TeamMember, on_delete=models.CASCADE)
    icon_class = models.CharField(max_length=50, default='🎯')
    button_color = models.CharField(max_length=20, default='blue')
    is_active = models.BooleanField(default=True)
    
    def __str__(self):
        return self.title

# Model untuk tabel ⁠ student ⁠
class Student(models.Model):
    stu_id = models.IntegerField(primary_key=True)  # ID mahasiswa
    name = models.CharField(max_length=100)         # Nama mahasiswa
    email = models.EmailField(max_length=100)       # Email mahasiswa
    gender = models.CharField(max_length=10)        # Gender mahasiswa
    dob = models.DateField()                        # Tanggal lahir mahasiswa

    class Meta:
        db_table = 'student'  # Nama tabel di database
        managed = False  # Django tidak akan mengelola tabel ini

    def __str__(self):
        return self.name

# Model untuk tabel ⁠ course ⁠
class Course(models.Model):
    course_id = models.IntegerField(primary_key=True)  # ID kursus
    course_name = models.CharField(max_length=100)     # Nama kursus

    class Meta:
        db_table = 'course'  # Nama tabel di database
        managed = False  # Django tidak akan mengelola tabel ini

    def __str__(self):
        return self.course_name

# Model untuk tabel ⁠ activity_type ⁠
class ActivityType(models.Model):
    type_id = models.IntegerField(primary_key=True)  # ID jenis aktivitas
    type_name = models.CharField(max_length=50)      # Nama jenis aktivitas (misalnya Quiz, Tugas, Forum)

    class Meta:
        db_table = 'activity_type'  # Nama tabel di database
        managed = False  # Django tidak akan mengelola tabel ini

    def __str__(self):
        return self.type_name

# Model untuk tabel ⁠ course_activity ⁠
class CourseActivity(models.Model):
    activity_id = models.IntegerField(primary_key=True)  # ID aktivitas
    type_id = models.ForeignKey(ActivityType, on_delete=models.CASCADE, db_column='type_id')  # tambahkan db_column
    course_id = models.ForeignKey(Course, on_delete=models.CASCADE, db_column='course_id')
    activity_name = models.CharField(max_length=100)     # Nama aktivitas
    activity_start_date = models.DateTimeField()          # Tanggal mulai aktivitas
    activity_end_date = models.DateTimeField()            # Tanggal selesai aktivitas

    class Meta:
        db_table = 'course_activity'  # Nama tabel di database
        managed = False  # Django tidak akan mengelola tabel ini

    def __str__(self):
        return self.activity_name

# Model untuk tabel ⁠ enrollment ⁠
class Enrollment(models.Model):
    enroll_id = models.IntegerField(primary_key=True)
    stu_id = models.ForeignKey(Student, on_delete=models.CASCADE, db_column='stu_id')
    course_id = models.ForeignKey(Course, on_delete=models.CASCADE, db_column='course_id')
    grade = models.IntegerField()

    class Meta:
        db_table = 'enrollment'
        managed = False

    def __str__(self):
        return f"{self.stu_id.name} - {self.course_id.course_name}"

# Model untuk tabel ⁠ student_activity_log ⁠
class StudentActivityLog(models.Model):
    stu_id = models.ForeignKey(Student, on_delete=models.CASCADE, db_column='stu_id')
    activity_id = models.ForeignKey(CourseActivity, on_delete=models.CASCADE, db_column='activity_id')  # <--- PENTING!
    activity_start = models.DateTimeField()
    activity_end = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = 'student_activity_log'
        managed = False

class ActivityPrediction(models.Model):
    GENDER_CHOICES = [
        (0, 'Female'),
        (1, 'Male'),
    ]
    
    ACTIVITY_TYPE_CHOICES = [
        (0, 'Activity Type A'),
        (1, 'Activity Type B'),
        (2, 'Activity Type C'),
        (3, 'Activity Type D'),
    ]
    
    gender = models.IntegerField(choices=GENDER_CHOICES)
    age = models.IntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(100)]
    )
    activity_type = models.IntegerField(choices=ACTIVITY_TYPE_CHOICES)
    duration_minutes = models.IntegerField(
        validators=[MinValueValidator(1)]
    )
    prediction_result = models.BooleanField()
    confidence_score = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Prediction {self.id} - {'Success' if self.prediction_result else 'Failure'}"
    
    class Meta:
        ordering = ['-created_at']




    