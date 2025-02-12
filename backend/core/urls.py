from django.urls import path
from .views import CalculateConfidenceView, get_csrf_token

urlpatterns = [
    path('calculate-confidence/', CalculateConfidenceView.as_view(), name='calculate-confidence'),
    path('get-csrf-token/', get_csrf_token, name='get_csrf_token'),
]