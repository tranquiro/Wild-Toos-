from django.urls import path
from .views import PredView

urlpatterns = [
    path('', PredView.as_view(), name='index'),
]