from django.urls import path, include

urlpatterns = [
    path('', include('search_api.urls')),
]
