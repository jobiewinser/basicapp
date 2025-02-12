import requests
from django.views.generic import TemplateView, View
from django.utils.decorators import method_decorator
from django.http import JsonResponse
from django.views.decorators.csrf import ensure_csrf_cookie  # Import csrf_exempt for bypassing CSRF for simplicity

@ensure_csrf_cookie
def get_csrf_token(request):
    # This view does not need to do anything except set the CSRF cookie,
    # the @ensure_csrf_cookie decorator handles that.
    return JsonResponse({'message': 'CSRF cookie set'})

class CalculateConfidenceView(View):
    def post(self, request, *args, **kwargs):
        uploaded_statement = request.POST.get("uploaded_statement")
        uploaded_statement = """With chronic illness at its highest ever level, many people are turning to
            treatments and activities such as reiki, healing touch, yoga and massage in
            search of answers to long term medical issues and pain."""
        if uploaded_statement:
            response = requests.post("http://ai-model:8000/predict_confidence", json={"text": uploaded_statement})
            confidence = response.json().get("confidence", "An error occurred")
            return JsonResponse({"confidence": confidence})
        return JsonResponse({"error": "No statement uploaded"}, status=400)