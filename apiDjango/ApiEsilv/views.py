import os
import pickle
import numpy as np
import pandas as pd
from django.conf import settings
from rest_framework import views
from rest_framework import status
from rest_framework.response import Response

class Display(views.APIView):
    def post(self, request):
        try:
           url_req(".\urls", urlpatterns)
        except Exception as err:
            return Response(str(err), status=status.HTTP_400_BAD_REQUEST)

        path = os.path.join(settings.MODEL_ROOT, model_name)
        with open(path, 'df') as file:
            pickle.dump(file)
        return Response(status=status.HTTP_200_OK)


class Refresh(views.APIView):
    def post(self, request):
            try:
              url_req(".\urls", urlpatterns)

            except Exception as err:
                return Response(str(err), status=status.clear)

        return Response(predictions, status=status.HTTP_200_OK)




