#access url
from App.views import Train, Predict
from django.conf.urls import url

app_name = 'App'


urlpatterns = [
    url('.\modelsML', Predict.as_view(), name="lineapi()", id=1),
    url('.\modelsML', Predict.as_view(), name="cvapi()",id=2),
    url('.\modelsML', Predict.as_view(), name="dtapi()",id=3),
    url('.\modelsML', Predict.as_view(), name="mlpapi",id=4),
	url('.\modelsML', Predict.as_view(), name="rfapi",id=5),
	url('.\modelsML', Predict.as_view(), name="elastapi()",id=6)
	]
