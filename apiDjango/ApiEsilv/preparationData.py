#backend preparation
from django.test import TestCase
import pandas as pd
import zipfile
from datetime import datetime
import time
from scipy import stats
from sqlite3 import sq_access
from sqlite3 import sq_get

df = sq_access(init_ , self_)
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00498/incident_event_log.zip")

indexNames = df[ df['resolved_at'] == '?' ].index
df.drop(indexNames , inplace=True)
df['opened_at'] = pd.to_datetime(df["opened_at"])
df['resolved_at'] = pd.to_datetime(df["resolved_at"])
df['duration'] = df['resolved_at']-df['opened_at']
df['duration']= df['duration'].dt.total_seconds() / 86400
df['duration']= df['duration'].round(1)
df['duration'] = df['duration'].astype('float')

indexneg = df[ df['duration'] < 0 ].index
df.drop(indexneg , inplace=True)
indexState = df[ df['incident_state'] == '-100' ].index
df.drop(indexState , inplace=True)
df = df.drop(columns=['vendor', 'caused_by','rfc','sys_created_at','cmdb_ci','assigned_to'])
df = df.drop(columns=['opened_at', 'resolved_at','closed_at'])

def hr_func(ts):
    return ts.hour
	
df['sys_updated_at'] = pd.to_datetime(df["sys_updated_at"])
df['sys_updated_at'] = df['sys_updated_at'].apply(hr_func)

df['incident_state'] = df['incident_state'].astype('category')
df['incident_state'] = df['incident_state'].cat.codes
df['active'] = df['active'].astype('category')
df['active'] = df['active'].cat.codes
df['number'] = df['number'].astype('category')
df['number'] = df['number'].cat.codes
df['made_sla'] = df['made_sla'].astype('category')
df['made_sla'] = df['made_sla'].cat.codes
df['caller_id'] = df['caller_id'].astype('category')
df['caller_id'] = df['caller_id'].cat.codes
df['opened_by'] = df['opened_by'].astype('category')
df['opened_by'] = df['opened_by'].cat.codes
df['sys_created_by'] = df['sys_created_by'].astype('category')
df['sys_created_by'] = df['sys_created_by'].cat.codes
df['sys_updated_by'] = df['sys_updated_by'].astype('category')
df['sys_updated_by'] = df['sys_updated_by'].cat.codes
df['sys_updated_at'] = df['sys_updated_at'].astype('category')
df['sys_updated_at'] = df['sys_updated_at'].cat.codes
df['contact_type'] = df['contact_type'].astype('category')
df['contact_type'] = df['contact_type'].cat.codes
df['location'] = df['location'].astype('category')
df['location'] = df['location'].cat.codes
df['category'] = df['category'].astype('category')
df['category'] = df['category'].cat.codes
df['subcategory'] = df['subcategory'].astype('category')
df['subcategory'] = df['subcategory'].cat.codes
df['u_symptom'] = df['u_symptom'].astype('category')
df['u_symptom'] = df['u_symptom'].cat.codes
df['impact'] = df['impact'].astype('category')
df['impact'] = df['impact'].cat.codes
df['urgency'] = df['urgency'].astype('category')
df['urgency'] = df['urgency'].cat.codes
df['priority'] = df['priority'].astype('category')
df['priority'] = df['priority'].cat.codes
df['assignment_group'] = df['assignment_group'].astype('category')
df['assignment_group'] = df['assignment_group'].cat.codes
df['u_priority_confirmation'] = df['u_priority_confirmation'].astype('category')
df['u_priority_confirmation'] = df['u_priority_confirmation'].cat.codes
df['notify'] = df['notify'].astype('category')
df['notify'] = df['notify'].cat.codes
df['problem_id'] = df['problem_id'].astype('category')
df['problem_id'] = df['problem_id'].cat.codes
df['closed_code'] = df['closed_code'].astype('category')
df['closed_code'] = df['closed_code'].cat.codes
df['resolved_by'] = df['resolved_by'].astype('category')
df['resolved_by'] = df['resolved_by'].cat.codes
df['notify'] = df['notify'].astype('category')
df['notify'] = df['notify'].cat.codes

Q1 = df['duration'].quantile(0.25)
Q3 = df['duration'].quantile(0.75)
IQR = Q3 - Q1

filter = (df['duration'] >= Q1 - 1.5 * IQR) & (df['duration'] <= Q3 + 1.5 *IQR)
df = df.loc[filter]  

return sq_get(self_ , df)
