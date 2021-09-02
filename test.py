from base64 import b64encode
import requests
import os



# fetches url for vf-commonit work-item api
def fetch_url(organization='vf-commonit', project='WebOperations', work_type='task', api_version='6.0', user='', auth=''):
    if user and auth:
        url = 'https://' + user + ':' + auth +'@dev.azure.com/' + organization + '/' + project + '/_apis/wit/workitems/$' + work_type + '?api-version=' + api_version
    else:
        url = 'https://dev.azure.com/' + organization + '/' + project + '/_apis/wit/workitems/$' + work_type + '?api-version=' + api_version
    return url

#user, auth, pat = read_credentials('py-ticket/credentials.txt')
user = 'bek64'
pat = 'eiury23tvjipqwdua74bvuvlmbnpycxpngji4ycfe3okppnanygqV'

user_64 = b64encode(user.encode('utf-8')).decode('utf-8') # For Azure Devops authentication
pat_64 = b64encode(pat.encode('utf-8')).decode('utf-8')

url = fetch_url()
data = [
    {
        "op": "add",
        "path": "/fields/System.Title",
        "value": "Test task"
    }
]

url = 'https://azure.microsoft.com'
proxies = {'https': ''}
r = requests.get(url, proxies=proxies)

print(r.status_code)
print(pat_64)
#print(r.content)
print(url)