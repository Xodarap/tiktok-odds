from imgurpython import ImgurClient
from dotenv import load_dotenv
load_dotenv()
import os
import base64
import requests
import ntpath

def upload_image(album_hash, path):
    fd = open(path, 'rb')
    contents = fd.read()
    fd.close()
    b64 = base64.b64encode(contents)
    url = "https://api.imgur.com/3/image"
    
    payload = {'image': b64,
               'album': album_hash,
               'name': ntpath.basename(path)}
    files = []
    headers = {
      'Authorization': f'Client-ID {os.getenv("IMGUR_ID")}'
    }
    
    response = requests.request("POST", url, headers=headers, data = payload, files = files)
    return response.json()

def create_album(title):
    url = "https://api.imgur.com/3/album"

    payload = {'title': title }
    files = []
    headers = {
      'Authorization': f'Bearer {os.getenv("IMGUR_ACCESS")}'
    }
    
    response = requests.request("POST", url, headers=headers, data = payload, files = files)
    return response.json()


# print(upload_image('1t9Mpfemtvldk1a', "D:\\Documents\\tiktok-live-graphs\\makeup-overtime\\Maybelline control.jpg"))