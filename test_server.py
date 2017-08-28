import requests
import base64
import os
import simplejson.scanner

URL = 'http://dev.dummy.ai/model/jimfan/colorization/0.0.1'
# URL = 'http://localhost:5900'


with open('demo/imgs/ansel_adams3.jpg', 'rb') as f:
    result = requests.post(URL, json={
        'img_in': ease64.b64encode(f.read()).decode('utf-8')
    })

    try:
        result = result.json()
    except simplejson.scanner.JSONDecodeError:
        print('Cannot decode JSON: ')
        print(result.text)
        exit(1)

    image_binary = base64.b64decode(result['img_out'])
    with open('output.png', 'wb') as f:
        f.write(image_binary)
    os.system('open output.png')

