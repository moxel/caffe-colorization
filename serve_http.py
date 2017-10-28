from flask import Flask, request, jsonify
from model import Model
import numpy as np
from PIL import Image
import scipy.misc
import base64
from io import StringIO, BytesIO

m = Model()

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)) / 255.


app = Flask(__name__)

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'OK'
    })


@app.route('/', methods=['POST'])
def detect():
    data = request.json

    image_binary = base64.b64decode(data['img_in'])

    image_f = BytesIO()
    image_f.write(image_binary)
    image_f.seek(0)

    image = Image.open(image_f)
    image_np = load_image_into_numpy_array(image)

    img_out = m.predict(image_np)['img_out']
    vis_file = BytesIO()
    scipy.misc.imsave(vis_file, img_out, format='png')
    vis_file.seek(0)
    vis_binary = vis_file.read()

    return jsonify({
        'img_out': base64.b64encode(vis_binary).decode('utf-8'),
    })

if __name__ == '__main__':
    app.run(debug=False, port=5900, host='0.0.0.0')
