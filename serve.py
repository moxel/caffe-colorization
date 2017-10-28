from flask import Flask, request, jsonify
from model import Model
import numpy as np
from PIL import Image
import scipy.misc
import base64
from io import StringIO, BytesIO

import moxel

m = Model()

def predict(img_in):
    img_out = m.predict(img_in.to_numpy())
    return {
        'img_out': moxel.space.Image(img_out)
    }
