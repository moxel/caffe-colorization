import argparse
import caffe
from model import Model

parser = argparse.ArgumentParser(description='iColor: deep interactive colorization')
parser.add_argument('-img_in',dest='img_in',help='grayscale image to read in', type=str)
parser.add_argument('-img_out',dest='img_out',help='colorized image to save off', type=str)
args = parser.parse_args()

img_rgb = caffe.io.load_image(args.img_in)
img_rgb_out = model.predict(img_rgb)['img_out']

plt.imsave(args.img_out, img_rgb_out)
