import moxel

model = moxel.Model('strin/colorization:latest', where='localhost')

img_in = moxel.space.Image.from_file('demo/imgs/ansel_adams3.jpg')

result = model.predict(img_in=img_in)

img_out = result['img_out'].to_PIL().save('output.png')
