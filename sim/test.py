import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw


img_size = 600
board = Image.new(mode="I", size=(img_size, img_size), color=0)
draw = ImageDraw.Draw(board)
draw.line((0, 0, 200, 100), fill = 1)
image = np.asarray(board)
plt.figure()
plt.imshow(image)
plt.show()