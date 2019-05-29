# Gnomonic-Projection
Realize gnomonic projection for extracting viewport from panoramic image

## GnomonicCoordinates:
Build coordinate according to panoramic image size. By the way, resolution of fov can be specified. Finally, use generate_grid((row, col)) for get grid_x and grid_y of kernel responding to source image.

*Code Example*
'''
from GnomonicProjection import GnomonicCoordinates
g=GnomonicCoordinates(320, 640, (3,4))
x,y=g.generate_grid((0,320))
'''

## SphereProjection:
Projection implementation of GnomonicCoordinates. You can input panoramic image and output each fov according to the center location of fov in source panoramic image.

*Code Example*
'''
from GnomonicProjection import SphereProjection
import cv2

img = cv2.imread(image_path)
t=SphereProjection(img, kernel_size = 373)
target_center = (0, 320)
kernel=t.gen_kernel(target_center)
cv2.imwrite(out_path,kernel)
'''
### Example image
![source panoramic image](https://github.com/FutureShow/Gnomonic-Projection/blob/master/example_projection/drive_city_small.jpg)
![fov (row:0 col:320)](https://github.com/FutureShow/Gnomonic-Projection/blob/master/example_projection/drive_city_fov_0_320.jpg)
![fov (row:160 col:0)](https://github.com/FutureShow/Gnomonic-Projection/blob/master/example_projection/drive_city_fov_160_0.jpg)
![fov (row:160 col:160)](https://github.com/FutureShow/Gnomonic-Projection/blob/master/example_projection/drive_city_fov_160_160.jpg)
![fov (row:160 col:320)](https://github.com/FutureShow/Gnomonic-Projection/blob/master/example_projection/drive_city_fov_160_320.jpg)
![fov (row:160 col:480)](https://github.com/FutureShow/Gnomonic-Projection/blob/master/example_projection/drive_city_fov_160_480.jpg)
![fov (row:319 col:320)](https://github.com/FutureShow/Gnomonic-Projection/blob/master/example_projection/drive_city_fov_319_320.jpg)


## BackProjection_Kernel:
Back-projection implementation GnomonicCoordinates for obtaining a rectangle to cover the back-projected fov in panoramic image. Here, it only considers the polar angle because we only want to know the result of back-projected fov which is not related to horizontal position.

*Code Example*
'''
from GnomonicProjection import BackProjection_Kernel
import numpy as np
temp = np.ones((2,2,3,3)) # batch_size, channel, height, width
b=BackProjection_Kernel(320,640,temp)
new_kernel = b.back_projection(310)
'''
