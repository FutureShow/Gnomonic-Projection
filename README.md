# Gnomonic-Projection
Realize gnomonic projection for extracting viewport from panoramic image

## GnomonicCoordinates:
Build coordinate according to panoramic image size. By the way, resolution of fov can be specified. Finally, use generate_grid((row, col)) for get grid_x and grid_y of kernel responding to source image.

## SphereProjection:
Projection implementation of GnomonicCoordinates. You can input panoramic image and output each fov according to the center location of fov in source panoramic image.

## BackProjection_Kernel:
Back-projection implementation GnomonicCoordinates for obtaining a rectangle to cover the back-projected fov in panoramic image. Here, it only considers the polar angle because we only want to know the result of back-projected fov which is not related to horizontal position.
