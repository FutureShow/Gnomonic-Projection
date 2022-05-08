# using python 3.5

import numpy as np
import torch

MAX_KERNEL_SIZE = 65
assert MAX_KERNEL_SIZE % 2 == 1
HALF_KERNEL_SIZE = int(MAX_KERNEL_SIZE/2)


def get_bilinear_index(grid_x, grid_y):
    ix0 = np.floor(grid_x).astype(int)
    ix1 = ix0 + 1
    iy0 = np.floor(grid_y).astype(int)
    iy1 = iy0 + 1
    c00 = (grid_x - ix0) * (grid_y - iy0)
    c01 = (grid_x - ix0) * (iy1 - grid_y)
    c10 = (ix1 - grid_x) * (grid_y - iy0)
    c11 = (ix1 - grid_x) * (iy1 - grid_y)
    return ix0, ix1, iy0, iy1, c00, c01, c10, c11
        

class GnomonicCoordinates(object):
    def __init__(self, img_h, img_w, kernel_size=3):
        '''
        kernel_size: length or (h, w)
        '''
        
        assert isinstance(kernel_size, (int, tuple, list))
        if isinstance(kernel_size, (tuple, list)):
            assert len(kernel_size) == 2
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.img_h = img_h
        self.img_w = img_w
        
        h, w = kernel_size[0], kernel_size[1]
        grid_x, grid_y = np.meshgrid(range(w), range(h-1, -1, -1))
        center_x, center_y = w / 2 - 0.5, h / 2 - 0.5
        self.grid_x = grid_x.astype(np.float64) - center_x
        self.grid_y = grid_y.astype(np.float64) - center_y

        self.fix_center = None
        if kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1:
            self.fix_center = (int(center_y), int(center_x))
            self.grid_x[self.fix_center] = 1
            self.grid_y[self.fix_center] = 1
        
        self.rho = np.sqrt(self.grid_x**2 + self.grid_y**2)
        self.unit_dist = np.tan(np.pi / self.img_h)
    
    
    def specify_fov_resolution(self, fov_angle, fov_size):
        '''
        fov_angle, fov_size: You can set both them to specify fov resolution; if one of them is None, fov will follow the projection grid from the unit ball (radius = 1)
        '''
        # try to change resolution
        if isinstance(fov_angle, (int, float)) and isinstance(fov_size, int):
            self.unit_dist = np.tan(np.deg2rad(fov_angle) / fov_size)
        else:
            print('Reset fov resolution as tan(np.pi / img_h)! Your fov_angle and fov_size are wrong!')
            self.unit_dist = np.tan(np.pi / self.img_h)
            
    
    def generate_grid(self, center_loc):
        '''
        center_loc: (row, col)
        '''
        
        assert len(center_loc) == 2
        assert (self.img_h > center_loc[0] >= 0) and (self.img_w > center_loc[1] >= 0)
        
        grid_x, grid_y, rho = self.grid_x * self.unit_dist, self.grid_y * self.unit_dist, self.rho * self.unit_dist
        v = np.arctan(rho)
        
        phi, theta = self.coor2rad(center_loc)
        new_phi = np.arcsin(np.cos(v) * np.sin(phi) + grid_y * np.sin(v) * np.cos(phi) / rho)

        INDn = np.array([False]*self.kernel_size[0])
        max_idx, min_idx = np.argmax(new_phi[::-1,int(self.kernel_size[0]/2-1)]), np.argmin(new_phi[:,int(self.kernel_size[0]/2-1)])
        if max_idx not in [0, self.kernel_size[0]-1] or min_idx not in [0, self.kernel_size[0]-1]:
            # When maximum or minimum value appears in the middle, 
            # it indicates that it has crossed the pole. At this time, theta plus pi is needed.
            if max_idx not in [0, self.kernel_size[0]-1]:
                INDn[:self.kernel_size[0]-max_idx-1] = True
            else:
                INDn[min_idx+1:] = True
        
        new_theta = theta + np.arctan(grid_x * np.sin(v) / (rho * np.cos(phi) * np.cos(v) - grid_y * np.sin(phi) * np.sin(v)))
        new_theta[INDn, :] += np.pi
        y, x = self.rad2coor((new_phi, new_theta))

        if self.fix_center:
            y[self.fix_center] = center_loc[0]
            x[self.fix_center] = center_loc[1]
        
        return x, y
    
    def coor2rad(self, coor_loc):
        assert len(coor_loc) == 2
        y, x = coor_loc
        phi = np.pi / 2 - y / self.img_h * np.pi
        theta = x / self.img_w * 2 * np.pi - np.pi
        return phi, theta

    def rad2coor(self, rad_loc):
        assert len(rad_loc) == 2
        phi, theta = rad_loc
        y = (-phi + np.pi / 2) * self.img_h / np.pi
        x = (theta + np.pi) * self.img_w / 2 / np.pi
        x = (x + self.img_w) % self.img_w
        return y, x
        
        

class SphereProjection(GnomonicCoordinates):
    def __init__(self, img, kernel_size=3):
        super(SphereProjection, self).__init__(img.shape[0], img.shape[1], kernel_size)
        self.img = img.copy()
        self.ch = img.shape[2]
        
        # self.specify_fov_resolution(fov_angle, fov_size)
        self.target_grid_x, self.target_grid_y = self.generate_grid((0,0))
        self.last_y = 0

    
    def gen_kernel(self, center_loc):
        assert len(center_loc) == 2
        assert (self.img_h > center_loc[0] >= 0) and (self.img_w > center_loc[1] >= 0)
        if center_loc[0] != self.last_y:
            self.target_grid_x, self.target_grid_y = self.generate_grid((center_loc[0],0))
            self.last_y = center_loc[0]

        _grid_x = (self.target_grid_x + center_loc[1] + self.img_w) % self.img_w
        ix0, ix1, iy0, iy1, c00, c01, c10, c11 = get_bilinear_index(_grid_x, self.target_grid_y)
        
        ix0 = ix0 % self.img_w
        ix1 = ix1 % self.img_w
        iy0 = np.clip(iy0, 0, self.img_h-1)
        iy1 = np.clip(iy1, 0, self.img_h-1)
        
        p00 = self.img[iy0, ix0]
        p01 = self.img[iy1, ix0]
        p10 = self.img[iy0, ix1]
        p11 = self.img[iy1, ix1]
        
        kernel = np.zeros((self.kernel_size[0], self.kernel_size[1], self.ch))
        for i in range(self.ch):
            kernel[..., i] = p00[..., i] * c00 + p01[..., i] * c01 + p10[..., i] * c10 + p11[..., i] * c11
        return kernel

# maybe modify
class BackProjection_Kernel(GnomonicCoordinates):
    def __init__(self, img_h, img_w, kernel):
        assert isinstance(kernel, np.ndarray) and len(kernel.shape) == 4
        self.out_ch, self.in_ch, kernel_h, kernel_w = kernel.shape
        super(BackProjection_Kernel, self).__init__(img_h, img_w, (kernel_h, kernel_w))
        self.kernel = kernel.copy()

        
    def back_projection(self, y):
        assert self.img_h > y >= 0
        center_loc = (y, self.img_w/2)
        (target_grid_x, target_grid_y), target_kernel = self._get_target_zero_kernel(center_loc)
        
        grid_phi, grid_theta = self.coor2rad((target_grid_y, target_grid_x))
        center_phi, center_theta = self.coor2rad(center_loc)
        diff_theta = grid_theta - center_theta
        down = np.sin(center_phi)*np.sin(grid_phi)+np.cos(center_phi)*np.cos(grid_phi)*np.cos(diff_theta)
        target_grid_x = (np.cos(grid_phi)*np.sin(diff_theta)) / down
        target_grid_y = (np.cos(center_phi)*np.sin(grid_phi)-np.sin(center_phi)*np.cos(grid_phi)*np.cos(diff_theta)) / down
        
        target_grid_x = target_grid_x / self.unit_dist + self.kernel_size[1] / 2 - 0.5
        target_grid_y = target_grid_y / self.unit_dist + self.kernel_size[0] / 2 - 0.5
        
        ix0, ix1, iy0, iy1, c00, c01, c10, c11 = get_bilinear_index(target_grid_x, target_grid_y)
        # c00, c01, c10, c11 = torch.from_numpy(c00.astype('float32')), torch.from_numpy(c01.astype('float32')), \
                             # torch.from_numpy(c10.astype('float32')), torch.from_numpy(c11.astype('float32'))
        p00 = self._get_sample_point(iy0, ix0)
        p01 = self._get_sample_point(iy1, ix0)
        p10 = self._get_sample_point(iy0, ix1)
        p11 = self._get_sample_point(iy1, ix1)
        
        for i in range(self.out_ch):
            for j in range(self.in_ch):
                target_kernel[i, j] = p00[i, j] * c00 + p01[i, j] * c01 + p10[i, j] * c10 + p11[i, j] * c11
        return target_kernel
        
        
    def _get_sample_point(self, iy, ix):
        valid_y = np.logical_and(self.kernel_size[0] > iy, iy >= 0)
        valid_x = np.logical_and(self.kernel_size[1] > ix, ix >= 0)
        valid_idx = np.logical_and(valid_y, valid_x)
        idx_x, idx_y = np.meshgrid(range(ix.shape[1]), range(ix.shape[0]))

        val = np.zeros((self.out_ch, self.in_ch, ix.shape[0], ix.shape[1]))
        val[..., idx_y[valid_idx], idx_x[valid_idx]] = self.kernel[..., iy[valid_idx], ix[valid_idx]]
        return val
    
    
    def _get_target_zero_kernel(self, center_loc):
        assert (self.img_h > center_loc[0] >= 0) and (self.img_w > center_loc[1] >= 0)
        kernel_grid_x, kernel_grid_y = self.generate_grid(center_loc)

        x_min, x_max = self._cut_range(kernel_grid_x.min(), kernel_grid_x.max(), center_loc[1])
        y_min, y_max = self._cut_range(kernel_grid_y.min(), kernel_grid_y.max(), center_loc[0])
        print(x_min, x_max, x_max-x_min, ' | ', y_min, y_max, y_max-y_min, ' | ', center_loc)
        return np.meshgrid(range(x_min, x_max), range(y_min, y_max)), \
               np.zeros((self.out_ch, self.in_ch, y_max-y_min, x_max-x_min))
    
    def _cut_range(self, min_, max_, center):
        min_ = np.floor(min_)
        max_ = np.ceil(max_) + 1
        if (max_ - min_) % 2 == 0: # keep odd
            max_ += 1
        if (max_ - min_) > MAX_KERNEL_SIZE: # less than 65
            min_ = center - HALF_KERNEL_SIZE
            max_ = center + HALF_KERNEL_SIZE + 1
        return np.int16(min_), np.int16(max_)

if __name__ == '__main__':
    import cv2
    img = cv2.imread('example_projection/drive_city_small.jpg')
    t=SphereProjection(img, kernel_size = 373)
    target_center = (0, 320)
    kernel=t.gen_kernel(target_center)
    cv2.imwrite('test.jpg',kernel)
