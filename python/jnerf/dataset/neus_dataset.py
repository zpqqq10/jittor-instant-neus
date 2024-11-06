import jittor as jt
import cv2
import cv2 as cv
import numpy as np
import os
from glob import glob
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

from jnerf.utils.registry import DATASETS
from jnerf.utils.config import get_cfg
from tqdm import tqdm

# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose
                
# dilate the masks with cv2
def large_mask(mask, kernel = np.ones((3,3), np.uint8), iter = 1):
    mask_dilate = cv2.dilate(mask, kernel, iterations = iter)
    mask_dilate = (mask_dilate > 0.).astype(float)
    return mask_dilate


@DATASETS.register_module()
class NeuSDataset:
    def __init__(self, dataset_dir, render_cameras_name, object_cameras_name):
        super(NeuSDataset, self).__init__()

        self.cfg = get_cfg()
        print('Load data: Begin . if_dilate : ', self.cfg.if_dilate)


        self.data_dir = dataset_dir
        self.render_cameras_name = render_cameras_name
        self.object_cameras_name = object_cameras_name

        self.camera_outside_sphere = True #conf.get_bool('camera_outside_sphere', default=True)
        # self.scale_mat_scale = 1.1 #conf.get_float('scale_mat_scale', default=1.1)

        camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
        self.camera_dict = camera_dict
        self.images_lis = sorted(glob(os.path.join(self.data_dir, 'image/*.png')))

        # print(self.images_lis)
        # exit(0)
        self.n_images = len(self.images_lis)
        # self.images = jt.stack([jt.array(cv.imread(im_name)) for im_name in self.images_lis]) / 255.0
        self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.png')))
        # self.masks = jt.stack([jt.array(cv.imread(im_name)) for im_name in self.masks_lis]) / 255.0

        np_images = np.stack([ cv.imread(im_name) for im_name in self.images_lis]) / 255.0
        np_masks = np.stack([ cv.imread(im_name) for im_name in self.masks_lis]) / 255.0
        np_dilate_masks = np.zeros(np_masks.shape)

        if self.cfg.if_dilate:

            if not os.path.exists(os.path.join(self.data_dir, 'dilate_mask')):
                os.mkdir(os.path.join(self.data_dir, 'dilate_mask'))


            self.dilate_masks_lis = sorted(glob(os.path.join(self.data_dir, 'dilate_mask/*.png')))
            if len(self.dilate_masks_lis) != self.n_images:
                print('generating dilate mask ')    
                dlt = 41
                iter = 1
                if self.cfg.mask_dlt:
                    dlt = self.cfg.mask_dlt[0]
                    iter = self.cfg.mask_dlt[1]
                
                mid = dlt // 2 
                kernel = np.ones((dlt,dlt), np.uint8)
                for i in range(dlt):
                    for j in range(dlt):
                        if (i-mid)**2 + (j-mid)**2 > (mid+0.5)**2:
                            kernel[i, j] = 0
                print('save dilate_mask at :', os.path.join(self.data_dir, 'dilate_mask'))

                for i in tqdm(range(np_masks.shape[0])):
                    np_dilate_masks[i] =  large_mask(np_masks[i], kernel = kernel, iter = iter)
                    n_mask_path = os.path.join(self.data_dir, 'dilate_mask', os.path.basename(self.masks_lis[i]))
                    cv2.imwrite(n_mask_path, (np_dilate_masks[i] * 255.).astype(np.uint8))

            else:
                print('reading dilate_mask at : ',os.path.join(self.data_dir, 'dilate_mask'))
                
                np_dilate_masks = np.stack([ cv.imread(im_name) for im_name in self.dilate_masks_lis]) / 255.0

            self.np_dilate_masks = np_dilate_masks
            self.np_dilate_masks = self.np_dilate_masks[:, :, :, :1]

            # self.np_dilate_masks =  (self.np_dilate_masks > 0.1).astype(float)

        self.np_images = np_images
        self.np_masks = np_masks

        self.np_masks = self.np_masks[:, :, :, :1]
        self.np_masks =  (self.np_masks > 0.1).astype(float)


        # to ignore some really bad images for input
        ignore_idx = []
        if self.cfg.ignore_index:
            ignore_idx = self.cfg.ignore_index
            print("ignore index : ", ignore_idx)
        
        save_index = np.zeros(self.n_images)
        now_num = 0
        for i in range(self.n_images):
            if i not in ignore_idx:
                save_index[now_num] = i
                now_num += 1

        self.save_idx_num = now_num
        self.save_index = save_index

        self.img_weight = self.np_images.shape[2]
        self.img_height = self.np_images.shape[1]
        self.pixels = self.img_height * self.img_weight

        ### change 0926
        mask_loc_list = np.zeros((self.n_images, self.pixels, 2))
        mask_num = np.zeros(self.n_images)

        if self.cfg.if_dilate:
            for i in range(self.np_dilate_masks.shape[0]):
                n_mask = self.np_dilate_masks[i]
                loc = (n_mask > 0.1).nonzero()
                
                temp_num = loc[0].shape[0]
                mask_loc_list[i, :temp_num, 0] = loc[0]
                mask_loc_list[i, :temp_num, 1] = loc[1]
                
                mask_num[i] = temp_num
                # print('num : ',i, temp_num)
            
            self.mask_loc_list = mask_loc_list
            self.mask_num = mask_num

        # world_mat is a projection matrix from world to image
        self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.scale_mats_np = []

        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        self.intrinsics_all = []
        self.pose_all = []

        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(jt.Var(intrinsics).float())
            self.pose_all.append(jt.Var(pose).float())

        #     K = intrinsics
        #     c2w = pose
        #     fx, fy, cx, cy = K[0,0] * self.factor, K[1,1] * self.factor, K[0,2] * self.factor, K[1,2] * self.factor
        #     directions = get_ray_directions(w, h, fx, fy, cx, cy)
        #     self.directions.append(directions)
        
        # for i in range(n_images):
        #     world_mat, scale_mat = cams[f'world_mat_{i}'], cams[f'scale_mat_{i}']
        #     # if i < 3 : 
        #     #     print(' i world_mat: ',i ,world_mat, scale_mat)
        #     # else:
        #     #     exit(0)
        #     P = (world_mat @ scale_mat)[:3,:4]
        #     K, c2w = load_K_Rt_from_P(P)
        #     fx, fy, cx, cy = K[0,0] * self.factor, K[1,1] * self.factor, K[0,2] * self.factor, K[1,2] * self.factor
        #     directions = get_ray_directions(w, h, fx, fy, cx, cy)
        #     self.directions.append(directions)


        # self.images = jt.Var(self.images_np.astype(np.float32))  # [n_images, H, W, 3]
        # self.masks  = jt.Var(self.masks_np.astype(np.float32))   # [n_images, H, W, 3]
        self.intrinsics_all = jt.stack(self.intrinsics_all)   # [n_images, 4, 4]
        self.intrinsics_all_inv = jt.linalg.inv(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = jt.stack(self.pose_all)  # [n_images, 4, 4]
        self.H, self.W = self.np_images.shape[1], self.np_images.shape[2]
        self.image_pixels = self.H * self.W

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
        # Object scale mat: region of interest to **extract mesh**
        object_scale_mat = np.load(os.path.join(self.data_dir, self.object_cameras_name))['scale_mat_0']
        object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]

        print('Load data: End')

    def jt_matmul(self,a,b):

        h,w,_,_ = b.shape
        a = a.expand(h,w,1,1)

        return jt.matmul(a,b)

    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = jt.linspace(0, self.W - 1, self.W // l)
        ty = jt.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = jt.meshgrid(tx, ty)
        p = jt.stack([pixels_x, pixels_y, jt.ones_like(pixels_y)], dim=-1) # W, H, 3
        p = self.jt_matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None])  # W, H, 3
        p = p.squeeze(dim=3)
        rays_v = p / jt.norm(p, p=2, dim=-1, keepdim=True, eps=1e-6)  # W, H, 3
        rays_v = self.jt_matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None])  # W, H, 3
        rays_v = rays_v.squeeze(dim=3)
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_rays_at_wmask(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera. with mask return
        """
        
        loc = (self.np_dilate_masks[img_idx] < 0.5).nonzero()
        # print('loc w mask : ',loc)
        pixels_x = jt.Var(loc[1]).reshape(1, -1).float()
        pixels_y = jt.Var(loc[0]).reshape(1, -1).float()

        np_pixels_x = pixels_x.view(-1).numpy().astype(int)
        np_pixels_y = pixels_y.view(-1).numpy().astype(int)
        np_img_idx = img_idx

        mask = jt.Var(self.np_dilate_masks[np_img_idx, np_pixels_y, np_pixels_x])
        
        assert(mask.sum() == 0)

        p = jt.stack([pixels_x, pixels_y, jt.ones_like(pixels_y)], dim=-1) # W, H, 3
        p = self.jt_matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None])  # W, H, 3
        p = p.squeeze(dim=3)
        rays_v = p / jt.norm(p, p=2, dim=-1, keepdim=True, eps=1e-6)  # W, H, 3
        rays_v = self.jt_matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None])  # W, H, 3
        rays_v = rays_v.squeeze(dim=3)
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)
    
    def get_batch_random_rays(self, batch_size, apply_mask = False):
        """
        Generate random rays at world space from one camera.
        """
        
        img_idx = jt.randint(low=0, high=self.save_idx_num, shape=[batch_size])
        img_idx = jt.Var(self.save_index[img_idx])
        
        if self.cfg.cal_bbox:
            ##### hidden in this version
            pixels_x = jt.randint(low=0, high=self.W, shape=[batch_size])
            pixels_y = jt.randint(low=0, high=self.H, shape=[batch_size])
            np_pixels_x = pixels_x.numpy().astype(int)
            np_pixels_y = pixels_y.numpy().astype(int)
            np_img_idx = img_idx.numpy().astype(int)
        elif self.cfg.if_dilate:

            ### select only mask area, ...
            np_img_idx = img_idx.numpy().astype(int)
            
            index = jt.randint(low=0, high=2**30, shape=[batch_size])
            mask_num = jt.Var(self.mask_num[np_img_idx])

            # print('mask_num : ',mask_num, mask_num.shape)
            
            mask_index = index % mask_num
            np_mask_index = mask_index.numpy().astype(int)


            np_pixels_x = (self.mask_loc_list[np_img_idx, np_mask_index, 1]).astype(int)
            np_pixels_y = (self.mask_loc_list[np_img_idx, np_mask_index, 0]).astype(int)
            pixels_x = jt.Var(np_pixels_x)
            pixels_y = jt.Var(np_pixels_y)
        else:
            raise NotImplementedError

            
        # print(type(img_idx.numpy()), type(pixels_x.numpy()), type(pixels_y.numpy()))
        # color = self.images[img_idx, pixels_y, pixels_x]
        # mask = self.masks[img_idx, pixels_y, pixels_x]
        color = jt.Var(self.np_images[np_img_idx, np_pixels_y, np_pixels_x])
        ### change 0926
        mask = jt.Var(self.np_masks[np_img_idx, np_pixels_y, np_pixels_x])
        # mask = jt.Var(self.np_dilate_masks[np_img_idx, np_pixels_y, np_pixels_x])

        # color = self.images[img_idx].squeeze(dim=0)[(pixels_y, pixels_x)]    # batch_size, 3
        # mask = self.masks[img_idx].squeeze(dim=0)[(pixels_y, pixels_x)]      # batch_size, 3
        point = jt.stack([pixels_x, pixels_y, jt.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        bs = point.shape[0]

        point = jt.matmul(self.intrinsics_all_inv[img_idx, :3, :3], point[:, :, None]) # batch_size, 3
        point = point.squeeze(dim=2)
        rays_v = point / jt.norm(point, p=2, dim=-1, keepdim=True, eps=1e-6)    # batch_size, 3
        
        rays_v = jt.matmul(self.pose_all[img_idx, :3, :3], rays_v[:, :, None])  # batch_size, 3
        rays_v = rays_v.squeeze(dim=2)
        
        rays_o = self.pose_all[img_idx, :3, 3] # batch_size, 3
        
        if apply_mask == True:
            color = color *  mask

        # print(mask.shape, mask.sum(), color.shape)
        # print('mask:', mask)
        # exit(0)
        

        return jt.concat([rays_o, rays_v, color, mask], dim=-1)    # batch_size, 10

    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = jt.randint(low=0, high=self.W, shape=[batch_size])
        pixels_y = jt.randint(low=0, high=self.H, shape=[batch_size])


        np_pixels_x = pixels_x.numpy().astype(int)
        np_pixels_y = pixels_y.numpy().astype(int)
        np_img_idx = img_idx.numpy().astype(int)
        

        # color = self.images[img_idx].squeeze(dim=0)[(pixels_y, pixels_x)]    # batch_size, 3
        # mask = self.masks[img_idx].squeeze(dim=0)[(pixels_y, pixels_x)]      # batch_size, 3
        color = jt.Var(self.np_images[np_img_idx].squeeze(dim=0)[(np_pixels_y, np_pixels_x)] )   # batch_size, 3
        mask = jt.Var(self.np_masks[np_img_idx].squeeze(dim=0)[(np_pixels_y, np_pixels_x)])      # batch_size, 3

        point = jt.stack([pixels_x, pixels_y, jt.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        bs = point.shape[0]
        point = jt.matmul(self.intrinsics_all_inv[img_idx, :3, :3].expand(bs,1,1), point[:, :, None]) # batch_size, 3
        point = point.squeeze(dim=2)
        rays_v = point / jt.norm(point, p=2, dim=-1, keepdim=True, eps=1e-6)    # batch_size, 3
        rays_v = jt.matmul(self.pose_all[img_idx, :3, :3].expand(bs,1,1), rays_v[:, :, None])  # batch_size, 3
        rays_v = rays_v.squeeze(dim=2)
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape).squeeze(dim=0) # batch_size, 3
        return jt.concat([rays_o, rays_v, color, mask[:, :1]], dim=-1)    # batch_size, 10

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        l = resolution_level
        tx = jt.linspace(0, self.W - 1, self.W // l)
        ty = jt.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = jt.meshgrid(tx, ty)
        p = jt.stack([pixels_x, pixels_y, jt.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = jt.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / jt.norm(p, p=2, dim=-1, keepdim=True, eps=1e-6)  # W, H, 3
        trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = jt.Var(pose[:3, :3])
        trans = jt.Var(pose[:3, 3])
        rays_v = jt.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def near_far_from_sphere(self, rays_o, rays_d):
        a = jt.sum(rays_d**2, dim=-1, keepdims=True)
        b = 2.0 * jt.sum(rays_o * rays_d, dim=-1, keepdims=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        # print('a b mid : ',mid.min(), mid.max())
        # exit(0)
        return near, far

    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx])
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)

