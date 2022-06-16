from .tensorBase import *

class CCNeRF(TensorBase):
    def __init__(self, aabb, gridSize, device, **kargs):
        super().__init__(aabb, gridSize, device, **kargs)

    def init_svd_volume(self):

        self.sigma_vec = nn.ParameterList()
        self.sigma_mat = nn.ParameterList()

        if self.rank_density[0][0] > 0:
            for i in range(3):
                vec_id = self.vecMode[i]
                w = torch.randn(self.rank_density[0][0], self.resolution[vec_id]) * 0.2 # [R, H]
                self.sigma_vec.append(nn.Parameter(w.view(1, self.rank_density[0][0], self.resolution[vec_id], 1))) # [1, R, H, 1]
        
        if self.rank_density[0][1] > 0:
            for i in range(3):    
                mat_id_0, mat_id_1 = self.matMode[i]
                w = torch.randn(self.rank_density[0][1], self.resolution[mat_id_1] * self.resolution[mat_id_0]) * 0.2 # [R, HW]
                self.sigma_mat.append(nn.Parameter(w.view(1, self.rank_density[0][1], self.resolution[mat_id_1], self.resolution[mat_id_0]))) # [1, R, H, W]
            
        self.sigma_vec.to(self.device)
        self.sigma_mat.to(self.device)

        self.color_vec = nn.ParameterList() 
        self.S_vec = nn.ParameterList()

        last_rank = 0
        for k in range(self.K[0]):

            rank = self.rank_vec[0][k] - last_rank

            if rank > 0:

                for i in range(3):                
                    vec_id = self.vecMode[i]
                    w = torch.randn(rank, self.resolution[vec_id]) * 0.2 # [R, H]
                    self.color_vec.append(nn.Parameter(w.view(1, rank, self.resolution[vec_id], 1))) # [1, R, H, 1]

                w = torch.ones(self.out_dim, rank)
                torch.nn.init.kaiming_normal_(w)
                self.S_vec.append(nn.Parameter(w))

            last_rank = self.rank_vec[0][k]

        self.color_vec.to(self.device)
        self.S_vec.to(self.device)

        self.color_mat = nn.ParameterList() 
        self.S_mat = nn.ParameterList()

        last_rank = 0
        for k in range(self.K[0]):
            rank = self.rank_mat[0][k] - last_rank

            if rank > 0:
                for i in range(3):
                    
                    mat_id_0, mat_id_1 = self.matMode[i]
                    w = torch.randn(rank, self.resolution[mat_id_1] * self.resolution[mat_id_0]) * 0.2 # [R, HW]
                    self.color_mat.append(nn.Parameter(w.view(1, rank, self.resolution[mat_id_1], self.resolution[mat_id_0]))) # [1, R, H, W]

                w = torch.ones(self.out_dim, rank)
                torch.nn.init.kaiming_normal_(w)
                self.S_mat.append(nn.Parameter(w))

            last_rank = self.rank_mat[0][k]

        self.color_mat.to(self.device)
        self.S_mat.to(self.device)
    
    
    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        grad_vars = []
        
        grad_vars.extend([
            {'params': self.sigma_vec, 'lr': lr_init_spatialxyz},
            {'params': self.color_vec, 'lr': lr_init_spatialxyz},
            {'params': self.S_vec, 'lr': lr_init_network},
        ])
    
        grad_vars.extend([
            {'params': self.sigma_mat, 'lr': lr_init_spatialxyz},
            {'params': self.color_mat, 'lr': lr_init_spatialxyz},
            {'params': self.S_mat, 'lr': lr_init_network},
        ])
        return grad_vars


    def compute_density_features(self, xyz_sampled, oid=0):

        prefix = xyz_sampled.shape[:-1]
        N = np.prod(prefix)

        feat = 0

        if self.rank_density[oid][0] > 0:

            offset = self.offset_density_vec[oid]

            vec_coord = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
            vec_coord = torch.stack((torch.zeros_like(vec_coord), vec_coord), dim=-1).detach().view(3, -1, 1, 2)

            vec_feat = F.grid_sample(self.sigma_vec[offset + 0], vec_coord[[0]], align_corners=True).view(-1, N) * \
                    F.grid_sample(self.sigma_vec[offset + 1], vec_coord[[1]], align_corners=True).view(-1, N) * \
                    F.grid_sample(self.sigma_vec[offset + 2], vec_coord[[2]], align_corners=True).view(-1, N) # [r, N]
            
            feat = feat + vec_feat.sum(0)
        
        if self.rank_density[oid][1] > 0:

            offset = self.offset_density_mat[oid]

            mat_coord = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2) # [3, N, 1, 2]

            mat_feat = F.grid_sample(self.sigma_mat[offset + 0], mat_coord[[0]], align_corners=True).view(-1, N) * \
                    F.grid_sample(self.sigma_mat[offset + 1], mat_coord[[1]], align_corners=True).view(-1, N) * \
                    F.grid_sample(self.sigma_mat[offset + 2], mat_coord[[2]], align_corners=True).view(-1, N) # [r, N]

            feat = feat + mat_feat.sum(0)
        
        feat = feat.view(*prefix)

        return feat        


    def compute_features(self, xyz_sampled, K=-1, is_train=False, oid=0):
        # xyz_sampled: [N, M, 3] or [N, 3]
        # return: [K, N, M, out_dim] or [K, N, out_dim]

        prefix = xyz_sampled.shape[:-1]
        N = np.prod(prefix)

        vec_coord = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        vec_coord = torch.stack((torch.zeros_like(vec_coord), vec_coord), dim=-1).detach().view(3, -1, 1, 2)

        mat_coord = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2) # [3, N, 1, 2]

        # calculate first K blocks
        if K <= 0:
            K = self.K[oid]
            
        # loop all blocks 
        if is_train:
            outputs = []

        last_y = None

        offset_vec = self.offset_vec[oid]
        offset_mat = self.offset_mat[oid]

        for k in range(K):

            y = 0

            if self.use_vec[oid][k]:
                vec_feat = F.grid_sample(self.color_vec[3 * offset_vec + 0], vec_coord[[0]], align_corners=True).view(-1, N) * \
                           F.grid_sample(self.color_vec[3 * offset_vec + 1], vec_coord[[1]], align_corners=True).view(-1, N) * \
                           F.grid_sample(self.color_vec[3 * offset_vec + 2], vec_coord[[2]], align_corners=True).view(-1, N) # [r, N]

                y = y + (self.S_vec[offset_vec] @ vec_feat)

                offset_vec += 1

            if self.use_mat[oid][k]:
                mat_feat = F.grid_sample(self.color_mat[3 * offset_mat + 0], mat_coord[[0]], align_corners=True).view(-1, N) * \
                           F.grid_sample(self.color_mat[3 * offset_mat + 1], mat_coord[[1]], align_corners=True).view(-1, N) * \
                           F.grid_sample(self.color_mat[3 * offset_mat + 2], mat_coord[[2]], align_corners=True).view(-1, N) # [r, N]

                y = y + (self.S_mat[offset_mat] @ mat_feat) # [out_dim, N]

                offset_mat += 1

            if last_y is not None:
                y = y + last_y

            if is_train:
                outputs.append(y)

            last_y = y
        
        if is_train:
            outputs = torch.stack(outputs, dim=0).permute(0, 2, 1).contiguous().view(K, *prefix, -1) # [K, out_dim, NM] --> [K, N, M, out_dim]
        else:
            outputs = last_y.permute(1, 0).contiguous().view(*prefix, -1) # [out_dim, NM] --> [N, M, out_dim]
        
        return outputs


    
    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        
        for i in range(len(self.color_vec)):
            vec_id = self.vecMode[i % 3]
            self.color_vec[i] = nn.Parameter(F.interpolate(self.color_vec[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))

        for i in range(len(self.color_mat)):
            mat_id_0, mat_id_1 = self.matMode[i % 3]
            self.color_mat[i] = nn.Parameter(F.interpolate(self.color_mat[i].data, size=(res_target[mat_id_1], res_target[mat_id_0]), mode='bilinear', align_corners=True))

        for i in range(3):
            
            if self.rank_density[0][0] > 0:
                vec_id = self.vecMode[i % 3]
                self.sigma_vec[i] = nn.Parameter(F.interpolate(self.sigma_vec[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))

            if self.rank_density[0][1] > 0:
                mat_id_0, mat_id_1 = self.matMode[i % 3]
                self.sigma_mat[i] = nn.Parameter(F.interpolate(self.sigma_mat[i].data, size=(res_target[mat_id_1], res_target[mat_id_0]), mode='bilinear', align_corners=True))
        
        self.update_stepSize(res_target)
        print(f'[INFO] upsample volume grid: {res_target}')

    @torch.no_grad()
    def shrink(self, new_aabb):
        
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units

        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)

        for i in range(len(self.color_vec)):
            vec_id = self.vecMode[i % 3]
            self.color_vec[i] = nn.Parameter(self.color_vec[i].data[...,t_l[vec_id]:b_r[vec_id], :])

        for i in range(len(self.color_mat)):
            mat_id_0, mat_id_1 = self.matMode[i % 3]
            self.color_mat[i] = nn.Parameter(self.color_mat[i].data[...,t_l[mat_id_1]:b_r[mat_id_1],t_l[mat_id_0]:b_r[mat_id_0]])

        for i in range(3):

            if self.rank_density[0][0] > 0:
                vec_id = self.vecMode[i % 3]
                self.sigma_vec[i] = nn.Parameter(self.sigma_vec[i].data[...,t_l[vec_id]:b_r[vec_id], :])

            if self.rank_density[0][1] > 0:
                mat_id_0, mat_id_1 = self.matMode[i % 3]
                self.sigma_mat[i] = nn.Parameter(self.sigma_mat[i].data[...,t_l[mat_id_1]:b_r[mat_id_1],t_l[mat_id_0]:b_r[mat_id_0]])

        # shrink alpha grid
        alpha = self.alphaMask.alpha_volume[:, :, t_l[2]:b_r[2], t_l[1]:b_r[1], t_l[0]:b_r[0]]
        self.alphaMask = AlphaGridMask(self.device, alpha)

        newSize = b_r - t_l
        self.aabb = new_aabb
        self.update_stepSize((newSize[0], newSize[1], newSize[2]))

    def density_L1(self):
        total = 0
        for idx in range(3):
            if self.rank_density[0][0] > 0:
                total = total + torch.mean(torch.abs(self.sigma_vec[idx])) 
            if self.rank_density[0][1] > 0:
                total = total + torch.mean(torch.abs(self.sigma_mat[idx]))
        return total