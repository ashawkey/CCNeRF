
templates = {

'hybrid': 
'''
dataset_name = blender
datadir = ./data/nerf_synthetic/{}
basedir = ./log

n_iters = 30000
batch_size = 4096

N_voxel_init = 2097156 # 128**3
N_voxel_final = 27000000 # 300**3
upsamp_list = [2000, 3000, 4000, 5500, 7000]
update_AlphaMask_list = [2000, 4000]

N_vis = 5
vis_every = 10000

render_test = 1

rank_mat = [0, 4, 16, 32, 64]
rank_vec = [96, 96, 96, 96, 96]

rank_density = [96, 0]
degree = 4

expname = {}_hybrid
model_name = CCNeRF

fea2denseAct = softplus
fea2rgbAct = sigmoid

L1_weight_inital = 1e-5
L1_weight_rest = 1e-5

rm_weight_mask_thre = 1e-4

''',

'retrain_5': 
'''
dataset_name = blender
datadir = ./data/nerf_synthetic/{}
basedir = ./log

n_iters = 30000
batch_size = 4096

N_voxel_init = 2097156 # 128**3
N_voxel_final = 27000000 # 300**3
upsamp_list = [2000, 3000, 4000, 5500, 7000]
update_AlphaMask_list = [2000, 4000]

N_vis = 5
vis_every = 10000

render_test = 1

rank_mat = [0, 4, 16, 32, 64]
rank_vec = [96, 96, 96, 96, 96]

rank_density = [96, 0]
degree = 4

expname = {}_retrain_5
model_name = CCNeRF
residual = 0

fea2denseAct = softplus
fea2rgbAct = sigmoid

L1_weight_inital = 1e-5
L1_weight_rest = 1e-5

rm_weight_mask_thre = 1e-4

''',


'retrain_4': 
'''
dataset_name = blender
datadir = ./data/nerf_synthetic/{}
basedir = ./log

n_iters = 30000
batch_size = 4096

N_voxel_init = 2097156 # 128**3
N_voxel_final = 27000000 # 300**3
upsamp_list = [2000, 3000, 4000, 5500, 7000]
update_AlphaMask_list = [2000, 4000]

N_vis = 5
vis_every = 10000

render_test = 1

rank_mat = [0, 4, 16, 32]
rank_vec = [96, 96, 96, 96]

rank_density = [96, 0]
degree = 4

expname = {}_retrain_4
model_name = CCNeRF
residual = 0

fea2denseAct = softplus
fea2rgbAct = sigmoid

L1_weight_inital = 1e-5
L1_weight_rest = 1e-5

rm_weight_mask_thre = 1e-4

''',

'retrain_3': 
'''
dataset_name = blender
datadir = ./data/nerf_synthetic/{}
basedir = ./log

n_iters = 30000
batch_size = 4096

N_voxel_init = 2097156 # 128**3
N_voxel_final = 27000000 # 300**3
upsamp_list = [2000, 3000, 4000, 5500, 7000]
update_AlphaMask_list = [2000, 4000]

N_vis = 5
vis_every = 10000

render_test = 1

rank_mat = [0, 4, 16]
rank_vec = [96, 96, 96]

rank_density = [96, 0]
degree = 4

expname = {}_retrain_3
model_name = CCNeRF
residual = 0

fea2denseAct = softplus
fea2rgbAct = sigmoid

L1_weight_inital = 1e-5
L1_weight_rest = 1e-5

rm_weight_mask_thre = 1e-4

''',


'retrain_2': 
'''
dataset_name = blender
datadir = ./data/nerf_synthetic/{}
basedir = ./log

n_iters = 30000
batch_size = 4096

N_voxel_init = 2097156 # 128**3
N_voxel_final = 27000000 # 300**3
upsamp_list = [2000, 3000, 4000, 5500, 7000]
update_AlphaMask_list = [2000, 4000]

N_vis = 5
vis_every = 10000

render_test = 1

rank_mat = [0, 4]
rank_vec = [96, 96]

rank_density = [96, 0]
degree = 4

expname = {}_retrain_2
model_name = CCNeRF
residual = 0

fea2denseAct = softplus
fea2rgbAct = sigmoid

L1_weight_inital = 1e-5
L1_weight_rest = 1e-5

rm_weight_mask_thre = 1e-4

''',


'retrain_1': 
'''
dataset_name = blender
datadir = ./data/nerf_synthetic/{}
basedir = ./log

n_iters = 30000
batch_size = 4096

N_voxel_init = 2097156 # 128**3
N_voxel_final = 27000000 # 300**3
upsamp_list = [2000, 3000, 4000, 5500, 7000]
update_AlphaMask_list = [2000, 4000]

N_vis = 5
vis_every = 10000

render_test = 1

rank_mat = [0]
rank_vec = [96]

rank_density = [96, 0]
degree = 4

expname = {}_retrain_1
model_name = CCNeRF
residual = 0

fea2denseAct = softplus
fea2rgbAct = sigmoid

L1_weight_inital = 1e-5
L1_weight_rest = 1e-5

rm_weight_mask_thre = 1e-4

''',
}

for name in ['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship']:
    for model in templates.keys():
        with open(f'{name}_{model}.txt', 'w') as f:
            f.write(templates[model].format(name, name))
