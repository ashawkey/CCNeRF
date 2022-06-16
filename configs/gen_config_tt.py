
templates = {

'hybrid': 
'''
dataset_name = tankstemple
datadir = ./data/TanksAndTemple/{}
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

}

for name in ['Barn', 'Caterpillar', 'Family', 'Ignatius', 'Truck']:
    for model in templates.keys():
        with open(f'{name}_{model}.txt', 'w') as f:
            f.write(templates[model].format(name, name))
