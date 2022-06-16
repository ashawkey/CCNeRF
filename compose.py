import os
import sys
from tqdm.auto import tqdm
from opt import config_parser

from renderer import *
from utils import *

from dataLoader import dataset_dict
from scipy.spatial.transform import Rotation as Rot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

renderer = OctreeRender_trilinear_fast


def load_model(path, model_name):
    ckpt = torch.load(path, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(model_name)(**kwargs)
    tensorf.load(ckpt)

    return tensorf


@torch.no_grad()
def render_test(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)
    white_bg = test_dataset.white_bg
    ndc_ray = args.ndc_ray

    if args.ckpt is None:
        logfolder = f'{args.basedir}/{args.expname}'
        args.ckpt = f'{logfolder}/{args.expname}_5.th'
        print(f'[INFO] auto choose ckpt {args.ckpt}')

    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists! Init an empty tensorf.')

        #aabb = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]]).to(device)
        aabb = torch.tensor([[-2, -2, -2], [2, 2, 2]]).to(device)

        reso_cur = N_to_reso(500**3, aabb)
        near_far = [0.1, 6.0]

        tensorf = eval(args.model_name)(aabb, reso_cur, device,
                    rank_density=args.rank_density, rank_mat=args.rank_mat, rank_vec=args.rank_vec, degree=args.degree, near_far=near_far,
                    alphaMask_thres=args.alpha_mask_thre, density_shift=args.density_shift, distance_scale=args.distance_scale,
                    step_ratio=args.step_ratio, fea2denseAct=args.fea2denseAct, fea2rgbAct=args.fea2rgbAct)

        # init empty alpha grid
        tensorf.updateAlphaMask(gridSize=reso_cur, return_aabb=False)

        logfolder = f'{args.basedir}/{args.expname}'

    else:
        tensorf = load_model(args.ckpt, args.model_name)

        logfolder = os.path.dirname(args.ckpt)

    ### IMPORTANT: adjust stepsize for better rendering quality (but also slower)
    tensorf.step_ratio = 0.4
    tensorf.update_stepSize(tensorf.resolution)

    ### IMPORTANT: adjust distance scale
    tensorf.distance_scale = 25

    ######################################
    ### custom composition

    ### compress
    #tensorf.compress([48, 0])

    ### plot rank-importance map
    #tensorf.plot_rank()
    
    ### 8 mics with a drum
    if False:
        drums = load_model('./log/drums_hybrid/drums_hybrid_5.th', 'CCNeRF')
        T0 = np.array([
            [0.5, 0, 0, 0],
            [0, 0.5, 0, 0],
            [0, 0, 0.5, 0],
            [0, 0, 0, 1],
        ])
        tensorf.compose(drums, T0)

        mic = load_model('./log/mic_hybrid/mic_hybrid_5.th', 'CCNeRF')

        T0 = np.array([
            [0.3, 0, 0, -0.9],
            [0, 0.3, 0, 0],
            [0, 0, 0.3, 0],
            [0, 0, 0, 1],
        ])
        R0 = np.eye(4)
        R0[:3, :3] = Rot.from_euler('zyx', [90, 0, 0], degrees=True).as_matrix()
        T0 = T0 @ R0
        tensorf.compose(mic, T0, R0[:3, :3])

        T0 = np.array([
            [0.3, 0, 0, 0.9],
            [0, 0.3, 0, 0],
            [0, 0, 0.3, 0],
            [0, 0, 0, 1],
        ])
        R0 = np.eye(4)
        R0[:3, :3] = Rot.from_euler('zyx', [-90, 0, 0], degrees=True).as_matrix()
        T0 = T0 @ R0
        tensorf.compose(mic, T0, R0[:3, :3])

        T0 = np.array([
            [0.3, 0, 0, 0],
            [0, 0.3, 0, 0.9],
            [0, 0, 0.3, 0],
            [0, 0, 0, 1],
        ])
        R0 = np.eye(4)
        R0[:3, :3] = Rot.from_euler('zyx', [0, 0, 0], degrees=True).as_matrix()
        T0 = T0 @ R0
        tensorf.compose(mic, T0, R0[:3, :3])    

        T0 = np.array([
            [0.3, 0, 0, 0],
            [0, 0.3, 0, -0.9],
            [0, 0, 0.3, 0],
            [0, 0, 0, 1],
        ])
        R0 = np.eye(4)
        R0[:3, :3] = Rot.from_euler('zyx', [180, 0, 0], degrees=True).as_matrix()
        T0 = T0 @ R0
        tensorf.compose(mic, T0, R0[:3, :3])     

        # corners
        T0 = np.array([
            [0.3, 0, 0, -0.64],
            [0, 0.3, 0, -0.64],
            [0, 0, 0.3, 0],
            [0, 0, 0, 1],
        ])
        R0 = np.eye(4)
        R0[:3, :3] = Rot.from_euler('zyx', [135, 0, 0], degrees=True).as_matrix()
        T0 = T0 @ R0
        tensorf.compose(mic, T0, R0[:3, :3])

        T0 = np.array([
            [0.3, 0, 0, 0.64],
            [0, 0.3, 0, 0.64],
            [0, 0, 0.3, 0],
            [0, 0, 0, 1],
        ])
        R0 = np.eye(4)
        R0[:3, :3] = Rot.from_euler('zyx', [-45, 0, 0], degrees=True).as_matrix()
        T0 = T0 @ R0
        tensorf.compose(mic, T0, R0[:3, :3])
        
        T0 = np.array([
            [0.3, 0, 0, -0.64],
            [0, 0.3, 0, 0.64],
            [0, 0, 0.3, 0],
            [0, 0, 0, 1],
        ])
        R0 = np.eye(4)
        R0[:3, :3] = Rot.from_euler('zyx', [45, 0, 0], degrees=True).as_matrix()
        T0 = T0 @ R0
        tensorf.compose(mic, T0, R0[:3, :3])    

        T0 = np.array([
            [0.3, 0, 0, 0.64],
            [0, 0.3, 0, -0.64],
            [0, 0, 0.3, 0],
            [0, 0, 0, 1],
        ])
        R0 = np.eye(4)
        R0[:3, :3] = Rot.from_euler('zyx', [-135, 0, 0], degrees=True).as_matrix()
        T0 = T0 @ R0
        tensorf.compose(mic, T0, R0[:3, :3])  

    ### 3 chairs and a ficus around a hotdog
    if True:
        hotdog = load_model('./log/hotdog_hybrid/hotdog_hybrid_5.th', 'CCNeRF')
        chair = load_model('./log/chair_hybrid/chair_hybrid_5.th', 'CCNeRF')
        ficus = load_model('./log/ficus_hybrid/ficus_hybrid_5.th', 'CCNeRF')

        T0 = np.array([
            [0.4, 0, 0, 0],
            [0, 0.4, 0, 0],
            [0, 0, 0.4, 0.2],
            [0, 0, 0, 1],
        ])
        tensorf.compose(hotdog, T0)

        T0 = np.array([
            [0.6, 0, 0, 0],
            [0, 0.6, 0, -0.8],
            [0, 0, 0.6, 0],
            [0, 0, 0, 1],
        ])
        R0 = np.eye(4)
        R0[:3, :3] = Rot.from_euler('zyx', [0, 0, 0], degrees=True).as_matrix()
        T0 = T0 @ R0
        tensorf.compose(ficus, T0, R0[:3, :3])

        T0 = np.array([
            [0.6, 0, 0, -0.8],
            [0, 0.6, 0, 0],
            [0, 0, 0.6, 0],
            [0, 0, 0, 1],
        ])
        R0 = np.eye(4)
        R0[:3, :3] = Rot.from_euler('zyx', [90, 0, 0], degrees=True).as_matrix()
        T0 = T0 @ R0
        tensorf.compose(chair, T0, R0[:3, :3])


        T0 = np.array([
            [0.6, 0, 0, 0.8],
            [0, 0.6, 0, 0],
            [0, 0, 0.6, 0],
            [0, 0, 0, 1],
        ])
        R0 = np.eye(4)
        R0[:3, :3] = Rot.from_euler('zyx', [-90, 0, 0], degrees=True).as_matrix()
        T0 = T0 @ R0
        tensorf.compose(chair, T0, R0[:3, :3])

        T0 = np.array([
            [0.6, 0, 0, 0],
            [0, 0.6, 0, 0.8],
            [0, 0, 0.6, 0],
            [0, 0, 0, 1],
        ])
        R0 = np.eye(4)
        R0[:3, :3] = Rot.from_euler('zyx', [0, 0, 0], degrees=True).as_matrix()
        T0 = T0 @ R0
        tensorf.compose(chair, T0, R0[:3, :3])    


    ### Lego on Chair
    if False:
        
        tensorf2 = load_model('./log/lego_hybrid/lego_hybrid_5.th', 'CCNeRF')
        
        T0 = np.array([
            [0.5, 0, 0, 0],
            [0, 0.5, 0, 0],
            [0, 0, 0.5, 0.5],
            [0, 0, 0, 1],
        ])
        R0 = np.eye(4)
        R0[:3, :3] = Rot.from_euler('zyx', [0, 0, 45], degrees=True).as_matrix()
        T0 = T0 @ R0
        tensorf.compose(tensorf2, T0, R0[:3, :3])

    ### Truck in Barn
    if False:
        tensorf2 = load_model('./log/Truck_hybrid/Truck_hybrid_5.th', 'CCNeRF')
        T0 = np.array([
            [0.35, 0, 0, 0.8],
            [0, 0.35, 0, -0.05],
            [0, 0, 0.35, 0.8],
            [0, 0, 0, 1],
        ])
        R0 = np.eye(4)
        R0[:3, :3] = Rot.from_euler('xyz', [0, 0, 0], degrees=True).as_matrix()
        T0 = T0 @ R0
        tensorf.compose(tensorf2, T0, R0[:3, :3])

    
    ### Lego on Barn
    if False:
        
        tensorf2 = load_model('./log/lego_hybrid/lego_hybrid_1.th', 'CCNeRF')
        
        T0 = np.array([
            [0.35, 0, 0, -1.0],
            [0, 0.35, 0, -0.15],
            [0, 0, 0.35, -1.5],
            [0, 0, 0, 1],
        ])
        R0 = np.eye(4)
        R0[:3, :3] = Rot.from_euler('xyz', [-90, 180, 0], degrees=True).as_matrix()
        T0 = T0 @ R0

        tensorf.compose(tensorf2, T0, R0[:3, :3])

        tensorf2 = load_model('./log/lego_hybrid/lego_hybrid_5.th', 'CCNeRF')

        T0 = np.array([
            [0.18, 0, 0, -1.2],
            [0, 0.18, 0, 0.35],
            [0, 0, 0.18, -0.6],
            [0, 0, 0, 1],
        ])
        R0 = np.eye(4)
        R0[:3, :3] = Rot.from_euler('xyz', [-90, 270, 30], degrees=True).as_matrix()
        T0 = T0 @ R0

        tensorf.compose(tensorf2, T0, R0[:3, :3])

    ######################################


    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_compose', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation_test(train_dataset,tensorf, args, renderer, f'{logfolder}/imgs_train_compose/', N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'======> {args.expname} train all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/imgs_test_compose', exist_ok=True)
        evaluation_test(test_dataset,tensorf, args, renderer, f'{logfolder}/imgs_test_compose/', N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)

    if args.render_path:
        c2ws = test_dataset.render_path
        os.makedirs(f'{logfolder}/imgs_path_compose', exist_ok=True)
        evaluation_path(test_dataset,tensorf, c2ws, renderer, f'{logfolder}/imgs_path_compose/', N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)


if __name__ == '__main__':

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    args = config_parser()
    print(args)

    render_test(args)

