"""
Based on mdm/sample/generate.py

Generate samples conditioned on video features. This means running GHMR inference. 

python -m mdm.sample.generate --model_path /path/to/model

Required args: 
    --model_path diffusion model to sample from. 
Optional args:     
    --condition_dataset one of 'h36m','3dpw'. If empty, defaults to the dataset the model was trained on. Note that args.dataset will be the same as model dataset ... this param is used for some other config (see the code)
    --condition_split. One of 'train','test'. Defaults to 'test'.  
    --output_dir. Where to put result images. if not specified, put in same folder as --model_path
    --num_samples default 10
    --num_repetitions default 3
"""
from mdm.utils.fixseed import fixseed
import os
import numpy as np
import torch
from mdm.utils.parser_util import generate_args
from mdm.utils.model_util import create_model_and_diffusion, load_model_wo_clip
from mdm.utils import dist_util
from mdm.model.cfg_sampler import ClassifierFreeSampleModel
from mdm.data_loaders.get_data import get_dataset_loader
from mdm.data_loaders.humanml.scripts.motion_process import recover_from_ric
import mdm.data_loaders.humanml.utils.paramUtil as paramUtil
from mdm.data_loaders.humanml.utils.plot_script import plot_3d_motion
import shutil
from mdm.data_loaders.tensors import collate
from gthmr.lib.utils.mdm_utils import viz_motions
import ipdb

def main():
    print("Generating samples")
    args = generate_args()
    print(f"Dataset split for conditioning variables, [{args.condition_split}]")
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    max_frames = 196 if args.dataset in ['kit', 'humanml'] else 60
    fps = 12.5 if args.dataset == 'kit' else 20
    n_frames = min(max_frames, int(args.motion_length*fps))
    is_using_data = not any([args.input_text, args.text_prompt, args.action_file, args.action_name])
    dist_util.setup_dist(args.device)

    if args.output_dir=="":
        args.output_dir = os.path.dirname(args.model_path)

    DO_AUTO_PATH_CREATE=True
    if DO_AUTO_PATH_CREATE:
        out_path = os.path.join(args.output_dir,
                                'samples_split_{}_{}_{}_seed{}'.format(args.condition_split, name, niter, args.seed))

        if args.text_prompt != '':
            out_path += '_' + args.text_prompt.replace(' ', '_').replace('.', '')
        elif args.input_text != '':
            out_path += '_' + os.path.basename(args.input_text).replace('.txt', '').replace(' ', '_').replace('.', '')
        os.makedirs(out_path, exist_ok=True)


    print(f"Out path: [{out_path}]")

    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples

    print(f'Loading dataset for conditioning features {args.condition_dataset}...')
    data = load_dataset(args, max_frames, n_frames, split=args.condition_split )

    total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    # model.cond_mode="no_cond"
    if args.guidance_param != 1 and model.cond_mode!="no_cond": # only do cfgsampler for conditional models
        model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    all_motions = []
    all_lengths = []
    all_text = []

    # first show the motion for the sampled condition
    def postprocess(sample, model, data,):

        # Recover XYZ *positions* from HumanML3D vector representation
        if model.data_rep == 'hml_vec':
            n_joints = 22 if sample.shape[1] == 263 else 21
            sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
            sample = recover_from_ric(sample, n_joints)
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

        rot2xyz_pose_rep = 'xyz' if model.data_rep in ['xyz', 'hml_vec'] else model.data_rep
        rot2xyz_mask = None if rot2xyz_pose_rep == 'xyz' else model_kwargs['y']['mask'].reshape(args.batch_size, n_frames).bool()
        sample = model.rot2xyz(x=sample, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                               jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
                               get_rotations_back=False)
        return sample

    # get data: 
    #   the `samples_vid_gt` are the ground truth motions 
    #   the `model_kwargs` have the video features used for conditioning. 
    samples_vid_gt, model_kwargs = next(iter(data))
    # get the ground trouth motion for the first column: we want the output samples to match this motion
    # 
    samples_vid_gt_processed = postprocess(samples_vid_gt.to(dist_util.dev()), model, data,)
    all_motions.append(samples_vid_gt_processed.cpu().numpy())
    all_text += [f'GT cond motion {args.condition_dataset }']*len(samples_vid_gt_processed)
    all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())

    for rep_i in range(args.num_repetitions):
        print(f'### Sampling [repetitions #{rep_i}]')

        # add CFG scale to batch

        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

        sample_fn = diffusion.p_sample_loop

        with torch.no_grad():
            NOISE_THE_INPUTS=False
            if NOISE_THE_INPUTS:
                print("Setting features to zero")
                model_kwargs['y']['features'] = torch.zeros(*model_kwargs['y']['features'].shape)
            sample = sample_fn(
                model,
                (args.batch_size, model.njoints, model.nfeats, n_frames),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,  
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )
        
        sample = postprocess(sample, model, data)
        if args.unconstrained:
            all_text += ['unconstrained'] * args.num_samples
        else:
            all_text += [f'conditioned sample {rep_i}']* args.num_samples

        all_motions.append(sample.cpu().numpy())
        all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())

        print(f"created {len(all_motions) * args.batch_size} samples")

    nrows = len(all_motions[0])
    ncols = args.num_repetitions+1 # bc we added the ground truth
    all_motions = np.vstack(all_motions)

    viz_motions(nrows, ncols, out_path, all_motions, dataset=data.dataset.dataname, all_text=all_text)
    # viz_motions(nrows, ncols, out_path, all_motions, dataset=data.dataset.dataname, all_text=None)
    print(f'[Done] Results are at [{os.path.abspath(out_path)}]')

def load_dataset(args, max_frames, n_frames, split):
    data = get_dataset_loader(name=args.condition_dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split=split,
                              hml_mode='text_only')
    data.fixed_length = n_frames
    return data


if __name__ == "__main__":
    main()
