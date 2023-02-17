from torch.utils.data import DataLoader
from mdm.data_loaders.tensors import collate as all_collate
from mdm.data_loaders.tensors import t2m_collate

def get_dataset_class(name):
    if "amass" in name or name in ['h36m','3dpw']:
        from VIBE.lib.dataset import vibe_dataset
        return vibe_dataset.VibeDataset
    elif name == "uestc":
        from .a2m.uestc import UESTC
        return UESTC
    elif name == "humanact12":
        from .a2m.humanact12poses import HumanAct12Poses
        return HumanAct12Poses
    elif name == "humanml":
        from data_loaders.humanml.data.dataset import HumanML3D
        return HumanML3D
    elif name == "kit":
        from data_loaders.humanml.data.dataset import KIT
        return KIT
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')

def get_collate_fn(name, hml_mode='train'):
    if hml_mode == 'gt':
        from data_loaders.humanml.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate
    if name in ["humanml", "kit"]:
        return t2m_collate
    else:
        return all_collate


def get_dataset(name, num_frames, split='train', hml_mode='train', no_motion_augmentation=False):
    DATA = get_dataset_class(name)
    if name in ["humanml", "kit"]:
        dataset = DATA(split=split, num_frames=num_frames, mode=hml_mode, no_motion_augmentation=no_motion_augmentation)
    elif 'amass' in name or name=='h36m':
        # these use the same Dataset class
        if ('amass' in name) and name!="amass":
            # Case where we take subsets of amass. Expect subdataset structure to be like "amass:KIT,CMU,HumanEva"
            restrict_subsets=name[6:].split(",")
            name='amass'
        else:
            restrict_subsets=None
        dataset = DATA(split=split, num_frames=num_frames, dataset=name, restrict_subsets=restrict_subsets)
    elif name in ('h36m','3dpw'):
         dataset = DATA(split=split, num_frames=num_frames, restrict_subsets=None, dataset=name)
    else:
        dataset = DATA(split=split, num_frames=num_frames)
    return dataset


def get_dataset_loader(name, batch_size, num_frames, split='train', hml_mode='train', 
    no_motion_augmentation=False, num_workers=0):
    dataset = get_dataset(name, num_frames, split, hml_mode, no_motion_augmentation=no_motion_augmentation)
    collate = get_collate_fn(name, hml_mode)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True, collate_fn=collate
    )

    return loader