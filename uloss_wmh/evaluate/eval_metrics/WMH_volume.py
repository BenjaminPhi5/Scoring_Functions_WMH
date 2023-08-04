from torch import Tensor

def GT_volumes(ys3d, voxel_volume=0.0003):
    volumes = []
    for y in ys3d:
        volumes.append(y.sum() * voxel_volume)
    return Tensor(volumes)


def get_volumes(dss, voxel_volume):
    """
    dss is a list of datasets
    """
    ys3d = []
    for ds in test_dss:
        for data in ds:
            label = data['label']
            ys3d.append(label)
            
    return GT_volumes(ys3d, voxel_volume=0.0003)