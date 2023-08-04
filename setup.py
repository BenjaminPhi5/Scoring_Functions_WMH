from setuptools import find_packages, setup

setup(
	name='losswmh',
	version='0.1.0',
	packages=find_packages(exclude=["cluster_scripts"]),
	
	install_requires = [
	  	"numpy",
		"nibabel",
        "pyrobex",
		"SimpleITK",
		"itkwidgets",
		"natsort",
		"matplotlib",
        "seaborn",
		"torch",
		"torchvision",
		"monai",
		"torchio",
       	"deepspeed",
		"tqdm",
        "connected-components-3d",
        "scipy",
        "scikit-image",
        "scikit-learn",
        "kornia",
		"tensorboard",
		"pytorch-lightning",
        "torchinfo",
		"jupyterlab==3.6.1",
        "timm",
        "segmentation-models-pytorch",
        "fiftyone",
        "wandb",
	]
)
