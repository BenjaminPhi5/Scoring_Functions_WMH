"""
use kornia to get the hausdorff loss that is based on the paper https://arxiv.org/pdf/1904.10030.pdf
the kornia page is: https://kornia.readthedocs.io/en/latest/_modules/kornia/losses/hausdorff.html

They only implement a binary version, and note they have a 2D and a 3D version, so need to be careful of that.

TODO: implement a multiclass version?
"""