import simpleITK as sitk

def est_nl_transf(im_ref, im_mov):
    """
    Estimate non-linear transform to align `im_mov` to `im_ref` and
    return the transform parameters.
    """
    pass

def apply_nl_transf(im_mov, nl_xfm):
    """
    Apply given non-linear transform `nl_xfm` to `im_mov` and return
    the transformed image.
    """
    pass
