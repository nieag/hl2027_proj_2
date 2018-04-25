import simpelITK as sitk


def est_lin_transf(im_ref, im_mov):
    """
    Estimate linear transform to align `im_mov` to `im_ref` and
    return the transform parameters.
    """
    regMethod = sitk.ImageRegistrationMethod()
    regMethod.SetMetricAsMeanSquares()
    regMethod.SetOptimizerAsRegularStepGradientDescent(4.0, .01, 200 )
    regMethod.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))
    regMethod.SetInterpolator(sitk.sitkLinear)

    regMethod.AddCommand(sitk.sitkIterationEvent, lambda:command_iteration(regMethod))
    outTrans = regMethod.Execute(im_ref, im_mov)

    return outTrans

def apply_lin_transf(im_mov, lin_xfm):
    """
    Apply given linear transform `lin_xfm` to `im_mov` and return
    the transformed image.
    """
    pass
