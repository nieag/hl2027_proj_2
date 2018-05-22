import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
from ipywidgets import interact, fixed
from IPython.display import clear_output
from scipy.ndimage import zoom
import numpy as np
from l_registration import est_lin_transf, apply_lin_transf
from utilities import *

def est_nl_transf(im_ref, im_mov):
    registration_method = sitk.ImageRegistrationMethod()

    # Create initial identity transformation.
    transform_to_displacment_field_filter = sitk.TransformToDisplacementFieldFilter()
    transform_to_displacment_field_filter.SetReferenceImage(im_ref)
    # The image returned from the initial_transform_filter is transferred to the transform and cleared out.
    initial_transform = sitk.DisplacementFieldTransform(transform_to_displacment_field_filter.Execute(sitk.Transform()))

    # Regularization (update field - viscous, total field - elastic).
    initial_transform.SetSmoothingGaussianOnUpdate(varianceForUpdateField=0.0, varianceForTotalField=2.0)


    registration_method.SetInitialTransform(initial_transform)

    registration_method.SetMetricAsDemons(10) #intensities are equal if the difference is less than 10HU

    # Multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[8,4,0])

    registration_method.SetInterpolator(sitk.sitkLinear)
    # If you have time, run this code as is, otherwise switch to the gradient descent optimizer
    # registration_method.SetOptimizerAsConjugateGradientLineSearch(learningRate=1.0, numberOfIterations=20, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=20, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
    registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
    registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations)
    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(registration_method))

    final_transform = registration_method.Execute(sitk.Cast(im_ref, sitk.sitkFloat32),
                                                  sitk.Cast(im_mov, sitk.sitkFloat32))

    return final_transform

def apply_nl_transf(im_ref, im_mov, nl_xfm):
    """
    Apply given non-linear transform `nl_xfm` to `im_mov` and return
    the transformed image.
    """

    mov_resampled = sitk.Resample(im_mov, im_ref, nl_xfm, sitk.sitkLinear, 0.0, im_mov.GetPixelID())
    return mov_resampled

def scale_volume(vol, new_size):
    new_spacing = [old_sz*old_spc/new_sz  for old_sz, old_spc, new_sz in zip(vol.GetSize(), vol.GetSpacing(), new_size)]

    interpolator_type = sitk.sitkLinear

    _new_vol = sitk.Resample(vol, new_size, sitk.Transform(), interpolator_type, vol.GetOrigin(), new_spacing, vol.GetDirection(), 0.0, vol.GetPixelIDValue())
    return _new_vol

if __name__ == '__main__':
    common_image = sitk.ReadImage("common_img_mask/common_40_image.nii", sitk.sitkFloat32)
    mov = sitk.ReadImage("group_img/g5_65_image.nii", sitk.sitkFloat32)
    seg = sitk.ReadImage("group_img/g5_65_image_manual_seg.mhd", sitk.sitkFloat32)

    lin_xfm = est_lin_transf(common_image, mov)
    mov_resampled = apply_lin_transf(common_image, mov, lin_xfm)
    nl_xfm = demons_test(common_image, mov_resampled)
