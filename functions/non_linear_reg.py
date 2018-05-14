import SimpleITK as sitk
import matplotlib.pyplot as plt
import sys
from ipywidgets import interact, fixed
from scipy.ndimage import zoom


def demons_registration(fixed_image, moving_image, fixed_image_mask=None, fixed_points=None, moving_points=None):
    reg_method = sitk.ImageRegistrationMethod()

    transform_to_displacement_field_filter = sitk.TransformToDisplacementFieldFilter()
    transform_to_displacement_field_filter.SetReferenceImage(fixed_image)
    initial_transform = sitk.DisplacementFieldTransform(transform_to_displacement_field_filter.Execute(sitk.Transform()))
    initial_transform.SetSmoothingGaussianOnUpdate(varianceForUpdateField=1, varianceForTotalField=1)

    reg_method.SetInitialTransform(initial_transform)

    # Be aware that you will need to provide a parameter (the intensity difference threshold) as input:
    # during the registration, intensities are considered to be equal if their difference is less than the given threshold.
    reg_method.SetMetricAsDemons(10)

    reg_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    reg_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[8, 4, 0])

    reg_method.SetInterpolator(sitk.sitkLinear)
    # If you have time, run this code as is, otherwise switch to the gradient descent optimizer
    #reg_method.SetOptimizerAsConjugateGradientLineSearch(learningRate=1.0, numberOfIterations=20, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    reg_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=20, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    reg_method.SetOptimizerScalesFromPhysicalShift()

    # If corresponding points in the fixed and moving image are given then we display the similarity metric
    # and the TRE during the registration.
    if fixed_points and moving_points:
        reg_method.AddCommand(sitk.sitkStartEvent, rc.metric_and_reference_start_plot)
        reg_method.AddCommand(sitk.sitkEndEvent, rc.metric_and_reference_end_plot)
        reg_method.AddCommand(sitk.sitkIterationEvent, lambda: rc.metric_and_reference_plot_values(reg_method, fixed_points, moving_points))

    return reg_method.Execute(fixed_image, moving_image)


if __name__ == '__main__':
    im_no = 41  # 40-42...?
    ref_path = 'common_img_mask/common_' + str(im_no) + '_image.nii'
    masks_path = 'common_img_mask/common_' + str(im_no) + '_mask.nii'

    mov_path = 'group_img/g5_65_image.nii'
    # 300 slices 512x512
    ref = sitk.GetArrayFromImage(sitk.ReadImage(ref_path, sitk.sitkFloat32))
    mov = sitk.GetArrayFromImage(sitk.ReadImage(mov_path, sitk.sitkFloat32))

    print(ref.shape)

    x = ref.shape[0] / ref.shape[0]
    y = 50 / ref.shape[1]
    z = 50 / ref.shape[2]

    ref = zoom(ref, (x, y, z))

    print(ref.shape)

    ref = sitk.GetImageFromArray(ref)
    mov = sitk.GetImageFromArray(mov)

    transformation = demons_registration(fixed_image=ref,
                                     moving_image=mov)
