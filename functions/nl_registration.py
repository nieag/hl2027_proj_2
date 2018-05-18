import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
from ipywidgets import interact, fixed
from IPython.display import clear_output
from scipy.ndimage import zoom
import numpy as np
from l_registration import est_lin_transf, apply_lin_transf

def display_images(im_ref_z, im_mov_z, fixed_npa, moving_npa):
    # Create a figure with two subplots and the specified size.
    plt.subplots(1, 2, figsize=(10, 8))

    # Draw the fixed image in the first subplot.
    plt.subplot(1, 2, 1)
    plt.imshow(fixed_npa[im_ref_z, :, :], cmap=plt.cm.Greys_r)
    plt.title('fixed image')
    plt.axis('off')

    # Draw the moving image in the second subplot.
    plt.subplot(1, 2, 2)
    plt.imshow(moving_npa[im_mov_z, :, :], cmap=plt.cm.Greys_r)
    plt.title('moving image')
    plt.axis('off')

# Callback invoked by the IPython interact method for scrolling and modifying the alpha blending
# of an image stack of two images that occupy the same physical space.
def display_images_with_alpha(image_z, alpha, fixed, moving):
    img = (1.0 - alpha) * fixed[:, :, image_z] + alpha * moving[:, :, image_z]
    plt.imshow(sitk.GetArrayViewFromImage(img), cmap=plt.cm.Greys_r)
    plt.axis('off')
    # plt.show()

# Callback invoked when the StartEvent happens, sets up our new data.
def start_plot():
    global metric_values, multires_iterations

    metric_values = []
    multires_iterations = []


# Callback invoked when the EndEvent happens, do cleanup of data and figure.
def end_plot():
    global metric_values, multires_iterations

    del metric_values
    del multires_iterations
    # Close figure, we don't want to get a duplicate of the plot latter on.
    plt.close()


# Callback invoked when the IterationEvent happens, update our data and display new figure.
def plot_values(registration_method):
    global metric_values, multires_iterations

    metric_values.append(registration_method.GetMetricValue())
    # Clear the output area (wait=True, to reduce flickering), and plot current data
    clear_output(wait=True)
    # Plot the similarity metric values
    plt.plot(metric_values, 'r')
    plt.plot(multires_iterations, [metric_values[index] for index in multires_iterations], 'b*')
    plt.xlabel('Iteration Number',fontsize=12)
    plt.ylabel('Metric Value',fontsize=12)
    # plt.show()
    plt.pause(0.05)


# Callback invoked when the sitkMultiResolutionIterationEvent happens, update the index into the
# metric_values list.
def update_multires_iterations():
    global metric_values, multires_iterations
    multires_iterations.append(len(metric_values))


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
    common_path = 'common_img_mask/common_40_image.nii'
    g5_65_path = 'group_img/g5_65_image.nii'
    # g5_66_path = 'group_img/g5_66_image.nii'
    # g5_67_path = 'group_img/g5_67_image.nii'
    # seg_path = 'group_img/g5_65_image_manual_seg.mhd'
    g5_65_image = sitk.ReadImage(g5_65_path, sitk.sitkFloat32)
    # g5_66_image = sitk.ReadImage(g5_66_path, sitk.sitkFloat32)
    # g5_67_image = sitk.ReadImage(g5_67_path, sitk.sitkFloat32)
    ref = sitk.ReadImage(common_path, sitk.sitkFloat32)
    # seg = sitk.ReadImage(seg_path, sitk.sitkFloat32)
    # display_images(100, 100, sitk.GetArrayViewFromImage(seg), sitk.GetArrayViewFromImage(seg))
    # print(ref.GetSize())
    # print(mov.GetSize())
    ref = scale_volume(ref, [250, 250, 100])
    g5_65_image = scale_volume(g5_65_image, [250, 250, 100])

    lin_xfm = est_lin_transf(ref, g5_65_image)
    # seg_resampled = apply_lin_transf(ref, seg, lin_xfm)
    g5_65_image_resampled = apply_lin_transf(ref, g5_65_image, lin_xfm)
    print(ref.GetOrigin())
    print(g5_65_image_resampled.GetOrigin())
    # nl_xfm_with_lin = est_nl_transf(ref, g5_65_image_resampled)
    # g5_65_image_resampled = apply_nl_transf(ref, g5_65_image_resampled, nl_xfm_with_lin)
    # seg_resampled = apply_nl_transf(ref, seg_resampled, nl_xfm_with_lin)
    #
    # display_images(75, 75, sitk.GetArrayViewFromImage(seg), sitk.GetArrayViewFromImage(seg_resampled))
    # display_images(75, 75, sitk.GetArrayViewFromImage(ref), sitk.GetArrayViewFromImage(g5_65_image_resampled))
    # plt.figure()
    # display_images_with_alpha(75, 0.5, ref, g5_65_image_resampled)
    # plt.show()
