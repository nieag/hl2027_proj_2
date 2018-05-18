import SimpleITK as sitk
import matplotlib.pyplot as plt
from ipywidgets import interact, fixed
from IPython.display import clear_output


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

    plt.show()


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
    # plt.close()


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


def est_lin_transf(im_ref, im_mov):
    """
    Estimate linear transform to align `im_mov` to `im_ref` and
    return the transform parameters.

    """
    initial_transform = sitk.CenteredTransformInitializer(im_ref, im_mov, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    mov_resampled = sitk.Resample(im_mov, im_ref, initial_transform, sitk.sitkLinear, 0.0, im_mov.GetPixelID())

    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    # registration_method.SetMetricAsMeanSquares()
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # Connect all of the observers so that we can perform plotting during registration.
    registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
    registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
    registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations)
    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(registration_method))

    final_transform = registration_method.Execute(sitk.Cast(im_ref, sitk.sitkFloat32),
                                                  sitk.Cast(im_mov, sitk.sitkFloat32))
    return final_transform


def apply_lin_transf(im_ref, im_mov, lin_xfm):
    """
    Apply given linear transform `lin_xfm` to `im_mov` and return
    the transformed image.
    """

    mov_resampled = sitk.Resample(im_mov, im_ref, lin_xfm, sitk.sitkLinear, 0.0, im_mov.GetPixelID())

    return mov_resampled


if __name__ == '__main__':
    common_image = sitk.ReadImage("common_img_mask/common_40_image.nii", sitk.sitkFloat32)
    mov_path = "group_img/g5_65_image.nii"

    mov = sitk.ReadImage(mov_path, sitk.sitkFloat32)
    # display_images(10, 10, sitk.GetArrayViewFromImage(ref), sitk.GetArrayViewFromImage(mov))
    print(common_image)
    print(mov)
    lin_xfm = est_lin_transf(common_image, mov)
    mov_resampled = apply_lin_transf(common_image, mov, lin_xfm)
    # plt.figure()
    # display_images_with_alpha(100, 0.5, ref, mov_init_resamp)
    # mov_resampled = apply_lin_transf(ref, mov, lin_xfm)
    # plt.figure()
    # display_images_with_alpha(100, 0.5, ref, mov_resampled)
    # plt.show()
