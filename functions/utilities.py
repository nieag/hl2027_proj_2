import matplotlib.pyplot as plt
from ipywidgets import interact, fixed
from IPython.display import clear_output
import SimpleITK as sitk

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

def get_binary_mask(mask, background_intensity=10):
    """
    Creates a binary mask from continous values.

    Returns the binary mask.
    """
    mask = sitk.GetArrayFromImage(mask)
    mask = mask != background_intensity

    mask = sitk.GetImageFromArray(mask.astype(float))

    return sitk.Cast(mask, sitk.sitkFloat32)

def scale_volume(vol, new_size):
    """
    Re-scale given volume to "new_size".

    Returns the re-scaled volume.
    """
    new_spacing = [old_sz*old_spc/new_sz  for old_sz, old_spc, new_sz in zip(vol.GetSize(), vol.GetSpacing(), new_size)]

    interpolator_type = sitk.sitkLinear

    _new_vol = sitk.Resample(vol, new_size, sitk.Transform(), interpolator_type, vol.GetOrigin(), new_spacing, vol.GetDirection(), 0.0, vol.GetPixelIDValue())
    return _new_vol

def extract_from_label(mask, labels):
    """
    Extracts part of a labelled mask.

    Returns the mask according to requested labels.
    """
    extracted = 0

    for label in labels:
        extracted += mask == label

    return extracted
