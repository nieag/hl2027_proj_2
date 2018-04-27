import SimpleITK as sitk
import matplotlib.pyplot as plt

def command_iteration(method) :
    print("{0:3} = {1:10.5f} : {2}".format(method.GetOptimizerIteration(),
                                           method.GetMetricValue(),
                                           method.GetOptimizerPosition()))

def est_lin_transf(im_ref, im_mov):
    """
    Estimate linear transform to align `im_mov` to `im_ref` and
    return the transform parameter    s.
    """
    # initial_transform = sitk.CenteredTransformInitializer(im_ref, im_mov, sitk.Similarity3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    regMethod = sitk.ImageRegistrationMethod()
    regMethod.SetMetricAsMeanSquares()
    regMethod.SetOptimizerAsGradientDescent(4.0, 100)

    initial_transform = sitk.CenteredTransformInitializer(im_ref,
                                                      im_mov,
                                                      sitk.AffineTransform(3),
                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY)
    regMethod.SetInitialTransform(initial_transform)
    regMethod.SetInterpolator(sitk.sitkLinear)

    regMethod.AddCommand(sitk.sitkIterationEvent, lambda:command_iteration(regMethod))
    lin_xfm = regMethod.Execute(im_ref, im_mov)

    return lin_xfm

def apply_lin_transf(im_ref, im_mov, lin_xfm):
    """
    Apply given linear transform `lin_xfm` to `im_mov` and return
    the transformed image.
    """

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(im_ref);
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(lin_xfm)
    out = resampler.Execute(im_mov)
    return out


if __name__ == '__main__':
    ref_path = '/home/niels/Documents/hl2027_proj_2/common_img_mask/common_41_image.nii'
    mov_path = '/home/niels/Documents/hl2027_proj_2/group_img/g5_65_image.nii'
    ref = sitk.ReadImage(ref_path, sitk.sitkFloat32)
    mov = sitk.ReadImage(mov_path, sitk.sitkFloat32)
    ref_array = sitk.GetArrayFromImage(ref)
    mov_array = sitk.GetArrayFromImage(mov)

    transform = est_lin_transf(ref, mov)

    final = apply_lin_transf(ref, mov, transform)
    final_array = sitk.GetArrayFromImage(final)
    plt.figure()
    plt.imshow(final_array[100, :, :], cmap="gray")
    plt.show()
