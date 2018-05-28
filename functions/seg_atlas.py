import SimpleITK as sitk
from utilities import *
import matplotlib.pyplot as plt
import numpy as np

def est_lin_transf(im_ref, im_mov):
    """
    Estimate linear transform to align `im_mov` to `im_ref` and
    return the transform parameters.
    """

    initial_transform = sitk.CenteredTransformInitializer(im_ref, im_mov, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=100) # Use mutual information as metric for linear reg
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.1) # Amount of samples to check metric at
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Apply the initial transform to line up the volumes
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # Plots of metric iterations
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

    # Resample the moving image with the estimated linear transform
    mov_resampled = sitk.Resample(im_mov, im_ref, lin_xfm, sitk.sitkLinear, 0.0, im_mov.GetPixelID())

    return mov_resampled

def est_nl_transf(im_ref, im_mov):
    """
    Estimate non-linear transform to align `im_mov` to `im_ref` and
    return the transform parameters.
    """
    registration_method = sitk.ImageRegistrationMethod()

    # Create a displacement field
    transform_to_displacment_field_filter = sitk.TransformToDisplacementFieldFilter()
    transform_to_displacment_field_filter.SetReferenceImage(im_ref)
    initial_transform = sitk.DisplacementFieldTransform(transform_to_displacment_field_filter.Execute(sitk.Transform()))

    # Regularization (update field - viscous, total field - elastic) for the non-linear transform
    initial_transform.SetSmoothingGaussianOnUpdate(varianceForUpdateField=0.0, varianceForTotalField=2.0)
    registration_method.SetInitialTransform(initial_transform)

    # Use demons as metric to allow for non linear transformation
    registration_method.SetMetricAsDemons(0.01) #intensities are equal if the difference is less than 0.01

    # Estimate transformation at multiple detail levels.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[8,4,0])

    registration_method.SetInterpolator(sitk.sitkLinear)
    # Use SGD optimizer for speed
    registration_method.SetOptimizerAsGradientDescent(learningRate=0.5, numberOfIterations=40, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Plots for metric iterations
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

def seg_atlas(im, atlas_ct_list, atlas_seg_list):
    """
    Apply atlas-based segmentation of `im` using the list of CT
    images in `atlas_ct_list` and the corresponding segmentation masks
    in `atlas_seg_list`. Return the resulting segmentation mask after
    majority voting.
    """

    nl_transforms = []
    l_transforms = []
    registred_segmentations = []
    label_for_undecided_pixels = 10
    i = 0
    for atlas_ct, atlas_seg in zip(atlas_ct_list, atlas_seg_list):
        print("Estimating transformations for atlas: {}".format(i))
        # Estimate and apply linear transform
        lin_xfm = est_lin_transf(im, atlas_ct)
        atlas_ct_resampled = apply_lin_transf(im, atlas_ct, lin_xfm)
        # Estimate non-linear transform
        nl_xfm = est_nl_transf(im, atlas_ct_resampled)

        print("Applying estimated transform for atlas: {}".format(i))
        # Apply estimated transforms to atlases
        seg_reg = apply_lin_transf(im, atlas_seg, lin_xfm)
        seg_reg = apply_nl_transf(im, seg_reg, nl_xfm)
        registred_segmentations.append(seg_reg)
        i += 1
    # Create a final segmentation through majority vote of different atlases
    majority_voted_segmentation = sitk.LabelVoting(registred_segmentations, label_for_undecided_pixels)

    return majority_voted_segmentation

def evaluate_segmentations(ground_truth, segmentation):
    """
    Segmentation evaluation according to dice score and Hausdorff distance between
    ground truth and segmentation.

    Returns the evaluation scores.
    """

    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
    overlap_measures_filter.Execute(ground_truth, segmentation)
    hausdorff_distance_filter.Execute(ground_truth, segmentation)
    return overlap_measures_filter.GetDiceCoefficient(), hausdorff_distance_filter.GetHausdorffDistance()

def binary_closing(mask, radius=10):
    """
    Binary closing of estimated segmentation to remove any unwanted holes in the
    masks.

    Returns the closed segmentation
    """
    f = sitk.BinaryMorphologicalClosingImageFilter()
    f.SetKernelType(sitk.BinaryMorphologicalClosingImageFilter.Ball)
    f.SetKernelRadius(radius)
    return f.Execute(mask)

if __name__ == '__main__':
    # Read data
    common_image = sitk.ReadImage("common_img_mask/common_40_image.nii", sitk.sitkFloat32)
    common_image = scale_volume(common_image, [128, 128, common_image.GetSize()[2]]) # scale if needed
    atlas_seg_list_paths = ["group_img/hip/g5_65_image_manual_seg.mhd",
                            "group_img/hip/g5_66_image_manual_seg.mhd",
                            "group_img/hip/g5_67_image_manual_seg.mhd"]
    atlas_ct_list_paths = ["group_img/g5_65_image.nii",
                           "group_img/g5_66_image.nii",
                           "group_img/g5_67_image.nii"]

    atlas_seg_list = [sitk.ReadImage(file_name, sitk.sitkFloat32) for file_name in atlas_seg_list_paths] # List of atlases
    atlas_seg_list = [atlas_seg < 10 for atlas_seg in atlas_seg_list] # Get binary masks
    atlas_ct_list = [sitk.ReadImage(file_name, sitk.sitkFloat32) for file_name in atlas_ct_list_paths] # List of CT volumes

    # Scale data if needed
    atlas_seg_list = [scale_volume(seg, [128, 128, seg.GetSize()[2]]) for seg in atlas_seg_list]
    atlas_ct_list = [scale_volume(ct, [128, 128, ct.GetSize()[2]]) for ct in atlas_ct_list]

    # Majority vote the segmentations
    majority_vote = seg_atlas(common_image, atlas_ct_list, atlas_seg_list)

    # Write final segmentation
    # sitk.WriteImage(majority_vote, "majority_vote_scaled_segmentation_img_40.nii")

    ground_truth = sitk.ReadImage("common_img_mask/common_40_mask.nii", sitk.sitkUInt8)
    # Scale if needed
    ground_truth = scale_volume(ground_truth, [128, 128, ground_truth.GetSize()[2]])

    # Get the correct parts of the ground truth
    ground_truth = extract_from_label(ground_truth, [3, 4])
    # seg = sitk.ReadImage("atlas_segmentations/femur/majority_vote_segmentation_img_43_femur.nii", sitk.sitkUInt8)
    seg = majority_vote

    # Close segmentation if needed to improve score
    seg_closed = binary_closing(seg)

    # Evaluate both the closed and un-closed segmentations
    dice, hauss = evaluate_segmentations(ground_truth, seg)
    dice_closed, hauss_closed = evaluate_segmentations(ground_truth, seg_closed)

    # Print scores
    print("Dice score un-closed: {}".format(dice))
    print("Haussdorff distance un-closed: {}".format(hauss))
    print("Dice score closed: {}".format(dice_closed))
    print("Hausdorff distance closed: {}".format(hauss_closed))
