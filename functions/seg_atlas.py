import SimpleITK as sitk
from utilities import *
from nl_registration import est_nl_transf, apply_nl_transf
from l_registration import est_lin_transf, apply_lin_transf
import matplotlib.pyplot as plt
import numpy as np

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

    for atlas_ct, atlas_seg in zip(atlas_ct_list, atlas_seg_list):
        # Estimate and apply linear transform
        lin_xfm = est_lin_transf(im, atlas_ct)
        # l_transforms.append(lin_xfm)
        atlas_ct_resampled = apply_lin_transf(im, atlas_ct, lin_xfm)

        # Estimate and apply non-linear transform
        nl_xfm = est_nl_transf(im, atlas_ct_resampled)
        # nl_transforms.append(nl_xfm)

        seg_reg = apply_lin_transf(im, atlas_seg, lin_xfm)
        seg_reg = apply_nl_transf(im, seg_reg, nl_xfm)
        registred_segmentations.append(seg_reg)

    majority_voted_segmentation = sitk.LabelVoting(registred_segmentations, label_for_undecided_pixels)

    return majority_voted_segmentation, registred_segmentations

def evaluate_segmentations(ground_truth, segmentation):
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
    overlap_measures_filter.Execute(ground_truth, segmentation)
    hausdorff_distance_filter.Execute(ground_truth, segmentation)
    return overlap_measures_filter.GetDiceCoefficient(), hausdorff_distance_filter.GetHausdorffDistance()

def binary_closing(mask):
    f = sitk.BinaryMorphologicalClosingImageFilter()
    f.SetKernelType(sitk.BinaryMorphologicalClosingImageFilter.Ball)
    f.SetKernelRadius(10)
    return f.Execute(mask)

if __name__ == '__main__':
    common_image = sitk.ReadImage("common_img_mask/common_42_image.nii", sitk.sitkFloat32)
    common_image = scale_volume(common_image, [256, 256, common_image.GetSize()[2]])
    atlas_seg_list_paths = ["group_img/g5_65_image_manual_seg.mhd",
                            "group_img/g5_66_image_manual_seg.mhd",
                            "group_img/g5_67_image_manual_seg.mhd"]
    atlas_ct_list_paths = ["group_img/g5_65_image.nii",
                           "group_img/g5_66_image.nii",
                           "group_img/g5_67_image.nii"]

    atlas_seg_list = [sitk.ReadImage(file_name, sitk.sitkFloat32) for file_name in atlas_seg_list_paths]
    atlas_seg_list = [atlas_seg < 10 for atlas_seg in atlas_seg_list]
    atlas_ct_list = [sitk.ReadImage(file_name, sitk.sitkFloat32) for file_name in atlas_ct_list_paths]

    atlas_seg_list = [scale_volume(seg, [256, 256, seg.GetSize()[2]]) for seg in atlas_seg_list]
    atlas_ct_list = [scale_volume(ct, [256, 256, ct.GetSize()[2]]) for ct in atlas_ct_list]

    majority_vote, registred_segmentations = seg_atlas(common_image, atlas_ct_list, atlas_seg_list)
    # sitk.WriteImage(majority_vote, "majority_vote_scaled_segmentation_img_42.nii")
    # for i, seg in enumerate(registred_segmentations):
    #     sitk.WriteImage(seg, "registred_seg{}_scaled_img_42.nii".format(i))

    ground_truth = sitk.ReadImage("common_img_mask/common_43_mask.nii", sitk.sitkUInt8)
    ground_truth = scale_volume(ground_truth, [256, 256, ground_truth.GetSize()[2]])
    ground_truth_temp1 = ground_truth == 3
    ground_truth_temp2 = ground_truth == 4
    ground_truth = ground_truth_temp1 + ground_truth_temp2
    # seg = sitk.ReadImage("majority_vote_scaled_segmentation_img_42.nii", sitk.sitkUInt8)
    seg_closed = binary_closing(majority_vote)

    sitk.WriteImage(seg_closed, "majority_vote_segmentation_img_42_closed.nii")
    #
    dice, hauss = evaluate_segmentations(ground_truth, seg)
    dice_closed, hauss_closed = evaluate_segmentations(ground_truth, seg_closed)
    print("Dice score un-closed: {}".format(dice))
    print("Haus score un-closed: {}".format(hauss))

    print("Dice score closed: {}".format(dice_closed))
    print("Haus score closed: {}".format(hauss_closed))
