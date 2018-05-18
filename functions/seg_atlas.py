import SimpleITK as sitk
from utilities import *
from nl_registration import est_nl_transf, apply_nl_transf
from l_registration import est_lin_transf, apply_lin_transf
import matplotlib.pyplot as plt

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

    for i, atlas_ct in enumerate(atlas_ct_list):
        # Estimate and apply linear transform
        lin_xfm = est_lin_transf(im, atlas_ct)
        l_transforms.append(lin_xfm)
        atlas_ct_resampled = apply_lin_transf(im, atlas_ct, lin_xfm)

        # Estimate and apply non-linear transform
        nl_xfm = est_nl_transf(im, atlas_ct_resampled)
        nl_transforms.append(nl_xfm)

    for atlas_seg, nl_transform, l_transform in zip(atlas_seg_list, nl_transforms, l_transforms):
        # Apply non-linear transforms to segmentations
        seg_reg = apply_lin_transf(im, atlas_seg, l_transform)
        seg_reg = apply_nl_transf(im, seg_reg, nl_transform)
        registred_segmentations.append(seg_reg)

    majority_voted_segmentation = sitk.LabelVoting(registred_segmentations, label_for_undecided_pixels)

    return majority_voted_segmentation, registred_segmentations

def evaluate_segmentations(segmentation, ground_truth):
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()

    dice_coeff = overlap_measures_filter.Execute(ground_truth, segmentation)
    hausdorff_score = hausdorff_distance_filter.Execute(ground_truth, segmentation)

    return dice_coeff, hausdorff_score

if __name__ == '__main__':
    common_image = sitk.ReadImage("common_img_mask/common_40_image.nii", sitk.sitkFloat32)
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
    sitk.WriteImage(majority_vote, "majority_vote_segmentation.nii")
    for i, seg in enumerate(registred_segmentations):
        sitk.WriteImage(seg, "registred_seg{}.nii".format(i))
