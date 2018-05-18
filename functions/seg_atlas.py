import SimpleITK as sitk
from nl_registration import est_nl_transf, apply_nl_transf, display_images, display_images_with_alpha
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
    # label_for_undecided_pixels = 10

    for i, atlas_ct in enumerate(atlas_ct_list):
        print(i)
        # Estimate and apply linear transform
        lin_xfm = est_lin_transf(im, atlas_ct)
        atlas_ct_resampled = apply_lin_transf(im, atlas_ct, lin_xfm)

        # Estimate and apply non-linear transform
        nl_xfm = est_nl_transf(im, atlas_ct_resampled)
        atlas_ct_resampled = apply_nl_transf(im, atlas_ct_resampled, nl_xfm)
        display_images_with_alpha(100, 1, atlas_ct_resampled, atlas_ct_resampled)
        nl_transforms.append(nl_xfm)
        l_transforms.append(lin_xfm)

    for atlas_seg, nl_transform, l_transform in zip(atlas_seg_list, nl_transforms, l_transforms):
        # Apply non-linear transforms to segmentations
        seg_reg = apply_lin_transf(im, atlas_seg, l_transform)
        seg_reg = apply_nl_transf(im, seg_reg, nl_transform)

        plt.show()
        registred_segmentations.append(sitk.Cast(seg_reg, sitk.sitkUInt8))

    # reference_segmentation_majority_vote = sitk.LabelVoting(registred_segmentations, label_for_undecided_pixels)

    return registred_segmentations

def scale_volume(vol, new_size):
    new_spacing = [old_sz*old_spc/new_sz  for old_sz, old_spc, new_sz in zip(vol.GetSize(), vol.GetSpacing(), new_size)]

    interpolator_type = sitk.sitkLinear

    _new_vol = sitk.Resample(vol, new_size, sitk.Transform(), interpolator_type, vol.GetOrigin(), new_spacing, vol.GetDirection(), 0.0, vol.GetPixelIDValue())
    return _new_vol

def get_binary_mask(mask):
    mask = sitk.GetArrayViewFromImage(mask)
    mask = mask != 10

    mask = sitk.GetImageFromArray(mask.astype(int))

    return mask

if __name__ == '__main__':
    common_image = scale_volume(sitk.ReadImage("common_img_mask/common_40_image.nii", sitk.sitkFloat32), [128, 128, 209])
    atlas_seg_list_paths = ["group_img/g5_65_image_manual_seg.mhd"]#,
                              # "group_img/g5_66_image_manual_seg.mhd",
                              # "group_img/g5_67_image_manual_seg.mhd"]
    atlas_ct_list_paths = ["group_img/g5_65_image.nii"]#,
                              # "group_img/g5_66_image.nii",
                              # "group_img/g5_67_image.nii"]

    atlas_seg_list = [scale_volume(sitk.ReadImage(file_name, sitk.sitkFloat32), [128, 128, 209]) for file_name in atlas_seg_list_paths]
    atlas_seg_list = [get_binary_mask(atlas_seg) for atlas_seg in atlas_seg_list]
    atlas_ct_list = [scale_volume(sitk.ReadImage(file_name, sitk.sitkFloat32), [128, 128, 209]) for file_name in atlas_ct_list_paths]

    # print(atlas_ct_list[0])
    # display_images_with_alpha(100, 100, common_image, atlas_ct_list[0])
    # display_images(100, 100, sitk.GetArrayViewFromImage(atlas_seg_list[0]), sitk.GetArrayViewFromImage(atlas_seg_list[0]))
    # plt.show()
    # print(common_image.GetOrigin())
    #
    # for atlas_seg in atlas_seg_list:
    #     print(atlas_seg.GetOrigin())
    #
    # for atlas_ct in atlas_ct_list:
    #     print(atlas_ct.GetOrigin())
    #
    registred_segmentations = seg_atlas(common_image, atlas_ct_list, atlas_seg_list)
    #
    #
    #
    # for i, seg in enumerate(registred_segmentations):
    #     sitk.WriteImage(seg,'seg{}.nii'.format(i))
