import sys
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt


def readLabelsAndFiles():

    vol1_path = 'group_img/g5_65_image.nii'
    vol2_path = 'group_img/g5_66_image.nii'
    vol3_path = 'group_img/g5_67_image.nii'

    test1_path = 'common_img_mask/common_40_image.nii.gz'
    test2_path = 'common_img_mask/common_41_image.nii.gz'
    test3_path = 'common_img_mask/common_42_image.nii.gz'

    vol1 = sitk.ReadImage(vol1_path, sitk.sitkFloat32)  # [:, :, 60:93]  # 60:93
    vol2 = sitk.ReadImage(vol2_path, sitk.sitkFloat32)  # [:, :, 64:102]  # 64:102
    vol3 = sitk.ReadImage(vol3_path, sitk.sitkFloat32)  # [:, :, 85:130]  # 85:130

    test1 = sitk.ReadImage(test1_path, sitk.sitkFloat32)
    test2 = sitk.ReadImage(test2_path, sitk.sitkFloat32)
    test3 = sitk.ReadImage(test3_path, sitk.sitkFloat32)

    labels1 = np.zeros(vol1.GetSize()[2])
    labels2 = np.zeros(vol2.GetSize()[2])
    labels3 = np.zeros(vol3.GetSize()[2])
    labels4 = np.zeros(vol1.GetSize()[2])
    labels5 = np.zeros(vol2.GetSize()[2])
    labels6 = np.zeros(vol3.GetSize()[2])

    labels1[60:93] = 1
    labels2[64:102] = 1
    labels3[85:130] = 1

    labels4[56:104] = 1
    labels4[66:115] = 1
    labels4[48:102] = 1

    vol1, vol2, vol3 = sitk.GetArrayFromImage(vol1), sitk.GetArrayFromImage(vol2), sitk.GetArrayFromImage(vol3)
    train = np.vstack((vol1, vol2, vol3))
    test1, test2, test3 = sitk.GetArrayFromImage(test1), sitk.GetArrayFromImage(test2), sitk.GetArrayFromImage(test3)
    test = np.vstack((test1, test2, test3))

    train_labels = np.hstack((labels1, labels2, labels3))
    test_labels = np.hstack((labels4, labels5, labels6))

    return train, train_labels, test, test_labels


if __name__ == '__main__':
    readLabelsAndFiles()
