import sys
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt


def readLabelsAndFiles():

    vol1_path = 'group_img/g5_65_image.nii'
    vol2_path = 'group_img/g5_66_image.nii'
    vol3_path = 'group_img/g5_67_image.nii'

    vol1 = sitk.ReadImage(vol1_path, sitk.sitkFloat32)  # [:, :, 60:93]  # 60:93
    vol2 = sitk.ReadImage(vol2_path, sitk.sitkFloat32)  # [:, :, 64:102]  # 64:102
    vol3 = sitk.ReadImage(vol3_path, sitk.sitkFloat32)  # [:, :, 85:130]  # 85:130

    labels1 = np.zeros(vol1.GetSize()[2])
    labels2 = np.zeros(vol2.GetSize()[2])
    labels3 = np.zeros(vol3.GetSize()[2])

    labels1[60:93] = 1
    labels2[64:102] = 1
    labels3[85:130] = 1

    vol1, vol2, vol3 = sitk.GetArrayFromImage(vol1), sitk.GetArrayFromImage(vol2), sitk.GetArrayFromImage(vol3)
    train = np.vstack((vol1, vol2, vol3))

    print(train.shape)
    labels = np.hstack((labels1, labels2, labels3))

    return train, labels


if __name__ == '__main__':
    readLabelsAndFiles()
