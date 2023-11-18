import os
import argparse
import pydicom
import matplotlib.pyplot as plt
from monai import transforms
import numpy as np
import nibabel as nib

class ExpandDimsd:
    def __call__(self, img):
        array = img['image']
        img_1 = np.moveaxis(array, -1, 0)
        img_2 = np.expand_dims(img_1, axis=0)

        img['image'] = img_2
        return img
    
def exibir_img_dicom(path_dcm):
    arquivo_dicom = pydicom.dcmread(path_dcm)
    imagem_dicom = arquivo_dicom.pixel_array
    
    plt.imshow(imagem_dicom, cmap=plt.cm.gray)
    plt.title('Imagem DICOM')
    plt.show()
    
def exibir_imagem_nii(path_nii):
    file = nib.load(path_nii)
    imagem_nii= file.get_fdata()

    plt.imshow(imagem_nii[:,:,0], cmap='gray')
    plt.title('Imagem nii')
    plt.show()

def save_img_nii(path_dcm, output_dir='/project/outputs/reference_image'):
    
    transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            ExpandDimsd(),
            transforms.EnsureChannelFirstd(keys=["image"]),
            transforms.Resized(keys=["image"], spatial_size=(1,256, 256))
        ]
    )

    test_data = {"image": path_dcm}
    result = transform(test_data)

    img = result["image"]
    os.makedirs(output_dir, exist_ok=True)
    
    saver = transforms.SaveImage(
        output_dir=output_dir,
        output_ext=".nii.gz",
        output_dtype=np.float32,
        resample=False,
        squeeze_end_dims=True,
        writer="NibabelWriter",
    )
    img = saver(img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_img", help="Path to directory where the dcm file are")

    args = parser.parse_args()
    
    exibir_imagem_nii(args.input_img)
    save_img_nii(args.input_img)