import pandas as pd
#import os
#from generative.metrics import MultiScaleSSIMMetric
#import tifffile
#import torch

if __name__ == '__main__':
    
    df = pd.read_csv('./table_msssim.csv')
    print(df.head())

    columns_to_compare = [
        'KVP', 'Exposure', 'Revolution Time', 'Pitch', 'FS_Res', 
        'Filter Type', 'Convolution Kernel', 'Slice Thickness', 
        'Pixel Spacing', 'Reconstruction Diameter', 'ReconTypeInt', 
        'MonoEkeV', 'CTDIvol', 'Image Type', 'Collimation', 
        'iDoseLevel', 'Slice Location'
    ]
    #columns_to_compare = ['ReconTypeInt','MonoEkeV']
    df = df.fillna({'MonoEkeV':'Faltante'})
    df_high = df[(df['MS-SSIM'] > 0.8) & (df['MS-SSIM'] < 0.9)]
    #df_low = df[df['MS-SSIM'] < 0.8]
    
    #df_100 = df_high[df_high['Folder'] == 100]
    #Wdf_101 = df_low[df_low['Folder'] == 101]
    #print(df_100.iloc[0])
    #print(df_101.iloc[0])

    print(df_high)
    #print(df_low[columns_to_compare].value_counts())
    exit()
    
    mssim_metric = MultiScaleSSIMMetric(spatial_dims=2, kernel_size=7)

    table = pd.read_csv('/project/i2_plaque5mgml_20240717.csv')
    path_imgs = r'/project/outputs/recons_imgs_valid_toy_2'
    
    table['Folder'] = table['Root'].apply(lambda x: x.split('.\\')[-1][1:])
    new_table = pd.DataFrame(columns=table.columns.tolist() + ['image_name','MS-SSIM'])
    files = os.listdir(path_imgs)

    for i in range(0, len(files) - 1, 2):
        image_name = files[i]
        image_recons_name = files[i+1]
        
        folder = image_name.split('.')[3]
        filter_table = table[table['Folder'] == folder].values

        path_img = os.path.join(path_imgs, image_name)
        path_img_recons = os.path.join(path_imgs, image_recons_name)

        image = tifffile.imread(path_img)
        image = torch.tensor(image).unsqueeze(0).unsqueeze(0)
        reconstruction = tifffile.imread(path_img_recons)
        reconstruction = torch.tensor(reconstruction).unsqueeze(0).unsqueeze(0)

        mssim_value = mssim_metric(image, reconstruction)[0,0].item()

        new_table.loc[len(new_table)] = filter_table[0].tolist() + [image_name, mssim_value]


    new_table.to_csv('/project/table_msssim.csv', index=False)
          

        




