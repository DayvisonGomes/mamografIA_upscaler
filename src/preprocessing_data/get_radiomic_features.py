import os
import pandas as pd
import numpy as np
import SimpleITK as sitk
import radiomics
from radiomics import featureextractor
import six
import time
import json

def change_path(path):
    new_path = r'\\rad-maid-004\D' + '\\' + '\\'.join(path.split('/')[1:])
    return new_path

def create_mask(image_array, roi_coords):
    mask_array = np.zeros_like(image_array, dtype=np.uint8)
    x1, y1, x2, y2 = roi_coords
    
    if x1 < 0 or y1 < 0 or x2 > image_array.shape[1] or y2 > image_array.shape[0]:
        print("ROI coordinates are out of image bounds")
        return 1
    
    mask_array[y1:y2, x1:x2] = 1
    return mask_array

def get_numeric_value(value):
    if isinstance(value, (np.ndarray, np.generic)):
        return value.item()
    return value

def extract_features(image_array, mask_array, extractor):
    image = sitk.GetImageFromArray(image_array)
    mask = sitk.GetImageFromArray(mask_array)
    mask.CopyInformation(image)

    result = extractor.execute(image, mask)
    features = {key: value for key, value in six.iteritems(result)}
    
    return features

def calculate_average_features(df, extractor):
    birads_classes = df['BIRADS'].unique()
    features_dict = {birads: [] for birads in birads_classes}
    total = df.shape[0]
    
    for _, row in df.iterrows():
        image_path = row['Descriptive_Path']
        roi_coords = eval(row['ROI_separated'])[0]
        print('Coordenadas: ', roi_coords)
        
        image = sitk.ReadImage(image_path)
        image_array = sitk.GetArrayFromImage(image)
        
        print('Shape imagem: ', image_array.shape)
        mask_array = create_mask(image_array[0], roi_coords)
        
        if type(mask_array) == int:
            continue
        
        print('Valores únicos máscara: ', np.unique(mask_array))
        tic = time.time()
        features = extract_features(image_array[0], mask_array, extractor)
        toc = time.time()
        features_dict[row['BIRADS']].append(features)

        print('\nTime (s): ', toc - tic)
        print('')
        print(f'Row: {_+1} / {total}')
        
    average_features_dict = {}
    for birads, features_list in features_dict.items():
        if features_list:
            all_keys = features_list[0].keys()
            average_features = {}
            for key in all_keys:
                if type(features_list[0][key]) == str:
                    average_features[key] = features_list[0][key]
                    continue
                    
                values = [get_numeric_value(features[key]) for features in features_list if isinstance(get_numeric_value(features[key]), (int, float))]
                
                if values:
                    average_features[key] = np.mean(values)
                    
            average_features_dict[birads] = average_features

    return average_features_dict

if __name__ == '__main__':
    path = r'\\rad-maid-004\D\Users\taylor\EMBED\data_processed\EMBED_OpenData_updated.csv'
    df = pd.read_csv(path)
    df['Descriptive_Path'] = df['Descriptive_Path'].apply(change_path)
    extractor = featureextractor.RadiomicsFeatureExtractor()

    average_features = calculate_average_features(df, extractor)
    with open('D:/Users/dayv/mamografIA_upscaler/average_features.json', 'w') as json_file:
        json.dump(average_features, json_file, indent=4)
        
    #for birads, features in average_features.items():
        #print(f"BIRADS {birads} - Average Features:")
        #for i, (feature, value) in enumerate(features.items()):
            #if i >= 20:
                #break
            #print(f"\t{feature}: {value}")
