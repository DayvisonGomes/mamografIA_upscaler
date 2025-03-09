import os
import pandas as pd
from sklearn.model_selection import train_test_split

def print_directory_contents(directory):
    paths = []
    for root, dirs, files in os.walk(directory):
        for subdir in dirs:
            subdir_path = os.path.join(root, subdir)
            for file in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, file)
                paths.append(file_path)
                
    return paths

base_path = r'D:\Users\bbruno\Data\ct_noel'

cases = os.listdir(base_path)
all_data = []

for case in cases:
    case_path = os.path.join(base_path, case)
    
    if os.path.isdir(case_path):
        print(f'Caso: {case}')
        
        paths = print_directory_contents(case_path)
        paths = paths[9:]  
        metade = len(paths) // 2
        paths = paths[metade - 500: metade + 500]
        print(len(paths))
        print(paths[0])
        for path in paths:
            path = '/'.join(path.split('\\'))
            all_data.append({"image": path.replace('D:/Users/bbruno/Data/ct_noel', '/project/ct_noel'), 'classe':case})

df = pd.DataFrame(all_data)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
#df_balanced = df.groupby('classe').head(14000 // len(df['classe'].unique()))

train_data_list, test_val_data = train_test_split(df, test_size=0.3, random_state=42, stratify=df['classe'])
val_data_list, test_data_list = train_test_split(test_val_data, test_size=0.5, random_state=42, stratify=test_val_data['classe'])

print('Distribution each class train:', train_data_list['classe'].value_counts())
print('Distribution each class test:', test_data_list['classe'].value_counts())
print('Distribution each class val:', val_data_list['classe'].value_counts())

output_dir = r'D:\Users\dayv\mamografIA_upscaler\outputs\tsv_files_train_valid_pulmao'
os.makedirs(output_dir, exist_ok=True)

train_data_tsv_path = os.path.join(output_dir, "train.tsv")
val_data_tsv_path = os.path.join(output_dir, "validation.tsv")
test_data_tsv_path = os.path.join(output_dir, "test.tsv")

train_data_list.to_csv(train_data_tsv_path, index=False, sep="\t")
val_data_list.to_csv(val_data_tsv_path, index=False, sep="\t")
test_data_list.to_csv(test_data_tsv_path, index=False, sep="\t")



    

        
        