import pandas as pd
import os

if __name__ == '__main__':
    
    origem = r'D:\Users\dayv\baixar'
    df = pd.read_excel(os.path.join(origem, 'i2_plaque5mgml_20240717.xlsx'))
    df_remove = df[(df['Number of Slices'] == 50) | (df['Number of Slices'] == 100)]
    df = df.drop(df_remove.index).reset_index(drop=True)

    print(df[df['Slice Thickness'] == 0.8].iloc[0])