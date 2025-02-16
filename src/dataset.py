import pickle

from tool.utils import load_data

if __name__=="__main__":
    # 加载融合后的数据
    # with open('datasets/mimic-datasets/mimic3_all/ts_note_all.pkl', 'rb') as f:
    #     merged_data = pickle.load(f)
    merged_data = load_data()
    
    # 查看数据结构样例
    sample = merged_data[0]
    print(type(sample))
    print((sample.keys()))
    print(f"PatientID: {sample['PatientID']}")
    print(f"EHR特征维度: {(sample['X'])}")  # [4（时间步）, D]
    print(f"文本数量: {len(sample['Texts'])}")
    print(f"标签-死亡率/LOS/再入院: {sample['Y']}") # [4, 3]

    {
        'PatientID': "患者ID",
        'X': "EHR特征矩阵",
        'Y': "三重预测标签",
        'Texts': "临床文本列表"
    }
    '''LOS：住院时间标签 (Length of Stay, LOS)'''