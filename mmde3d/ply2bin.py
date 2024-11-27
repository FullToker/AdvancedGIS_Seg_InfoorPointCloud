import numpy as np
import pandas as pd
from plyfile import PlyData


def convert_ply(input_path, output_path):
    plydata = PlyData.read(input_path)  # 读取文件
    data = plydata.elements[0].data  # 读取数据
    data_pd = pd.DataFrame(data)  # 转换成 DataFrame
    data_np = np.zeros(data_pd.shape, dtype=np.float32)  # 初始化数组来存储数据
    property_names = data[0].dtype.names  # 读取属性名称
    for i, name in enumerate(property_names):  # 通过属性读取数据
        data_np[:, i] = data_pd[name]
    data_np.astype(np.float32).tofile(output_path)


if __name__ == "__main__":
    """
    convert_ply(
        "/media/fys/T7 Shield/AdvancedGIS/read_test/GE006_downsample.ply",
        "/media/fys/T7 Shield/AdvancedGIS/read_test/GE006_downsample.bin",
    )
    """
    convert_ply(
        "/media/fys/T7 Shield/AdvancedGIS/read_test/UZH_dataset/2C05/2C05 - downsample.ply",
        "/media/fys/T7 Shield/AdvancedGIS/read_test/2C05_downsample.bin",
    )
