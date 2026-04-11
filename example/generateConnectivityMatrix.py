import pandas as pd
import numpy as np
import os
import sys

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import Const
from elements2eci import elements2eci

def generateConnectivityMatrix(sat_states_file_path = "satstates.csv",
                               connectivity_file_path = "satconnectivity.csv"):
    """
    GENERATECONNECTIVITYMATRIX Формирует матрицу связности КА-КА.
    
       Функция генерирует квадратную матрицу, описывающую желаемую топологию
       связи между КА через терминалы лазерной связи (ТЛС).
    
       Inputs:
           satStatesFilePath (char | string) - Путь к файлу 'satstates.csv', содержащему
               таблицу начальных состояний КА (ID, орбитальные элементы, кол-во ТЛС).
           connectivityFilePath (char | string) - Путь для сохранения результирующего CSV-файла.
    
       Outputs:
           Функция создает файл 'satconnectivity.csv' (или по указанному пути).
           Матрица размерности (N_КА x N_КА), где:
               1 - наличие желаемой связи (линка),
               0 - отсутствие связи.
    
       Notes:
           Данный алгоритм является примером. 
           Он выстраивает цепочку связности, соединяя каждый КА с соседними
           по ID (ID+1 и ID-1). Это соответствует линейной топологии "цепочка".
           При проверке решения будет учитываться ограничение по количеству ТЛС
           на борту каждого КА.
    """
    # 1. Чтение данных
    statesTable = pd.read_csv(sat_states_file_path)

    # 2. Генерация матрицы связности
    ids = statesTable['id'].values
    id_diff = np.abs(ids[:, None] - ids[None, :])
    connectivity_matrix = (id_diff == 1).astype(int)

    # может пригодиться:
    # пример использования констант
    earthRadius = Const.earthRadius

    # пример использования перевода из кеплеровых элементов в ECI (Earth-centered inertial — система координат с началом координат в центре Земли и НЕ вращается с планетой)
    raw_elements = np.column_stack([
        statesTable['sma'].values,
        statesTable['ecc'].values,
        np.deg2rad(statesTable['aop'].values),
        np.deg2rad(statesTable['raan'].values),
        np.deg2rad(statesTable['inc'].values),
        np.deg2rad(statesTable['aol'].values)
    ])
    eci = elements2eci(raw_elements)

    # 3. Сохранение результата в указанный файл
    pd.DataFrame(connectivity_matrix).to_csv(connectivity_file_path, header=False, index=False)


if __name__ == "__main__":
    # Чтение аргументов из командной строки
    # sys.argv[0] - это имя скрипта, argv[1] - первый аргумент и т.д.
    if len(sys.argv) >= 3:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        generateConnectivityMatrix(input_path, output_path)
    else:
        # Запуск без аргументов (для отладки внутри Python IDE)
        generateConnectivityMatrix()
