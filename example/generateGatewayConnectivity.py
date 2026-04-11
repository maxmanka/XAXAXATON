import numpy as np
import pandas as pd
import os, sys, ast

from astropy import units as u
from astropy.time import Time
from astropy.coordinates import (
    CartesianRepresentation,
    CartesianDifferential,
    GCRS,          # «Earth‑Centered Inertial» (ECI)
    ITRS,          # «Earth‑Centered Earth‑Fixed» (ECEF)
)

from datetime import datetime

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)


import Const
from elements2eci import elements2eci


def generateGatewayConnectivity(satStatesFilePath = "satstates.csv",
                                gatewayFilePath = "gateways.csv",
                                epoch = [2026, 7, 1, 0, 0, 0],
                                connectivityFilePath = "satconnectivity.csv"):
    """
    GENERATEGATEWAYCONNECTIVITY Формирует матрицу связности между шлюзовыми станциями (ШС) и КА.
    
       Функция генерирует матрицу связности, определяющую, к каким КА подключается каждая ШС
       на заданный момент времени. 
       Алгоритм в качестве примера выполняет случайный выбор КА для каждого ШС
       в количестве, равном числу антенн на станции.

       Inputs:
           satStatesFilePath (char | string) - Путь к файлу 'satstates.csv', содержащему
               таблицу орбитальных элементов всех КА.
           gatewayFilePath (char | string)   - Путь к файлу 'gateways.csv', содержащему
               координаты и количество антенн шлюзовых станций
           epoch (numeric [1x6])             - Эпоха (дата и время), для которой строится
               связность, в формате datevec [YYYY MM DD HH MM SS].
           connectivityFilePath (char | string) - Путь для сохранения результирующего CSV-файла.
    
       Outputs:
           Функция создает CSV-файл по адресу connectivityFilePath.
           Файл содержит матрицу размерности (N_ШС x N_КА), где:
               1 - соединение установлено,
               0 - соединение отсутствует.
    
       Notes:
           Это базовый пример алгоритма. Данный метод реализует жадный подход:
           каждая ШС выбирает случайные КА до заполнения всех своих антенн.
           При проверке решения данная матрица будет сверяться с реальной
           геометрической видимостью.
    
       Example:
           generateGatewayConnectivity('satstates.csv', 'gateways.csv', [2026 7 1 0 0 0], 'gc_res.csv');
    """
    if isinstance(epoch, str):
        # Преобразуем строку вида "[2038, 2, 13, 15, 31, 0.135312]" в настоящий список int
        epoch = ast.literal_eval(epoch)
    epoch = [int(epoch[i]) for i in range(6)]

    # 1. Чтение данных
    statesTable = pd.read_csv(satStatesFilePath)
    gatewayTable = pd.read_csv(gatewayFilePath)

    # 2. Генерация матрицы соединений
    # Пример алгоритма
    numSats = len(statesTable)
    numGateways = len(gatewayTable)
    gatewayConnectivity = np.zeros((numGateways, numSats), dtype=bool)

    for gw_idx in range(numGateways):
        n_ant = gatewayTable.at[gw_idx, 'antennas']
        if n_ant:
            gatewayConnectivity[gw_idx, np.random.randint(0, numSats, n_ant)] = True

    # Может пригодиться
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

    # первод из ECI в ECEF в заданный момент времени для одного КА
    # dt = datetime(*epoch)
    dt = datetime(year=epoch[0],
                  month=epoch[1],
                  day=epoch[2],
                  hour=epoch[3],
                  minute=epoch[4],
                  second=epoch[5])
    t = Time(dt, scale='utc')  # Astropy Time

    sat_idx = 0
    pos_eci = eci[sat_idx, :3]
    vel_eci = eci[sat_idx, 3:]
    ecef_pos, ecef_vel = eci2ecef_astropy(t, pos_eci, vel_eci)


    # 3. Сохранение результатов в файл CSV
    pd.DataFrame(gatewayConnectivity.astype(int)).to_csv(connectivityFilePath, header=False, index=False)


def eci2ecef_astropy(t, pos_eci, vel_eci):
    """
    Переводит позицию и скорость из ECI (GCRS) в ECEF (ITRS).

    Parameters
    ----------
    t : astropy.time.Time
        Время наблюдения (может быть скаляром или массивом).
    pos_eci : array‑like, shape (..., 3)
        Координаты в системе ECI (GCRS) **в метрах**.
    vel_eci : array‑like, shape (..., 3)
        Скорости в системе ECI **в метрах/секунду**.

    Returns
    -------
    pos_ecef : ndarray, shape (..., 3)
    vel_ecef : ndarray, shape (..., 3)
    """
    # 1) Представление позиции
    rep = CartesianRepresentation(pos_eci * u.m)

    # 2) Добавляем дифференциал
    diff = CartesianDifferential(vel_eci * u.m / u.s)
    rep = rep.with_differentials(diff)

    # 3) Объект GCRS (ECI) в нужный момент времени
    gcrs = GCRS(rep, obstime=t)

    # 4) Переводим в ITRS (ECEF)
    itrs = gcrs.transform_to(ITRS(obstime=t))

    # 5) Вынимаем значения в чистый numpy
    pos_ecef = itrs.cartesian.xyz.to_value(u.m).T
    vel_ecef = itrs.cartesian.differentials['s'].d_xyz.to_value(u.m / u.s).T
    return pos_ecef, vel_ecef


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        satStatesFilePath = sys.argv[1]
        gatewayFilePath = sys.argv[2]
        epoch = sys.argv[3]
        connectivityFilePath = sys.argv[4]
        generateGatewayConnectivity(satStatesFilePath, gatewayFilePath, epoch, connectivityFilePath)
    else:
        # Запуск без аргументов (для отладки внутри Python IDE)
        generateGatewayConnectivity()
