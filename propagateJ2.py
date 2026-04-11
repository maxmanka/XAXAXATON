import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime, timedelta

import Const
from elements2eci import elements2eci


def prpagateJ2(
    start_elems: np.ndarray[np.float64], delta_t_list_seconds
) -> np.ndarray[np.float64]:
    """
    Выполняет аналитическое прогнозирование орбитальных элементов с учетом эффекта возмущения J2 на один момент времени для множества заданных КА.

    Args:
        start_elems (NDArray[np.float64]): Начальные классические орбитальные элементы записанные в виде массива формы (число аппаратов, 6)
        в формате:
            [[sma, ecc, inc, aop, raan, mean_anom], ...], где:
            - sma (float): большая полуось [км],
            - ecc (float): эксцентриситет [-],
            - inc (float): наклонение [рад],
            - aop (float): аргумент перицентра [рад],
            - raan (float): долгота восходящего узла [рад],
            - mean_anom (float): средняя аномалия [рад].
        delta_t_list_seconds float: Сдвиг по времени от момента, на который рассчитаны элементы start_elems для прогнозирования, в секундах.

    Returns:
        NDArray[np.float64]: Орбитальные элементы в каждой временной точке в том же формате, что и `start_elems`. Форма (число аппаратов, 6).
    """
    sma_km, ecc, aop_rad_start, raan_rad_start, inc_rad, mean_anom_rad_start = np.split(start_elems, 6, axis=1)

    MU_EARTH = Const.earthGM
    p = sma_km * (1 - ecc**2)
    n = np.sqrt(MU_EARTH / sma_km**3)

    EARTH_RADIUS = Const.earthRadius
    J2 = Const.earthJ2 / Const.earthGM / Const.earthRadius**2;

    aop_rad_velocity = 3 * n * (EARTH_RADIUS**2) * J2 * (4 - 5 * (np.sin(inc_rad) ** 2)) / (4 * (p**2))
    raan_rad_velocity = -3 * n * (EARTH_RADIUS**2) * J2 * np.cos(inc_rad) / (2 * (p**2))
    mean_anom_rad_velocity = n - 3 * n * (EARTH_RADIUS**2) * J2 * np.sqrt(1 - ecc**2) * (
        3 * np.sin(inc_rad) ** 2 - 2
    ) / (4 * (p**2))

    # Подготовка вывода
    sat_count = start_elems.shape[0]
    epoch_count = 1
    res_elems = np.empty((sat_count, 6), dtype=float)

    nu0 = mean_anom_rad_start - aop_rad_start
    denom = 1.0 + ecc * np.cos(nu0)  # (…)
    sinE0 = np.sqrt(1.0 - ecc ** 2) * np.sin(nu0) / denom
    cosE0 = (ecc + np.cos(nu0)) / denom

    E0 = np.arctan2(sinE0, cosE0)

    M0 = E0 - ecc * np.sin(E0)

    aop_rad_end = aop_rad_start + delta_t_list_seconds * aop_rad_velocity
    raan_rad_end = raan_rad_start + delta_t_list_seconds * raan_rad_velocity
    M = M0 + delta_t_list_seconds * mean_anom_rad_velocity

    E = M.copy()
    err = np.full((M.shape[0], 1), np.inf)

    it =0
    while np.max(err) > 1e-5 and it < 1000:
        E1 = M + ecc * np.sin(E)
        err = np.abs(E - E1)
        E = E1
        it+=1

    nu = np.arctan2(np.sqrt(1.0 - ecc ** 2) * np.sin(E),
                    np.cos(E) - ecc)

    mean_anom_rad_end = nu + aop_rad_end

    res_elems[:, 0] = sma_km.ravel()
    res_elems[:, 1] = ecc.ravel()
    res_elems[:, 2] = aop_rad_end.ravel() % (2 *np.pi)
    res_elems[:, 3] = raan_rad_end.ravel()  % (2 *np.pi)
    res_elems[:, 4] = inc_rad.ravel()
    res_elems[:, 5] = mean_anom_rad_end.ravel()  % (2 *np.pi)

    return res_elems


# Пример прогноза движения КА на один момент времени
if __name__ == '__main__':
    sat_states_file_path = "satstates.csv"

    # 1. Чтение данных
    states_df = pd.read_csv(sat_states_file_path)

    ids = states_df['id'].values

    # 2. Подготовка данных для пропагатора (J2)
    raw_elements = np.column_stack([
        states_df['sma'].values,
        states_df['ecc'].values,
        np.deg2rad(states_df['aop'].values),
        np.deg2rad(states_df['raan'].values),
        np.deg2rad(states_df['inc'].values),
        np.deg2rad(states_df['aol'].values)
    ])

    time_from_start_sec = 900  # сек

    # 3. Прогноз движения КА на момент времени start_date + time_from_start_sec,
    # если raw_elements заданы на момент времени start_date
    res_elements = prpagateJ2(raw_elements, time_from_start_sec)
    res_eci = elements2eci(res_elements)

