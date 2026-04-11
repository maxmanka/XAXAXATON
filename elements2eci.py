import numpy as np
import pandas as pd
import Const


def elements2eci(elements: np.ndarray) -> np.ndarray:
    """
    Перевод орбитальных элементов в ECI (X, Y, Z, VX, VY, VZ).

    Принимает:
        eph – (N,6) или (N,6,T)
            0 – a [km] (Большая полуось)
            1 – e [-] (Эксцентриситет)
            2 – ω [rad] (Argument of perigee)
            3 – Ω [rad] (RAAN)
            4 – i [rad] (Inclination)
            5 – M [rad] (Mean anomaly)

    Возвращает:
        eci – (N,6) или (N,6,T) (X, Y, Z, VX, VY, VZ)
    """
    mu = Const.earthGM    # км³/с²

    # Приводим вход к трёхмерному виду (N,6,T)
    elements = np.asarray(elements, dtype=float)
    if elements.ndim == 2:                           # один момент времени
        elements = elements[:, :, np.newaxis]            # (N,6,1)

    a   = elements[:, 0, :]          # (N,T)
    ecc = elements[:, 1, :]
    w   = elements[:, 2, :]          # ω   – аргумент перицентра
    Omega = elements[:, 3, :]        # Ω   – долгота восходящего узла
    inc = elements[:, 4, :]          # i   – наклон
    M   = elements[:, 5, :]          # M   – средняя аномалия

    # ------------------------------------------------------------------
    # 1. Истинная аномалия (MATLAB‑логика: ν = M - ω)
    # ------------------------------------------------------------------
    nu = M - w                                          # (N,T)

    # ------------------------------------------------------------------
    # 2. Параметр орбиты и вспомогательные величины
    # ------------------------------------------------------------------
    p = a * (1.0 - ecc**2)                              # (N,T)
    sqrtUoP = np.sqrt(mu / p)                           # (N,T)

    cos_nu = np.cos(nu)
    sin_nu = np.sin(nu)

    # ------------------------------------------------------------------
    # 3. Координаты и скорости в системе PQW
    # ------------------------------------------------------------------
    # 3.1  позиция PQW
    r_pf_x = p * cos_nu / (1.0 + ecc * cos_nu)          # (N,T)
    r_pf_y = p * sin_nu / (1.0 + ecc * cos_nu)          # (N,T)

    # 3.2  скорость PQW
    v_pf_x = -sqrtUoP * sin_nu                         # (N,T)
    v_pf_y =  sqrtUoP * (ecc + cos_nu)                  # (N,T)

    zeros = np.zeros_like(r_pf_x)

    # Меняем порядок осей, как делает MATLAB `permute([ … ], [3 2 1])`
    # После этого получаем (T, N, 3)
    R_pqw = np.stack([r_pf_x, r_pf_y, zeros], axis=2)   # (N,T,3)
    V_pqw = np.stack([v_pf_x, v_pf_y, zeros], axis=2)   # (N,T,3)

    R_pqw = np.transpose(R_pqw, (1, 0, 2))   # (T, N, 3)
    V_pqw = np.transpose(V_pqw, (1, 0, 2))   # (T, N, 3)

    # ------------------------------------------------------------------
    # 4. DCM‑матрицы для каждого (N,T)
    # ------------------------------------------------------------------
    # Углы: [-ω, -inc, -Ω]  → (N,T,3) → (T,N,3)  (согласовано с R_pqw/V_pqw)
    angles = np.stack([-w, -inc, -Omega], axis=2)        # (N,T,3)
    angles = np.transpose(angles, (1, 0, 2))             # (T,N,3)

    # Плоский массив (T·N, 3) → вызываем векторизованный angle2dcmFast
    angles_flat = angles.reshape(-1, 3)                  # (T·N, 3)
    dcm_flat = angle2dcmFast(angles_flat)                # (T·N, 3, 3)

    # Возвращаем форму (T, N, 3, 3) – удобно для умножения
    dcm = dcm_flat.reshape(R_pqw.shape[0], R_pqw.shape[1], 3, 3)  # (T,N,3,3)

    # ------------------------------------------------------------------
    # 5. Поворот PQW → ECI (аналог `pagemtimes` в MATLAB)
    # ------------------------------------------------------------------
    # r_eci[k] = R_pqw[k] @ dcm[k].T   (по‑строке)
    # используем einsum, который одновременно умеет «батч‑умножать»
    r_eci = np.einsum('tnj,tnjk->tnk', R_pqw, dcm)   # (T,N,3)
    v_eci = np.einsum('tnj,tnjk->tnk', V_pqw, dcm)   # (T,N,3)

    # ------------------------------------------------------------------
    # 6. Формируем окончательный массив (N,6,T)
    # ------------------------------------------------------------------
    eci = np.concatenate([r_eci, v_eci], axis=2)     # (T,N,6)
    eci = np.transpose(eci, (1, 2, 0))               # (N,6,T)

    # Если вход был единственный момент – убираем лишнее измерение
    if eci.shape[2] == 1:
        eci = np.squeeze(eci, axis=2)                # (N,6)

    return eci

def angle2dcmFast(angles: np.ndarray) -> np.ndarray:
    """
    Транспонированный ZXZ‑Эйлер (полный аналог MATLAB‑функции).

    Параметры
    ----------
    angles : (N, 3)   – массив углов [α, β, γ] в радианах.
                       (в нашем случае α = -ω, β = -inc, γ = -Ω)

    Возвращает
    ----------
    dcm : (N, 3, 3)   – матрицы направления‑косинуса.
                       Для каждого спутника:
                           r_eci = r_pqw @ dcm.T
                           v_eci = v_pqw @ dcm.T
    """
    # --------------------------------------------------------------
    # 1. Разбиваем на отдельные векторы
    # --------------------------------------------------------------
    a = angles[:, 0]          # α  (= –ω)
    b = angles[:, 1]          # β  (= –inc)
    c = angles[:, 2]          # γ  (= –Ω)

    # --------------------------------------------------------------
    # 2. Предварительные синусы / косинусы
    # --------------------------------------------------------------
    sa, ca = np.sin(a), np.cos(a)
    sb, cb = np.sin(b), np.cos(b)
    sc, cc = np.sin(c), np.cos(c)

    # --------------------------------------------------------------
    # 3. Формируем матрицу (по‑строчно, как в MATLAB)
    # --------------------------------------------------------------
    dcm = np.empty((angles.shape[0], 3, 3), dtype=float)

    # первая строка
    dcm[:, 0, 0] = -sa * cb * sc + ca * cc
    dcm[:, 0, 1] = -sa * cc * cb - ca * sc   # ← исправлена перестановка sin‑cos
    dcm[:, 0, 2] =  sa * sb

    # вторая строка
    dcm[:, 1, 0] =  ca * cb * sc + sa * cc
    dcm[:, 1, 1] =  ca * cc * cb - sa * sc
    dcm[:, 1, 2] = -ca * sb

    # третья строка
    dcm[:, 2, 0] =  sb * sc
    dcm[:, 2, 1] =  sb * cc
    dcm[:, 2, 2] =  cb

    return dcm


# Пример использования перевода из кеплеровых элементов в ECI
# (Earth-centered inertial — система координат с началом координат в центре Земли и НЕ вращается с планетой)
if __name__ == '__main__':
    sat_states_file_path = "satstates.csv"
    connectivity_file_path = "satconnectivity.csv"

    # 1. Чтение данных
    states_Table = pd.read_csv(sat_states_file_path)

    # 2. Формирование кеплеровых элементов для перевода в ECI
    raw_elements = np.column_stack([
        states_df['sma'].values,
        states_df['ecc'].values,
        np.deg2rad(states_df['aop'].values),
        np.deg2rad(states_df['raan'].values),
        np.deg2rad(states_df['inc'].values),
        np.deg2rad(states_df['aol'].values)
    ])

    # 3. Перевод кеплеровых элементов в ECI
    eci = elements2eci(raw_elements)
    print(eci)
