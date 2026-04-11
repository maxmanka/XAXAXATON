function generateConnectivityMatrix(satStatesFilePath, connectivityFilePath)
    %GENERATECONNECTIVITYMATRIX Формирует матрицу связности КА-КА.
    %
    %   Функция генерирует квадратную матрицу, описывающую желаемую топологию
    %   связи между КА через терминалы лазерной связи (ТЛС).
    %
    %   Inputs:
    %       satStatesFilePath (char | string) - Путь к файлу 'satstates.csv', содержащему
    %           таблицу начальных состояний КА (ID, орбитальные элементы, кол-во ТЛС).
    %       connectivityFilePath (char | string) - Путь для сохранения результирующего CSV-файла.
    %
    %   Outputs:
    %       Функция создает файл 'satconnectivity.csv' (или по указанному пути).
    %       Матрица размерности (N_КА x N_КА), где:
    %           1 - наличие желаемой связи (линка),
    %           0 - отсутствие связи.
    %
    %   Notes:
    %       Данный алгоритм является примером. 
    %       Он выстраивает цепочку связности, соединяя каждый КА с соседними
    %       по ID (ID+1 и ID-1). Это соответствует линейной топологии "цепочка".
    %       При проверке решения будет учитываться ограничение по количеству ТЛС
    %       на борту каждого КА.
    %
    %   Example:
    %       generateConnectivityMatrix('satstates.csv', 'connectivity.csv');

    arguments (Input)
        satStatesFilePath {mustBeTextScalar} = "satstates.csv"
        connectivityFilePath {mustBeTextScalar} = "satconnectivity.csv"
    end

    % Чтение таблиц
    statesTable = readtable(satStatesFilePath);

    % Генерация матрицы связности
    connectivityMatrix = false(height(statesTable));

    for satIdx = 1:height(statesTable)
        connectivityMatrix(satIdx, abs(statesTable.id - statesTable.id(satIdx)) == 1) = true;
    end

    % Может пригодиться
    earthRadius = Const.earthRadius;

    rawStates.elements = [statesTable.sma, statesTable.ecc, ...
        deg2rad(statesTable.aop), deg2rad(statesTable.raan), ...
        deg2rad(statesTable.inc), deg2rad(statesTable.aol)];

    eci = elements2eci(rawStates.elements);

    
    % Запись в файл
    writematrix(connectivityMatrix, connectivityFilePath);
end