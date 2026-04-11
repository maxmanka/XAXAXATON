function generateGatewayConnectivity(satStatesFilePath, gatewayFilePath, epoch, connectivityFilePath)
    %GENERATEGATEWAYCONNECTIVITY Формирует матрицу связности между шлюзовыми станциями (ШС) и КА.
    %
    %   Функция генерирует матрицу связности, определяющую, к каким КА подключается каждая ШС
    %   на заданный момент времени. 
    %   Алгоритм в качестве примера выполняет случайный выбор КА для каждого ШС
    %   в количестве, равном числу антенн на станции.
    %
    %   Inputs:
    %       satStatesFilePath (char | string) - Путь к файлу 'satstates.csv', содержащему
    %           таблицу орбитальных элементов всех КА.
    %       gatewayFilePath (char | string)   - Путь к файлу 'gateways.csv', содержащему
    %           координаты и количество антенн шлюзовых станций.
    %       epoch (numeric [1x6])             - Эпоха (дата и время), для которой строится
    %           связность, в формате datevec [YYYY MM DD HH MM SS].
    %       connectivityFilePath (char | string) - Путь для сохранения результирующего CSV-файла.
    %
    %   Outputs:
    %       Функция создает CSV-файл по адресу connectivityFilePath.
    %       Файл содержит матрицу размерности (N_ШС x N_КА), где:
    %           1 - соединение установлено,
    %           0 - соединение отсутствует.
    %
    %   Notes:
    %       Это базовый пример алгоритма. Данный метод реализует жадный подход:
    %       каждая ШС выбирает случайные КА до заполнения всех своих антенн.
    %       При проверке решения данная матрица будет сверяться с реальной
    %       геометрической видимостью.
    %
    %   Example:
    %       generateGatewayConnectivity('satstates.csv', 'gateways.csv', [2026 7 1 12 0 0], 'gc_res.csv');

    arguments (Input)
        satStatesFilePath   {mustBeTextScalar} = "satstates.csv"
        gatewayFilePath     {mustBeTextScalar} = "gateways.csv"
        epoch               (1,6) {mustBeNumeric} = [2029 01 01 0 0 0]
        connectivityFilePath {mustBeTextScalar} = 'gatewayconnectivity.csv'
    end

    % Чтение таблиц
    gatewayTable = readtable(gatewayFilePath);
    statesTable = readtable(satStatesFilePath);

    % Генерация матрицы связности
    gatewayConnectivity = false(height(gatewayTable), height(statesTable));

    for gatewayIdx = 1:height(gatewayTable)
        satellitesToConnect = randi(height(statesTable), [gatewayTable.antennas(gatewayIdx), 1]);
        gatewayConnectivity(gatewayIdx, satellitesToConnect) = true;
    end

    % Может пригодиться
    earthRadius = Const.earthRadius;

    rawStates.elements = [statesTable.sma, statesTable.ecc, ...
        deg2rad(statesTable.aop), deg2rad(statesTable.raan), ...
        deg2rad(statesTable.inc), deg2rad(statesTable.aol)];

    eci = elements2eci(rawStates.elements);
    ecef = zeros(size(eci(:, 1:3)));

    for satIdx = 1:size(ecef, 1)
        ecef(satIdx, :) = eci2ecef(datetime(epoch), eci(satIdx, 1:3));
    end


    % Запись в файл
    writematrix(gatewayConnectivity, connectivityFilePath);
end