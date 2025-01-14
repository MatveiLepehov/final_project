import numpy as np

def collision(x1, y1, vx1, vy1, x2, y2, vx2, vy2, radius, mass1, mass2, K):

    """Аргументы функции:
    x1,y1,vx1,vy1 - координаты и компоненты скорости 1-ой частицы
    x2,y2,vx2,vy2 - ... 2-ой частицы
    radius,mass1,mass2 - радиус частиц и их массы (массы разные можно задавать,
    радиус для простоты взят одинаковый)
    K - коэффициент восстановления (K=1 для абсолютного упругого удара, K=0
    для абсолютно неупругого удара, 0<K<1 для реального удара)
    Функция возвращает компоненты скоростей частиц, рассчитанные по формулам для
    реального удара, если стокновение произошло. Если удара нет, то возвращаются
    те же значения скоростей, что и заданные в качестве аргументов.
    """
    r12 = np.sqrt((x1-x2)**2 + (y1-y2)**2) #расчет расстояния между центрами частиц
    v1 = np.sqrt(vx1**2 + vy1**2) #расчет модулей скоростей частиц
    v2 = np.sqrt(vx2**2 + vy2**2)


    # проверка условия на столкновение: расстояние должно быть меньше 2-х радиусов
    if r12 <= 2*radius:
        '''вычисление углов движения частиц theta1(2), т.е. углов между
        направлением скорости частицы и положительным направлением оси X.
        Если частица  покоится, то угол считается равным нулю. Т.к. функция
        arccos имеет область значений от 0 до Pi, то в случае отрицательных
        y-компонент скорости для вычисления угла theta1(2) надо из 2*Pi
        вычесть значение arccos(vx/v)
        '''
        if v1 != 0:
            theta1 = np.arccos(vx1 / v1)
        else:
            theta1 = 0
        if v2 != 0:
            theta2 = np.arccos(vx2 / v2)
        else:
            theta2 = 0
        if vy1 < 0:
            theta1 = - theta1 + 2 * np.pi
        if vy2 < 0:
            theta2 = - theta2 + 2 * np.pi

        #вычисление угла соприкосновения.
        if (y1-y2) < 0:
            phi = - np.arccos((x1-x2) / r12) + 2 * np.pi
        else:
            phi = np.arccos((x1-x2) / r12)

        # Пересчет  x-компоненты скорости первой частицы
        VX1 = v1 * np.cos(theta1 - phi) * (mass1 - K * mass2) \
        * np.cos(phi) / (mass1 + mass2)\
        + ((1 + K) * mass2 * v2 * np.cos(theta2 - phi))\
        * np.cos(phi) / (mass1 + mass2)\
        + K * v1 * np.sin(theta1 - phi) * np.cos(phi + np.pi / 2)

        # Пересчет y-компоненты скорости первой частицы
        VY1 = v1 * np.cos(theta1 - phi) * (mass1 - K * mass2) \
        * np.sin(phi) / (mass1 + mass2) \
        + ((1 + K) * mass2 * v2 * np.cos(theta2 - phi)) \
        * np.sin(phi) / (mass1 + mass2) \
        + K * v1 * np.sin(theta1 - phi) * np.sin(phi + np.pi / 2)

        # Пересчет x-компоненты скорости второй частицы
        VX2 = v2 * np.cos(theta2 - phi) * (mass2 - K * mass1) \
        * np.cos(phi) / (mass1 + mass2)\
        + ((1 + K) * mass1 * v1 * np.cos(theta1 - phi)) \
        * np.cos(phi) / (mass1 + mass2)\
        + K * v2 * np.sin(theta2 - phi) * np.cos(phi + np.pi / 2)

        # Пересчет y-компоненты скорости второй частицы
        VY2 = v2 * np.cos(theta2 - phi) * (mass2 - K * mass1) \
        * np.sin(phi) / (mass1 + mass2) \
        + ((1 + K) * mass1 * v1 * np.cos(theta1 - phi)) \
        * np.sin(phi) / (mass1 + mass2)\
        + K * v2 * np.sin(theta2 - phi) * np.sin(phi + np.pi / 2)

    else:
        # если условие столкновнеия не выполнено, то скорости частиц не пересчитываются
        VX1, VY1, VX2, VY2 = vx1, vy1, vx2, vy2

    return VX1, VY1, VX2, VY2

def collision_in_box(x1, y1, vx1, vy1, Lx, Ly, radius, K1):
    """Аргументы функции:
    x1, y1, vx1, vy1 - координаты и компоненты скорости частицы
    Ly - x-я (для вертикальной) или y-я (для горизонтальной) координаты
    коробки. Центр коробки находится в точке с координатами (0, 0)
    radius - радиус частицы
    K1 - коэффициент восстановления (K1=1 для абсолютного упругого удара, K1=0
    для абсолютно неупругого удара, 0<K1<1 для реального удара)
    Функция возвращает компоненты скорости частицы, рассчитанные по формулам для
    реального удара о стенку, если стокновение произошло. Если удара нет,
    то возвращаются те же значения скоростей, что и заданные в
    качестве аргументов.
    """
    if x1 <= (-Lx + radius):  #проверка условия столкновения
        if vx1 > 0: #если частица отлетает от стенки вниз
        #считаем, что стокновения на самом деле нет
            key = 1
        else:
            key = 0

    if x1 >= (Lx - radius):  #проверка условия столкновения
        if vx1 < 0:#если частица отлетает от стенки вверх, то
        #считаем, что стокновения на самом деле нет
            key = 1
        else:
            key = 0

    if (x1 > (-Lx + radius) and x1 < (Lx - radius)):  #проверка условия столкновения
        key=0

    if y1 <= (-Ly + radius):  #проверка условия столкновения
        if vy1 > 0: #если частица стремится пролететь сквозь стенку сверху вниз
        #то считаем, что это и есть стокновение
            key1 = 1
        else:
            key1 = 0

    if y1 >= (Ly - radius):  #проверка условия столкновения
        if vy1 < 0: #если частица стремится пролететь сквозь стенку снизу вверх
        #то считаем, что это и есть стокновение
            key1 = 1
        else:
            key1 = 0

    if (y1 > (-Ly + radius) and y1 < (Ly - radius)):  #проверка условия столкновения
        key1 = 0

    #условие, при котором пересчитываются скорости
    if (x1 <= (-Lx + radius) or x1 >= (Lx - radius)) and key == 0:
        VX = - K1 * vx1
    else:
        VX = vx1
    if (y1 <= (-Ly + radius) or y1 >= (Ly - radius)) and key1 == 0:
        VY = - K1 * vy1
    else:
        VY = vy1
    return VX, VY
