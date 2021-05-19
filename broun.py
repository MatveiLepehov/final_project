import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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

radius = 0.5  # Радиус шариков
mass = 0.5  # Масса шариков

# Границы стенок коробки
Lx = 10
Ly = 10

K = 1 # Коэффициент столкновений между шариками
K1 = 1 # Коэффициент столкновений со стенками

T = 50 # Общее время анимации
n = 5000 # Количество итераций / кадров
tau = np.linspace(0,T,n) # Массив для одного шага анимации
dT = T / n # Время одного шага итерации

N = 8 # Количество чатсиц
p = np.zeros((N,4)) # Массивы для текущих значений положений и скоростей частиц

# Массивы для записи итоговых координат на каждой итерации для итоговой анимации
x = np.zeros((N,n))
y = np.zeros((N,n))

# Массивы для записи х, y, vx, vy для каждой частицы
p[0,0], p[0,1], p[0,2], p[0,3] = -1, 2.5, 1.5, 0.5
p[1,0], p[1,1], p[1,2], p[1,3] = -2, 6, -1, -1
p[2,0], p[2,1], p[2,2], p[2,3] = -3, 3, 1, 1
p[3,0], p[3,1], p[3,2], p[3,3] = -4, 2.1, 3, 1.9
p[4,0], p[4,1], p[4,2], p[4,3] = -5, 1, 4, 1.5
p[5,0], p[5,1], p[5,2], p[5,3] = -6, 5, 7, 2
p[6,0], p[6,1], p[6,2], p[6,3] = -7, 4, 2, 3
p[7,0], p[7,1], p[7,2], p[7,3] = -8, 7, 6, 4



x[0,0], y[0,0] = p[0,0], p[0,1]
x[1,0], y[1,0] = p[1,0], p[1,1]
x[2,0], y[2,0] = p[2,0], p[2,1]
x[3,0], y[3,0] = p[3,0], p[3,1]
x[4,0], y[4,0] = p[4,0], p[4,1]
x[5,0], y[5,0] = p[5,0], p[5,1]
x[6,0], y[6,0] = p[6,0], p[6,1]
x[7,0], y[7,0] = p[7,0], p[7,1]


g = 9.80  # Ускорение свободного падения

def circle_func(x_centre_point,
                y_centre_point,
                R):
    x = np.zeros(30) 
    y = np.zeros(30) 
    for i in range(0, 30, 1): 
        alpha = np.linspace(0, 2*np.pi, 30)
        x[i] = x_centre_point + R*np.cos(alpha[i])
        y[i] = y_centre_point + R*np.sin(alpha[i])

    return x, y

def move_func(s, t):
    x, v_x, y, v_y = s

    dxdt = v_x
    dv_xdt = 0

    dydt = v_y
    dv_ydt = -g

    return dxdt, dv_xdt, dydt, dv_ydt


for k in range(n-1):  # Цикл перебора шагов временеи анимации
    t = [tau[k],tau[k+1]]

    for m in range(N):  # Цикл перебора частиц для столкновений со стенками
        s0 = p[m,0], p[m,2], p[m,1], p[m,3]
        sol = odeint(move_func, s0, t)

        # Перезаписываем положения частиц
        p[m,0] = sol[1,0]
        p[m,2] = sol[1,1]
        p[m,1] = sol[1,2]
        p[m,3] = sol[1,3]

        # Заноим новые положения в итоговый массив для анимации
        x[m,k+1], y[m,k+1] = p[m,0], p[m,1]

        # Проверка условий столкновения с граничными стенками
        res = collision_in_box(p[m,0],p[m,1],p[m,2],p[m,3],Lx,Ly,radius,K1)
        p[m,2], p[m,3] = res[0], res[1] # Пересчет скоростей


    # Циклы перебора частиц для столкновений друг с другом
    for i in range(N): # Базовая частица
        x1, y1, vx1, vy1 = p[i,0], p[i,1], p[i,2], p[i,3] # Запись текущих координат базовой частицы
        x10, y10 = x[i,k], y[i,k] # Запись координат предыдущего шага базовой частицы

        for j in range(i+1,N): # Запись текущих координат остальных частиц
            x2, y2, vx2, vy2 = p[j,0], p[j,1], p[j,2], p[j,3] # Запись текущих
            x20, y20 = x[j,k], y[j,k] # Запись координат предыдущего шага

            # Проверка условий столкновения
            r1 = np.sqrt((x1-x2)**2+(y1-y2)**2)
            r0 = np.sqrt((x10-x20)**2+(y10-y20)**2)
            if  r1 <= radius*2 and r0 > 2*radius:
                res = collision(x1,y1,vx1,vy1,x2,y2,vx2,vy2,radius,mass,mass,K)

                # Перезаписывание условий, в случае столкновения
                p[i,2], p[i,3] = res[0], res[1]
                p[j,2], p[j,3] = res[2], res[3]

# Графический вывод
fig = plt.figure()

plt.plot([Lx, Lx], [-Ly, Ly], '-', color='b')
plt.plot([-Lx, -Lx], [-Ly, Ly], '-', color='b')
plt.plot([-Lx, Lx], [Ly, Ly], '-', color='b')
plt.plot([-Lx, Lx], [-Ly, -Ly], '-', color='b')

ball1, = plt.plot([], [], 'o', color='g', ms=1)
ball2, = plt.plot([], [], 'o', color='g', ms=1)
ball3, = plt.plot([], [], 'o', color='r', ms=1)
ball4, = plt.plot([], [], 'o', color='r', ms=1)
ball5, = plt.plot([], [], 'o', color='g', ms=1)
ball6, = plt.plot([], [], 'o', color='g', ms=1)
ball7, = plt.plot([], [], 'o', color='r', ms=1)

def animate(i):
    ball1.set_data(circle_func(x[0, i], y[0, i], radius))
    ball2.set_data(circle_func(x[1, i], y[1, i], radius))
    ball3.set_data(circle_func(x[2, i], y[2, i], radius))
    ball4.set_data(circle_func(x[3, i], y[3, i], radius))
    ball5.set_data(circle_func(x[4, i], y[2, i], radius))
    ball6.set_data(circle_func(x[5, i], y[3, i], radius))
    ball7.set_data(circle_func(x[6, i], y[2, i], radius))

ani = FuncAnimation(fig, animate, frames=n, interval=1)

plt.axis('equal')
plt.xlim(-Lx, Lx)
plt.ylim(-Ly, Ly)
plt.show()
