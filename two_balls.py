# Лабораторная работа №14. Неупругое двумерное столкновение
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def circle_func(x_centre_point, # х-координата центральной точки окружности
                y_centre_point, # у-координата центральной точки окружности
                R):
    """ Функция, возвращающая точки окружности относительно определенного центра
    """
    x = np.zeros(30) #Создание массива для координаты х
    y = np.zeros(30) #Создание массива для координаты у
    for i in range(0, 30, 1): # Цикл, определяющий множество точек окружности относительно центра
        alpha = np.linspace(0, 2*np.pi, 30)
        x[i] = x_centre_point + R*np.cos(alpha[i])
        y[i] = y_centre_point + R*np.sin(alpha[i])

    return x, y

def collision(x1,y1,vx1,vy1,x2,y2,vx2,vy2,radius,mass1,mass2,K):

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
    r12=np.sqrt((x1-x2)**2+(y1-y2)**2) #расчет расстояния между центрами частиц
    v1=np.sqrt(vx1**2+vy1**2) #расчет модулей скоростей частиц
    v2=np.sqrt(vx2**2+vy2**2)

    #проверка условия на столкновение: расстояние должно быть меньше 2-х радиусов
    if r12<=2*radius:

        '''вычисление углов движения частиц theta1(2), т.е. углов между
        направлением скорости частицы и положительным направлением оси X.
        Если частица  покоится, то угол считается равным нулю. Т.к. функция
        arccos имеет область значений от 0 до Pi, то в случае отрицательных
        y-компонент скорости для вычисления угла theta1(2) надо из 2*Pi
        вычесть значение arccos(vx/v)
        '''
        if v1!=0:
            theta1 = np.arccos(vx1 / v1)
        else:
            theta1 = 0
        if v2!=0:
            theta2 = np.arccos(vx2 / v2)
        else:
            theta2 = 0
        if vy1<0:
            theta1 = - theta1 + 2 * np.pi
        if vy2<0:
            theta2 = - theta2 + 2 * np.pi

        #вычисление угла соприкосновения.
        if (y1-y2)<0:
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
        #если условие столкновнеия не выполнено, то скорости частиц не пересчитываются
        VX1, VY1, VX2, VY2 = vx1, vy1, vx2, vy2

    return VX1, VY1, VX2, VY2

#система уравнений для равномерного движения 2-х частиц в двумерном пространстве
def move_func(s, t):
    x1, v_x1, y1, v_y1, x2, v_x2, y2, v_y2 = s

    dx1dt = v_x1
    dv_x1dt = 0

    dy1dt = v_y1
    dv_y1dt = 0

    dx2dt = v_x2
    dv_x2dt = 0

    dy2dt = v_y2
    dv_y2dt = 0

    return dx1dt, dv_x1dt, dy1dt, dv_y1dt, dx2dt, dv_x2dt, dy2dt, dv_y2dt

T=10
N=1000
tau=np.linspace(0,T,N)

x10,y10=0,0
x20,y20=5,6
v_x10,v_y10=1,1
v_x20,v_y20=-1,-1
mass1=1
mass2=1
radius=0.5
K=1
x1,y1=[],[]
x2,y2=[],[]
x1.append(x10)
x2.append(x20)
y1.append(y10)
y2.append(y20)

for k in range(N-1):
    t=[tau[k],tau[k+1]]
    s0 = x10,v_x10,y10,v_y10,x20,v_x20,y20,v_y20
    sol = odeint(move_func, s0, t)
    x10=sol[1,0]
    x1.append(x10)
    v_x10=sol[1,1]
    y10=sol[1,2]
    y1.append(y10)
    v_y10=sol[1,3]
    x20=sol[1,4]
    x2.append(x20)
    v_x20=sol[1,5]
    y20=sol[1,6]
    y2.append(y20)
    v_y20=sol[1,7]
    r1=np.sqrt((x1[k]-x2[k])**2+(y1[k]-y2[k])**2)
    r0=np.sqrt((x1[k-1]-x2[k-1])**2+(y1[k-1]-y2[k-1])**2)
    if r1<=radius*2 and r0>radius*2:
        res=collision(x10,y10,v_x10,v_y10,x20,y20,v_x20,v_y20,radius,mass1,mass2,K)
        v_x10,v_y10=res[0],res[1]
        v_x20, v_y20=res[2],res[3]

fig, ax = plt.subplots()
ball1, = plt.plot([], [], 'o', color='r', ms=1)
ball2, = plt.plot([], [], 'o', color='r', ms=1)

balls = []

def animate(i):
    ball1.set_data(circle_func(x1[i], y1[i], radius))
    ball2.set_data(circle_func(x2[i], y2[i], radius))


ani = FuncAnimation(fig, animate, frames=N, interval=1)
plt.axis('equal')
plt.ylim(-10, 10)
plt.xlim(-10, 10)

plt.show()
