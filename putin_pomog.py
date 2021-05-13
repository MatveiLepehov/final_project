import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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

def collision(x1,y1,vx1,vy1,x2,y2,vx2,vy2,radius,mass1,mass2,K):
    r12=np.sqrt((x1-x2)**2+(y1-y2)**2) 
    v1=np.sqrt(vx1**2+vy1**2) 
    v2=np.sqrt(vx2**2+vy2**2)
    if r12<=2*radius:
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
        if (y1-y2)<0:
            phi = - np.arccos((x1-x2) / r12) + 2 * np.pi
        else:
            phi = np.arccos((x1-x2) / r12)

        VX1 = v1 * np.cos(theta1 - phi) * (mass1 - K * mass2) \
        * np.cos(phi) / (mass1 + mass2)\
        + ((1 + K) * mass2 * v2 * np.cos(theta2 - phi))\
        * np.cos(phi) / (mass1 + mass2)\
        + K * v1 * np.sin(theta1 - phi) * np.cos(phi + np.pi / 2)

        VY1 = v1 * np.cos(theta1 - phi) * (mass1 - K * mass2) \
        * np.sin(phi) / (mass1 + mass2) \
        + ((1 + K) * mass2 * v2 * np.cos(theta2 - phi)) \
        * np.sin(phi) / (mass1 + mass2) \
        + K * v1 * np.sin(theta1 - phi) * np.sin(phi + np.pi / 2)

        VX2 = v2 * np.cos(theta2 - phi) * (mass2 - K * mass1) \
        * np.cos(phi) / (mass1 + mass2)\
        + ((1 + K) * mass1 * v1 * np.cos(theta1 - phi)) \
        * np.cos(phi) / (mass1 + mass2)\
        + K * v2 * np.sin(theta2 - phi) * np.cos(phi + np.pi / 2)
        
        VY2 = v2 * np.cos(theta2 - phi) * (mass2 - K * mass1) \
        * np.sin(phi) / (mass1 + mass2) \
        + ((1 + K) * mass1 * v1 * np.cos(theta1 - phi)) \
        * np.sin(phi) / (mass1 + mass2)\
        + K * v2 * np.sin(theta2 - phi) * np.sin(phi + np.pi / 2)
    else:
        VX1, VY1, VX2, VY2 = vx1, vy1, vx2, vy2     
    return VX1, VY1, VX2, VY2
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
X = []
Y = []
X1 = -7.5
X2 = 7.5
Y1 = -7.5
Y2 = 7.5
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
    if np.abs(x10-X1) <= radius or np.abs(x10-X2) <=radius:
      v_x10 = -v_x10
    if np.abs(y10-Y1) <= radius or np.abs(y10-Y2) <= radius:
      v_y10 = -v_y10
    if np.abs(x20-X1) <= radius or np.abs(x20-X2) <= radius:
      v_x20 = -v_x20
    if np.abs(y20-Y1) <= radius or np.abs(y20-Y2) <= radius:
      v_y20 = -v_y20
fig, ax = plt.subplots()
plt.xlim([X1, X2])
plt.ylim([Y1, Y2])
plt.plot([X1, Y1],[X2, Y1],color='b')
plt.plot([X2, Y1],[X2, Y2],color='b')
plt.plot([X2, Y2],[X1, Y2],color='b')
plt.plot([X1, Y2],[X1, Y1],color='b')
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