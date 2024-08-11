import numpy as np
import control
from scipy import signal, linalg
import matplotlib.pyplot as plt


#------- Known constants-----# 
lr = 1.39
lf = 1.55
Ca = 20000
Iz = 25854
m = 1888.6
g = 9.81


#----- for loop to input xdot= 2, 5, 8---#
for i in range(3):
	if i == 0:
		xdot = 2
	elif i == 1:
		xdot = 5
	elif i == 2:
		xdot = 8

#-----calculate A,B,C---------#
	A = np.array([[0, 1, 0, 0], [0, -4*Ca / (m * xdot), 4*Ca/m, (2*Ca*(lr - lf))/(m*xdot)], [0, 0, 0, 1], [0, (2*Ca*(lr - lf)) / (Iz * xdot), (2*Ca*(lf - lr)) / Iz, (-2*Ca*(np.power(lf, 2) + np.power(lr, 2))) / (Iz * xdot)]])
	B = np.array([[0], [2*Ca / m], [0], [(2 * Ca* lf) / Iz]])
	C = np.identity(4)


#------calculate P, Q ------#
	P = np.hstack((B, np.matmul(A, B), np.matmul(np.linalg.matrix_power(A, 2), B), np.matmul(np.linalg.matrix_power(A, 3), B)))
	Q = np.vstack((C, np.matmul(C, A), np.matmul(C, np.linalg.matrix_power(A, 2)), np.matmul(C, np.linalg.matrix_power(A, 3))))
	
#------check P, Q ranks------#
	rankP = np.linalg.matrix_rank(P)
	rankQ = np.linalg.matrix_rank(Q)

	
	print(f'P is {P}.')
	print(f'Q is {Q}.')
	print(f'Case #{i+1}: Vx = {xdot}m/s\n')
	print(f'Rank of Controllability matrix = {rankP} --> The system is controllable.')
	print(f'Rank of Observability matrix = {rankQ} --> The system is observable.\n')

log_list=[]
i_list=[]
list_pole1=[]
list_pole2=[]
list_pole3=[]
list_pole4=[]

for i in range(1,41):
    xdot=i
    A = np.array([[0, 1, 0, 0], [0, -4*Ca / (m * xdot), 4*Ca / m, -(2*Ca*(lf - lr))/(m*xdot)], [0, 0, 0, 1], [0, -(2*Ca*(lf - lr)) / (Iz * xdot), (2*Ca*(lf - lr)) / Iz, (-2*Ca*(np.power(lf, 2) + np.power(lr, 2))) / (Iz * xdot)]])
    B = np.array([[0], [2*Ca / m], [0], [(2 * Ca* lf) / Iz]])
    C = np.array([1,1,1,1])
    D=np.array([0])
    controlability_matrix = control.ctrb(A, B)
    _, s, _ = np.linalg.svd(controlability_matrix)
    log_list.append(np.log10(np.max(s)/np.min(s)))
    i_list.append(i)

    sys = control.StateSpace(A, B, C, D)
    p = control.pole(sys)

    #print(p)
    list_pole1.append(np.real(p[0]))
    list_pole2.append(np.real(p[1]))
    list_pole3.append(np.real(p[2]))
    list_pole4.append(np.real(p[3]))


plt.plot(i_list,log_list)
plt.ylabel('log')
plt.xlabel('V(m/s)')
plt.show()

fig = plt.figure()
plt.subplot(2,2,1)
plt.plot(i_list,list_pole1)
plt.ylabel('Re(pole 1)')
plt.xlabel('V(m/s)')


plt.subplot(2,2,2)
plt.plot(i_list,list_pole2)
plt.ylabel('Re(pole 2)')
plt.xlabel('V(m/s)')


plt.subplot(2,2,3)
plt.plot(i_list,list_pole3)
plt.ylabel('Re(pole 3)')
plt.xlabel('V(m/s)')


plt.subplot(2,2,4)
plt.plot(i_list,list_pole4)
plt.ylabel('Re(pole 4)')
plt.xlabel('V(m/s)')

plt.tight_layout()
plt.show()


