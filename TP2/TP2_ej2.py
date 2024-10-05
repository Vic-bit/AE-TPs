# ..................................................................................
# algoritmo PSO que minimiza la funcion unimodal f(x, y) = (x-a)^2 + (y + b)^2
# ..................................................................................

import numpy as np
import matplotlib.pyplot as plt

# funcion objetivo hiperboloide eliptico
def funcion_objetivo(x, y, a, b):
    return (x-a)**2 + (y+b)**2

# parametros
num_particulas = 20  # numero de particulas
dim = 2  # dimensiones
cantidad_iteraciones = 10  # maximo numero de iteraciones
c1 = 2.0  # componente cognitivo
c2 = 2.0  # componente social
w = 0  # factor de inercia
limite_inf = -100  # limite inferior de busqueda
limite_sup = 100  # limite superior de busqueda

paramA = float(input('ingrese el valor de a ( f(x, y) = (x-a)^2 + (y + b)^2 ): '))
paramB = float(input('ingrese el valor de b ( f(x, y) = (x-a)^2 + (y + b)^2 ): '))

gbestIteracion = []
if(paramA > 50 or paramA < -50):
    print('Parametro a no pertenerce al rango correcto');
    exit();

if(paramB > 50 or paramB < -50):
    print('Parametro b no pertenerce al rango correcto');
    exit();
# inicializacion
particulas = np.random.uniform(limite_inf, limite_sup, (num_particulas, dim))  # posiciones iniciales de las particulas

velocidades = np.zeros((num_particulas, dim))  # inicializacion de la matriz de velocidades en cero

# inicializacion de pbest y gbest
pbest = particulas.copy()  # mejores posiciones personales iniciales

fitness_pbest = np.empty(num_particulas)  # mejores fitness personales iniciales
for i in range(num_particulas):
    fitness_pbest[i] = funcion_objetivo(particulas[i][0], particulas[i][1], paramA, paramB)

gbest = pbest[np.argmin(fitness_pbest)]  # mejor posicion global inicial
fitness_gbest = np.min(fitness_pbest)  # fitness global inicial

# busqueda
for iteracion in range(cantidad_iteraciones):
    for i in range(num_particulas):  # iteracion sobre cada partícula
        r1, r2 = np.random.rand(), np.random.rand()  # generacion dos numeros aleatorios

        # actualizacion de la velocidad de la particula en cada dimension
        for d in range(dim):
            velocidades[i][d] = (w * velocidades[i][d] + c1 * r1 * (pbest[i][d] - particulas[i][d]) + c2 * r2 * (gbest[d] - particulas[i][d]))

        for d in range(dim):
            particulas[i][d] = particulas[i][d] + velocidades[i][d]  # cctualizacion de la posicion de la particula en cada dimension

            # mantenimiento de las partículas dentro de los limites
            particulas[i][d] = np.clip(particulas[i][d], limite_inf, limite_sup)

        fitness = funcion_objetivo(particulas[i][0], particulas[i][1], paramA, paramB)  # Evaluacion de la funcion objetivo para la nueva posicion

        # actualizacion el mejor personal
        if fitness < fitness_pbest[i]:
            fitness_pbest[i] = fitness  # actualizacion del mejor fitness personal
            pbest[i] = particulas[i].copy()  # actualizacion de la mejor posicion personal

            # actualizacion del mejor global
            if fitness < fitness_gbest:
                fitness_gbest = fitness  # actualizacion del mejor fitness global
                gbest = particulas[i].copy()  # actualizacion de la mejor posicion global

    # imprimir el mejor global en cada iteracion
    print(f"Iteración {iteracion + 1}: Mejor posición global {gbest}, Valor {fitness_gbest}")
    gbestIteracion.append(fitness_gbest)

# resultado
solucion_optima = gbest  # mejor posicion global final
valor_optimo = fitness_gbest  # mejor fitness global final

print("\nSolucion optima (x, y):", solucion_optima)
print("Valor optimo:", valor_optimo)

print("_________________________________________________________________________________")
print('c)')

fig = plt.figure(figsize = (15,15))
ax = plt.axes(projection='3d')
plt.title("Superficie con a=10 y b=10")
ax.grid()

x = np.arange(limite_inf, limite_sup, (limite_sup-limite_inf)/50)
y = np.arange(limite_inf, limite_sup, (limite_sup-limite_inf)/50)
X, Y = np.meshgrid(x, y)
Z = funcion_objetivo(X,Y,paramA,paramB)

surf = ax.plot_surface(X, Y, Z, cmap = plt.cm.cividis)
# Set axes label
ax.set_xlabel('x', labelpad=20)
ax.set_ylabel('y', labelpad=20)
ax.set_zlabel('f(x,y)', labelpad=20)

ax.scatter(solucion_optima[0], solucion_optima[1],valor_optimo,label="Solucion optima",s=50,c='r')
plt.legend()
plt.show()

print("_________________________________________________________________________________")
print('d)')
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
# Gráfico en el primer subplot (ax[0])
ax.plot(range(1, cantidad_iteraciones + 1), gbestIteracion, marker='o')
ax.set_xlabel('Iteración')  # Cambiado a set_xlabel
ax.set_ylabel('Valor de la Función Objetivo')  # Cambiado a set_ylabel
ax.set_title('Gbest por Iteración')  # Cambiado a set_title
ax.grid(True)  # Añadir la grilla al gráfico

plt.legend()
plt.show()

print("_________________________________________________________________________________")
print('f)')

from pyswarm import pso
def funcion_objetivo_pyswarm(Xin):
    x,y = Xin
    return (x-paramA)**2 + (y+paramB)**2

lb = [-100, -100]  # limite inf
ub = [100, 100]  # limite sup
solucion_optima, valor_optimo = pso(funcion_objetivo_pyswarm, lb, ub, swarmsize=num_particulas, maxiter=cantidad_iteraciones, debug=True,omega=w, phip=c1, phig=c2)
print("\nSolución óptima (x, y):", solucion_optima)
print("Valor óptimo:", valor_optimo)

fig = plt.figure(figsize = (15,15))
ax = plt.axes(projection='3d')
plt.title("Superficie con a=10 y b=10 pyswarm")
ax.grid()

x = np.arange(limite_inf, limite_sup, (limite_sup-limite_inf)/50)
y = np.arange(limite_inf, limite_sup, (limite_sup-limite_inf)/50)
X, Y = np.meshgrid(x, y)
Z = funcion_objetivo(X,Y,paramA,paramB)

surf = ax.plot_surface(X, Y, Z, cmap = plt.cm.cividis)
# Set axes label
ax.set_xlabel('x', labelpad=20)
ax.set_ylabel('y', labelpad=20)
ax.set_zlabel('f(x,y)', labelpad=20)

ax.scatter(solucion_optima[0], solucion_optima[1],valor_optimo,label="Solucion optima",s=50,c='r')
plt.legend()
plt.show()