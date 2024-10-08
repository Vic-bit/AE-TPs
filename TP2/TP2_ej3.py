# ..................................................................................
# algoritmo PSO que maximiza la funcion f(x, y) = e^(-0.1 * (x**2 + y**2)) * cos(x) * sin(x)
# ..................................................................................

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# parametros
num_particulas = 10  # numero de particulas
dim = 2  # dimensiones, solo una
cantidad_iteraciones = 15  # maximo numero de iteraciones
c1 = 2.0  # componente cognitivo
c2 = 2.0  # componente social
w = 0.5 # .5  # factor de inercia
limite_inf = -50  # limite inferior de busqueda
limite_sup = 50  # limite superior de busqueda


# funcion objetivo
def funcion_objetivo(x, y):
    return np.exp(-0.1 * (x**2 + y**2)) * np.cos(x) * np.sin(y)


def PSO(num_particulas, dim, cantidad_iteraciones, c1, c2, w, limite_inf, limite_sup):
    # inicializacion
    particulas = np.random.uniform(limite_inf, limite_sup, (num_particulas, dim))  # posiciones iniciales de las particulas

    velocidades = np.zeros((num_particulas, dim))  # inicializacion de la matriz de velocidades en cero

    # inicializacion de pbest y gbest
    pbest = particulas.copy()  # mejores posiciones personales iniciales

    fitness_pbest = np.empty(num_particulas)  # mejores fitness personales iniciales
    for i in range(num_particulas):
        fitness_pbest[i] = funcion_objetivo(particulas[i][0], particulas[i][1])

    gbest = pbest[np.argmax(fitness_pbest)]  # mejor posicion global inicial
    fitness_gbest = np.max(fitness_pbest)  # fitness global inicial

    gbest_fit_list = []    # lista de los gbest en cada iteración
    iteracion_list = []

    # busqueda
    for iteracion in range(cantidad_iteraciones):
        for i in range(num_particulas):  # iteracion sobre cada partícula
            r1, r2 = np.random.rand(), np.random.rand()  # generacion dos numeros aleatorios

            # actualizacion de la velocidad de la particula en cada dimension
            for d in range(dim):
                velocidades[i][d] = (w * velocidades[i][d] + c1 * r1 * (pbest[i][d] - particulas[i][d]) + c2 * r2 * (gbest[d] - particulas[i][d]))

            for d in range(dim):
                particulas[i][d] = particulas[i][d] + velocidades[i][d]  # actualizacion de la posicion de la particula en cada dimension

                # mantenimiento de las partículas dentro de los limites
                particulas[i][d] = np.clip(particulas[i][d], limite_inf, limite_sup)

            fitness = funcion_objetivo(particulas[i][0], particulas[i][1])  # Evaluacion de la funcion objetivo para la nueva posicion

            # actualizacion el mejor personal
            if fitness > fitness_pbest[i]:
                fitness_pbest[i] = fitness  # actualizacion del mejor fitness personal
                pbest[i] = particulas[i].copy()  # actualizacion de la mejor posicion personal

                # actualizacion del mejor global
                if fitness > fitness_gbest:
                    fitness_gbest = fitness  # actualizacion del mejor fitness global
                    gbest = particulas[i].copy()  # actualizacion de la mejor posicion global

        # Se agrega a la lista el mejor gbest
        gbest_fit_list.append(fitness_gbest)
        iteracion_list.append(iteracion+1)

    return gbest, fitness_gbest, gbest_fit_list, iteracion_list


# resultado
print('------------------------------------------------------------------------------------------------')
print('a)')

gbest, fitness_gbest, gbest_fit_list, iteracion_list = PSO(num_particulas, dim, cantidad_iteraciones, c1, c2, w, limite_inf, limite_sup)

solucion_optima = gbest  # mejor posicion global final
valor_optimo = fitness_gbest  # mejor fitness global final

print("\nSolucion optima (x, y):", solucion_optima)
print("Valor optimo:", valor_optimo)


# c - Gráfica de la función objetivo y su máximo

x = np.linspace(limite_inf, limite_sup, 201)
y = np.linspace(limite_inf, limite_sup, 201)
X, Y = np.meshgrid(x,y)

Z = funcion_objetivo(X, Y)

fig = plt.figure(1,figsize = (10,5))
ax = fig.add_subplot(111, projection = '3d')

ax.plot_surface(X, Y, Z, cmap = 'inferno')
ax.scatter(solucion_optima[0], solucion_optima[1], valor_optimo, c='green', s=100, label='Punto máximo encontrado')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')
ax.set_title('PSO para la maximización de la función objetivo con w=0.5')
ax.legend()
ax.grid()
plt.show()


# d - Gráfica de la línea que muestra gbest

plt.figure(2,figsize = (10,5))
plt.plot(iteracion_list, np.array(gbest_fit_list),'g',label = 'gbest_fit')
plt.xlabel('Cantidad de iteraciones')
plt.ylabel('fitness gbest')
plt.title('Gráfico de fitness gbest VS iteraciones con w=0.5')
plt.legend()
plt.grid()
plt.axis('tight')
plt.xlim(1,cantidad_iteraciones)
plt.show()


print('------------------------------------------------------------------------------------------------')
print('e)')

w0=0

gbest, fitness_gbest, gbest_fit_list, iteracion_list = PSO(num_particulas, dim, cantidad_iteraciones, c1, c2, w0, limite_inf, limite_sup)

solucion_optima = gbest         # mejor posicion global final
valor_optimo = fitness_gbest    # mejor fitness global final

print("\nSolucion optima (x, y):", solucion_optima)
print("Valor optimo:", valor_optimo)


# Gráfica de la función objetivo y su máximo

x = np.linspace(limite_inf, limite_sup, 201)
y = np.linspace(limite_inf, limite_sup, 201)
X, Y = np.meshgrid(x,y)

Z = funcion_objetivo(X, Y)

fig = plt.figure(3,figsize = (10,5))
ax = fig.add_subplot(111, projection = '3d')

ax.plot_surface(X, Y, Z, cmap = 'inferno')
ax.scatter(solucion_optima[0], solucion_optima[1], valor_optimo, c='green', s=100, label='Punto máximo encontrado')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')
ax.set_title('PSO para la maximización de la función objetivo con w=0')
ax.legend()
ax.grid()
plt.show()


# Gráfica de la línea que muestra gbest

plt.figure(4,figsize = (10,5))
plt.plot(iteracion_list, np.array(gbest_fit_list),'g',label = 'gbest_fit')
plt.xlabel('Cantidad de iteraciones')
plt.ylabel('fitness gbest')
plt.title('Gráfico de fitness gbest VS iteraciones con w=0')
plt.legend()
plt.grid()
plt.axis('tight')
plt.xlim(1,cantidad_iteraciones)
plt.show()


print('------------------------------------------------------------------------------------------------')
print('f)') # Con w = 0

# Valor de w a probar
w0 = 0
w08 = 0.8

# Inicializar las listas
fitness_gbest_05_list = []
fitness_gbest_0_list = []
fitness_gbest_08_list = []

# Iterar obteniendo los el fitness gbest para cada uno de los casos
for _ in range(100):
    _, fitness_gbest05, _, _ = PSO(num_particulas, dim, cantidad_iteraciones, c1, c2, w, limite_inf, limite_sup)
    fitness_gbest_05_list.append(fitness_gbest05)
    
    _, fitness_gbest0, _, _ = PSO(num_particulas, dim, cantidad_iteraciones, c1, c2, w0, limite_inf, limite_sup)
    fitness_gbest_0_list.append(fitness_gbest0)

    _, fitness_gbest08, _, _ = PSO(num_particulas, dim, cantidad_iteraciones, c1, c2, w08, limite_inf, limite_sup)
    fitness_gbest_08_list.append(fitness_gbest08)

# Graficar el boxplot de los 3 casos de manera comparativa
plt.figure(5, figsize = (10,5))
plt.boxplot([fitness_gbest_05_list, fitness_gbest_0_list, fitness_gbest_08_list], labels = ['w=0.5', 'w=0', 'w=0.8'])
plt.xlabel('Valores de w')
plt.ylabel('Fitness gbest')
plt.title('Fitness gbest para diferentes valores de w')
plt.show()
