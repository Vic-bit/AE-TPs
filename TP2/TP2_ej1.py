# ..................................................................................
# algoritmo PSO que maximiza la funcion unimodal f(x, y) = sin(x)+sin(x^2)
# ..................................................................................

import numpy as np
import matplotlib.pyplot as plt

# funcion objetivo
def funcion_objetivo(x):
    return np.sin(x) + np.sin(x**2)

# parametros
num_particulas = 2  # numero de particulas
dim = 1  # dimensiones, solo una
cantidad_iteraciones = 30  # maximo numero de iteraciones
c1 = 1.49  # componente cognitivo
c2 = 1.49  # componente social
w = 0.5  # factor de inercia
limite_inf = 0  # limite inferior de busqueda
limite_sup = 10  # limite superior de busqueda

# inicializacion
particulas = np.random.uniform(limite_inf, limite_sup, (num_particulas, dim))  # posiciones iniciales de las particulas

velocidades = np.zeros((num_particulas, dim))  # inicializacion de la matriz de velocidades en cero

# inicializacion de pbest y gbest
pbest = particulas.copy()  # mejores posiciones personales iniciales

fitness_pbest = np.empty(num_particulas)  # mejores fitness personales iniciales
for i in range(num_particulas):
    fitness_pbest[i] = funcion_objetivo(particulas[i][0] )

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

        fitness = funcion_objetivo(particulas[i][0])  # Evaluacion de la funcion objetivo para la nueva posicion

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

    # imprimir el mejor global en cada iteracion
    print(f"Iteración {iteracion + 1}: Mejor posición global {gbest}, Valor {fitness_gbest}")


# resultado
print('------------------------------------------------------------------------------------------------')
print('a)')
solucion_optima = gbest  # mejor posicion global final
valor_optimo = fitness_gbest  # mejor fitness global final

print("\nSolucion optima (x):", solucion_optima)
print("Valor optimo:", valor_optimo)


# c - Gráfica de la función objetivo y su máximo

x = np.linspace(limite_inf, limite_sup, 201)
plt.figure(1,figsize = (10,5))
plt.plot(x, np.sin(x) + np.sin(x**2),'r')
plt.scatter(solucion_optima, valor_optimo, c='cyan', label='Punto máximo encontrado')
plt.xlabel('x')
plt.ylabel('Función objetivo')
plt.title('PSO para la maximización de la función objetivo')
plt.legend(['sin(x)+sin(x^2)'])
plt.grid()
plt.axis('tight')
plt.show()


# d - Gráfica de la línea que muestra gbest

plt.figure(2,figsize = (10,5))
plt.plot(iteracion_list, gbest_fit_list,'r')
plt.xlabel('Cantidad de iteraciones')
plt.ylabel('fitness gbest')
plt.title('Gráfico de fitness gbest VS iteraciones')
plt.legend(['fitness gbest'])
plt.grid()
plt.axis('tight')
plt.xlim(1,30)
plt.show()