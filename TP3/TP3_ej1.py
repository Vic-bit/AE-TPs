# .................................................................
# Ejemplo de PSO con restricciones
#      maximizar     375x1 + 275x2 + 475x3 + 325x4
#
#      sujeto a:
#                     2.5x1 + 1.5x2 + 2.75x3 + 2x4 <= 640
#                     3.5x1 +   3x2 +    3x3 = 2x4 <= 960
# .................................................................
import numpy as np
import matplotlib.pyplot as plt
# función objetivo a maximizar
def f(x):
    return 375 * x[0] + 275 * x[1] + 475 * x[2] + 325 * x[3]  # funcion objetivo


# primera restriccion
def g1(x):
    return 2.5 * x[0] + 1.5 * x[1] + 2.75 * x[2] + 2 * x[3] - 640 <= 0  # restriccion1


# segunda restriccion
def g2(x):
    return 3.5 * x[0] + 3 * x[1] + 3 * x[2] + 2 * x[3] - 640 <= 0 # restriccion2

# tercera restriccion
def g3(x):
    return x[0] >= 0 and x[1] >= 0 and x[2] >= 0 and x[3] >= 0 # restriccion3

# parametros
n_particles = 20  # numero de particulas en el enjambre
n_dimensions = 4  # dimensiones del espacio de busqueda (x1 y x2)
max_iterations = 50  # numero máximo de iteraciones para la optimizacion
c1 = c2 = 1.4944  # coeficientes de aceleracion
w = 0.6  # Factor de inercia
wmax = 0.9
wmin = 0.4

def pso(metodo):
    # inicialización de particulas
    x = np.zeros((n_particles, n_dimensions))  # matriz para las posiciones de las particulas
    v = np.zeros((n_particles, n_dimensions))  # matriz para las velocidades de las particulas
    pbest = np.zeros((n_particles, n_dimensions))  # matriz para los mejores valores personales
    pbest_fit = -np.inf * np.ones(n_particles)  # mector para las mejores aptitudes personales (inicialmente -infinito)
    gbest = np.zeros(n_dimensions)  # mejor solución global
    gbest_fit = -np.inf  # mejor aptitud global (inicialmente -infinito)

    bestFits = []
    # inicializacion de particulas factibles
    for i in range(n_particles):
        while True:  # bucle para asegurar que la particula sea factible
            x[i] = np.random.uniform(0, 10, n_dimensions)  # inicializacion posicion aleatoria en el rango [0, 10]
            if g1(x[i]) and g2(x[i]) and g3(x[i]):  # se comprueba si la posicion cumple las restricciones
                break  # Salir del bucle si es factible
        v[i] = np.random.uniform(-1, 1, n_dimensions)  # inicializar velocidad aleatoria
        pbest[i] = x[i].copy()  # ee establece el mejor valor personal inicial como la posicion actual
        fit = f(x[i])  # calculo la aptitud de la posicion inicial
        if fit > pbest_fit[i]:  # si la aptitud es mejor que la mejor conocida
            pbest_fit[i] = fit  # se actualiza el mejor valor personal

    # Optimizacion
    for iter in range(max_iterations):  # Repetir hasta el número máximo de iteraciones
        for i in range(n_particles):
            fit = f(x[i])  # Se calcula la aptitud de la posicion actual
            # Se comprueba si la nueva aptitud es mejor y si cumple las restricciones
            if fit > pbest_fit[i] and g1(x[i]) and g2(x[i]) and g3(x[i]):
                pbest_fit[i] = fit  # Se actualiza la mejor aptitud personal
                pbest[i] = x[i].copy()  # Se actualizar la mejor posicion personal
                if fit > gbest_fit:  # Si la nueva aptitud es mejor que la mejor global
                    gbest_fit = fit  # Se actualizar la mejor aptitud global
                    gbest = x[i].copy()  # Se actualizar la mejor posicion global

            # actualizacion de la velocidad de la particula
            if metodo == 'inercia':
                v[i] = w * v[i] + c1 * np.random.rand() * (pbest[i] - x[i]) + c2 * np.random.rand() * (gbest - x[i])
            elif metodo == 'inercia_lineal':
                wi = wmax - (wmax - wmin) * iter/max_iterations
                v[i] = wi * v[i] + c1 * np.random.rand() * (pbest[i] - x[i]) + c2 * np.random.rand() * (gbest - x[i])
            elif metodo == 'factor_c':
                phy = c1+c2
                factorC = 2/(np.abs(2-phy-np.sqrt(phy*phy-4*phy)))
                v[i] = factorC * (v[i] + c1 * np.random.rand() * (pbest[i] - x[i]) + c2 * np.random.rand() * (gbest - x[i]))
            x[i] += v[i]  # Se actualiza la posicion de la particula

            # se asegura de que la nueva posicion esté dentro de las restricciones
            if not (g1(x[i]) and g2(x[i]) and g3(x[i])):
                # Si la nueva posicion no es válida, revertir a la mejor posicion personal
                x[i] = pbest[i].copy()
        
        bestFits.append(gbest_fit)
    return gbest, gbest_fit, bestFits

gbest, gbest_fit, bestFits = pso('inercia')
# Se imprime la mejor solucion encontrada y también su valor optimo
print(f"Mejor solucion: [{gbest[0]:.4f}, {gbest[1]:.4f}, {gbest[2]:.4f}, {gbest[3]:.4f}]")
print(f"Valor optimo: {gbest_fit}")

print("_________________________________________________________________________________")
print('d)')
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
# Gráfico en el primer subplot (ax[0])
ax.plot(range(1, max_iterations + 1), bestFits, marker='o')
ax.set_xlabel('Iteración')  # Cambiado a set_xlabel
ax.set_ylabel('Utilidad')  # Cambiado a set_ylabel
ax.set_title('Best fits')  # Cambiado a set_title
ax.grid(True)  # Añadir la grilla al gráfico

plt.legend()
plt.show()
print("_________________________________________________________________________________")
print('e)')
fitness_gbest_inercia_list = []
fitness_gbest_inercia_lineal_list = []
fitness_gbest_factor_c_list = []
w=0.8
for _ in range(100):
    _, fitness_gbest, _ = pso('inercia')
    fitness_gbest_inercia_list.append(fitness_gbest)
    
    _, fitness_gbest, _ = pso('inercia_lineal')
    fitness_gbest_inercia_lineal_list.append(fitness_gbest)
c1=2.05
c2=2.05
for _ in range(100):
    _, fitness_gbest, _= pso('factor_c')
    fitness_gbest_factor_c_list.append(fitness_gbest)

plt.figure(5, figsize = (10,5))
plt.boxplot([fitness_gbest_inercia_list, fitness_gbest_inercia_lineal_list, fitness_gbest_factor_c_list], labels = ['Inercia', 'Inercia lineal dinámica', 'Factor de constricción'])
plt.xlabel('Modelo')
plt.ylabel('Fitness gbest')
plt.title('Fitness gbest para diferentes modelos')

plt.show()

print("_________________________________________________________________________________")
print('h)')
n_particles = 6
fitness_gbest_inercia_list = []
fitness_gbest_inercia_lineal_list = []
fitness_gbest_factor_c_list = []
w=0.8
for _ in range(100):
    _, fitness_gbest, _ = pso('inercia')
    fitness_gbest_inercia_list.append(fitness_gbest)
    
    _, fitness_gbest, _ = pso('inercia_lineal')
    fitness_gbest_inercia_lineal_list.append(fitness_gbest)
c1=2.05
c2=2.05
for _ in range(100):
    _, fitness_gbest, _= pso('factor_c')
    fitness_gbest_factor_c_list.append(fitness_gbest)

plt.figure(5, figsize = (10,5))
plt.boxplot([fitness_gbest_inercia_list, fitness_gbest_inercia_lineal_list, fitness_gbest_factor_c_list], labels = ['Inercia', 'Inercia lineal dinámica', 'Factor de constricción'])
plt.xlabel('Modelo')
plt.ylabel('Fitness gbest')
plt.title('Fitness gbest para diferentes modelos')

plt.show()