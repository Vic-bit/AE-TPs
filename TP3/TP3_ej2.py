# .................................................................
# PSO con restricciones
#      maximizar     500x1 + 400x2 = Z
#
#      sujeto a:
#                    300x1 + 400x2 <= 127000
#                     20x1 +  10x2 <= 4270
# .................................................................
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# función objetivo a maximizar
def f(x):
    return 500 * x[0] + 400 * x[1]  # funcion objetivo: 500x1 + 400x2


# primera restriccion - capital
def g1(x):
    return 300 * x[0] + 400 * x[1] - 127000 <= 0  # restriccion: 300x1 + 400x2 <= 127000


# segunda restriccion - mano de obra
def g2(x):
    return 20 * x[0] + 10 * x[1] - 4270 <= 0  # restriccion: 20x1 +  10x2 <= 4270


# parametros
n_particles = 10  # numero de particulas en el enjambre
n_dimensions = 2  # dimensiones del espacio de busqueda (x1 y x2)
max_iterations = 80  # numero máximo de iteraciones para la optimizacion
c1 = c2 = 2  # coeficientes de aceleracion
w = 0.5  # Factor de inercia

def PSO_restricciones(n_particles, n_dimensions, max_iterations, c1, c2, w_max, w_min=0.4, w_type=''):
    # inicialización de particulas
    x = np.zeros((n_particles, n_dimensions))  # matriz para las posiciones de las particulas
    v = np.zeros((n_particles, n_dimensions))  # matriz para las velocidades de las particulas
    pbest = np.zeros((n_particles, n_dimensions))  # matriz para los mejores valores personales
    pbest_fit = -np.inf * np.ones(n_particles)  # mector para las mejores aptitudes personales (inicialmente -infinito)
    gbest = np.zeros(n_dimensions)  # mejor solución global
    gbest_fit = -np.inf  # mejor aptitud global (inicialmente -infinito)

    # inicializacion de particulas factibles
    for i in range(n_particles):
        while True:  # bucle para asegurar que la particula sea factible
            x[i] = np.random.uniform(0, 10, n_dimensions)  # inicializacion posicion aleatoria en el rango [0, 10]
            if g1(x[i]) and g2(x[i]):  # se comprueba si la posicion cumple las restricciones
                break  # Salir del bucle si es factible
        v[i] = np.random.uniform(-1, 1, n_dimensions)  # inicializar velocidad aleatoria
        pbest[i] = x[i].copy()  # ee establece el mejor valor personal inicial como la posicion actual
        fit = f(x[i])  # calculo la aptitud de la posicion inicial
        if fit > pbest_fit[i]:  # si la aptitud es mejor que la mejor conocida
            pbest_fit[i] = fit  # se actualiza el mejor valor personal

    # Optimizacion
    gbest_fit_list = []    # lista de los gbest en cada iteración
    iteracion_list = []
    for iteracion in range(max_iterations):  # Repetir hasta el número máximo de iteraciones
        for i in range(n_particles):
            fit = f(x[i])  # Se calcula la aptitud de la posicion actual
            # Se comprueba si la nueva aptitud es mejor y si cumple las restricciones
            if fit > pbest_fit[i] and g1(x[i]) and g2(x[i]):
                pbest_fit[i] = fit  # Se actualiza la mejor aptitud personal
                pbest[i] = x[i].copy()  # Se actualizar la mejor posicion personal
                if fit > gbest_fit:  # Si la nueva aptitud es mejor que la mejor global
                    gbest_fit = fit  # Se actualizar la mejor aptitud global
                    gbest = x[i].copy()  # Se actualizar la mejor posicion global

            # comprobar cual w corresponde
            if w_type == 'constante':
                w = w_max
            elif w_type == 'lineal':
                w = w_max *  (w_max - w_min)/max_iterations * iteracion
            elif w_type == 'sin':
                w = 0
            else:
                print('Ingrese un coeficiente de inercia válido')
                return 

            # actualizacion de la velocidad de la particula
            v[i] = w * v[i] + c1 * np.random.rand() * (pbest[i] - x[i]) + c2 * np.random.rand() * (gbest - x[i])
            x[i] += v[i]  # Se actualiza la posicion de la particula

            # se asegura de que la nueva posicion esté dentro de las restricciones
            if not (g1(x[i]) and g2(x[i])):
                # Si la nueva posicion no es válida, revertir a la mejor posicion personal
                x[i] = pbest[i].copy()

        # Se agrega a la lista el mejor gbest
        gbest_fit_list.append(gbest_fit)
        iteracion_list.append(iteracion+1)

    return gbest, gbest_fit, gbest_fit_list, iteracion_list

# Se imprime la mejor solucion encontrada y también su valor optimo
print('-----------------------------------------------------------------------------------------')
print('a')          
gbest, gbest_fit, gbest_fit_list, iteracion_list = PSO_restricciones(n_particles, n_dimensions, max_iterations, c1, c2, w, w_type='constante')
print(f"Mejor solucion: [{gbest[0]:.4f}, {gbest[1]:.4f}]")
print(f"Valor optimo: {gbest_fit}")

print('-----------------------------------------------------------------------------------------')
print('d') # d - Gráfica de la línea que muestra gbest

plt.figure(1,figsize = (10,5))
plt.plot(iteracion_list, np.array(gbest_fit_list),'g',label = 'gbest_fit')
plt.xlabel('Cantidad de iteraciones')
plt.ylabel('gbest_fit')
plt.title('Gráfico de gbest_fit VS iteraciones')
plt.legend()
plt.grid()
plt.axis('tight')
plt.xlim(1,max_iterations)
plt.show()

print('------------------------------------------------------------------------------------------------')
print('e)')

# w = 0.9
w_09 = 0.9

# w lineal, variando de un wmax de 0.9 a uno de 0.4
# w_lineal = w_max *  (w_max - w_min)/max_iterations * iteracion
w_max = 0.9
w_min = 0.4

# w = 0, con factor de constricción de phi = 5 = c1 + c2
w_0 = 0
c1_phi5 = 3
c2_phi5 = 2

# Inicializar las listas
gbest_fit_w09_list = []
gbest_fit_wlineal_list = []
gbest_fit_w0_list = []

# Iterar obteniendo los el fitness gbest para cada uno de los casos
for _ in range(100):
    _, gbest_fit_w09, _, _ = PSO_restricciones(n_particles, n_dimensions, max_iterations, c1, c2, w_max=w_09, w_type='constante')
    gbest_fit_w09_list.append(gbest_fit_w09)
    
    _, gbest_fit_wlineal, _, _ = PSO_restricciones(n_particles, n_dimensions, max_iterations, c1, c2, w_max=w_max, w_min=w_min, w_type='lineal')
    gbest_fit_wlineal_list.append(gbest_fit_wlineal)

    _, gbest_fit_w0, _, _ = PSO_restricciones(n_particles, n_dimensions, max_iterations, c1_phi5, c2_phi5, w_max=0, w_type='sin')
    gbest_fit_w0_list.append(gbest_fit_w0)

# Graficar el boxplot de los 3 casos de manera comparativa
plt.figure(2, figsize = (10,5))
plt.boxplot([gbest_fit_w09_list, gbest_fit_wlineal_list, gbest_fit_w0_list], labels = ['w=0.9', 'w:lineal', 'w=0'])
plt.xlabel('Valores de w')
plt.ylabel('Fitness gbest')
plt.title('Fitness gbest para diferentes valores de w')
plt.show()

print('------------------------------------------------------------------------------------------------')
print('g)')

# Crear dataframe para tener los datos en formato tabular
df_particles = pd.DataFrame(columns=['n_particles', 'Mejor solución', 'Valor óptimo'])

# Inicializar las listas
n_particles_list = []
mejor_solucion_list = []
valor_optimo_list = []

# Iterar con diferente número de partículas
for particles in range(10,1,-1):
    gbest_n_particles, gbest_fit_n_particles, _, _ = PSO_restricciones(particles, n_dimensions, max_iterations, c1, c2, w, w_type='constante')
    n_particles_list.append(particles)
    mejor_solucion_list.append(gbest_n_particles)
    valor_optimo_list.append(gbest_fit_n_particles) 

# Crear el DataFrame con las listas generadas
df_particles['n_particles'] = n_particles_list
df_particles['Mejor solución'] = mejor_solucion_list
df_particles['Valor óptimo'] = valor_optimo_list

# Imprimir el DataFrame final
print(df_particles)
