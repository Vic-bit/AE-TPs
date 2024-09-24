import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

CROMOSOMASX = 15
CROMOSOMASY = 15
LONGITUD_CROMOSOMA = CROMOSOMASX + CROMOSOMASY
Y_MIN = 0
Y_MAX = 20
X_MIN = -10
X_MAX = 10
TAMANIO_TORNEO = 3  # tamaño del torneo
LANZAMIENTOS = 10 # Cantidad de lanzamientos por método
TAMANIO_POBLACION = 100

def binario_a_decimal_X(cromosoma):
    decimal = int(cromosoma, 2)
    x = X_MIN + decimal * (X_MAX - X_MIN) / ((2 ** CROMOSOMASX) - 1)
    return x

def binario_a_decimal_Y(cromosoma):
    decimal = int(cromosoma, 2)
    y = Y_MIN + decimal * (Y_MAX - Y_MIN) / ((2 ** CROMOSOMASY) - 1)
    return y

def aptitud(cromosoma):
    # Convierto a entrero, el 2 indica que esto pasando un número binario
    x = binario_a_decimal_X (cromosoma[:CROMOSOMASX])
    y = binario_a_decimal_Y(cromosoma[CROMOSOMASX:])
    # Elevo al cuadrado el valor entero y devuelvo
    return 7.7 + 0.15 * x + 0.22 * y - 0.05 * x ** 2 - 0.016 * y ** 2 - 0.007 * x * y

#  -----------------------------------------------------------------
# se inicializa la poblacion
#  -----------------------------------------------------------------
def inicializar_poblacion(tamanio_poblacion, longitud_cromosoma):
    poblacion = []
    for tp in range(tamanio_poblacion):
        cromosoma = ''
        for lc in range(longitud_cromosoma):
            cromosoma = cromosoma+str(random.randint(0, 1))
        poblacion.append(cromosoma)
    return poblacion
#  -----------------------------------------------------------------
# seleccion por ruleta
#  -----------------------------------------------------------------
def seleccion_ruleta(poblacion, aptitud_total):
    probabilidades = []
    for individuo in poblacion:
        prob = aptitud(individuo) / aptitud_total
        probabilidades.append(prob)

    probabilidades_acumuladas = []
    suma = 0
    for prob in probabilidades:
        suma += prob
        probabilidades_acumuladas.append(suma)

    r = random.random()
    for i, acumulada in enumerate(probabilidades_acumuladas):
        if r <= acumulada:
            return poblacion[i]

#  -----------------------------------------------------------------
# seleccion por torneo
#  -----------------------------------------------------------------
def seleccion_torneo(poblacion, tamanio_torneo=TAMANIO_TORNEO):
    progenitores = []
    for _ in range(len(poblacion)):
        candidatos = random.sample(poblacion, tamanio_torneo)
        progenitor = max(candidatos, key=aptitud)  # se selecciona el mejor individuo del torneo
        progenitores.append(progenitor)
    return progenitores


#  -----------------------------------------------------------------
# cruce monopunto con probabilidad de cruza pc = 0.92
#  -----------------------------------------------------------------
def cruce_mono_punto(progenitor1, progenitor2, tasa_cruce):
    if random.random() < tasa_cruce:
        punto_cruce = random.randint(1, len(progenitor1) - 1)
        descendiente1 = progenitor1[:punto_cruce] + progenitor2[punto_cruce:]
        descendiente2 = progenitor2[:punto_cruce] + progenitor1[punto_cruce:]
    else:
        descendiente1, descendiente2 = progenitor1, progenitor2
    return descendiente1, descendiente2


#  -----------------------------------------------------------------
# mutacion
#  -----------------------------------------------------------------
def mutacion(cromosoma, tasa_mutacion):
    cromosoma_mutado = ""
    for bit in cromosoma:
        if random.random() < tasa_mutacion:
            cromosoma_mutado = cromosoma_mutado + str(int(not int(bit)))
        else:
            cromosoma_mutado = cromosoma_mutado + bit
    return cromosoma_mutado


#  -----------------------------------------------------------------
# aplicación de operadores geneticos
#  -----------------------------------------------------------------
def algoritmo_genetico(tamanio_poblacion, longitud_cromosoma, tasa_mutacion, tasa_cruce, generaciones, metodo):
    poblacion = inicializar_poblacion(tamanio_poblacion, longitud_cromosoma)
    mejor_funcion_objetivo_generaciones = []  # Lista para almacenar la aptitud del mejor individuo y grficar luego

    for generacion in range(generaciones):
        #print("Generación:", generacion + 1)

        if metodo == 'ruleta':
            # se calcula aptitud total para luego
            aptitud_total = sum(aptitud(cromosoma) for cromosoma in poblacion)
            # seleccion de progenitores con el metodo ruleta
            progenitores = []
            for _ in range(tamanio_poblacion):
                progenitores.append(seleccion_ruleta(poblacion, aptitud_total))
        elif metodo == 'torneo':
             # Seleccion de progenitores con el metodo torneo
            progenitores = seleccion_torneo(poblacion)

        #  -----------------------------------------------------------------
        # Selección método para obtener los progenitores

        #  -----------------------------------------------------------------
        # Cruce
        descendientes = []
        for i in range(0, tamanio_poblacion, 2):
            descendiente1, descendiente2 = cruce_mono_punto(progenitores[i], progenitores[i + 1], tasa_cruce)
            descendientes.extend([descendiente1, descendiente2])

        #  -----------------------------------------------------------------
        # Mutacion
        descendientes_mutados = []
        for descendiente in descendientes:
            descendientes_mutados.append(mutacion(descendiente, tasa_mutacion))

        # Aquí se aplica elitismo
        if (metodo == 'ruleta' or metodo == 'ranking'):
            # Se reemplazan los peores cromosomas con los mejores progenitores
            poblacion.sort(key=aptitud) # se ordena la poblacion por aptitud en forma ascendente
                                        # se ordena los descendientes por aptitud en forma descendente
            descendientes_mutados.sort(key=aptitud, reverse=True)
            for i in range(len(descendientes_mutados)):
                if aptitud(descendientes_mutados[i]) > aptitud(poblacion[i]):
                    poblacion[i] = descendientes_mutados[i]
        
        else:
            # se reemplaza la poblacion con los descendientes mutados
            poblacion = descendientes_mutados

        # Mostrar el mejor individuo de la generacion
        mejor_individuo = max(poblacion, key=aptitud)  # Buscar el maximo para la aptitud
        mejor_funcion_objetivo_generaciones.append(aptitud(mejor_individuo))

        '''
        print("mi", mejor_individuo)
        print("Mejor individuo:", binario_a_decimal(mejor_individuo), "Aptitud:", aptitud(mejor_individuo))
        print("_________________________________________________________________________________")
        '''
    return max(poblacion, key=aptitud), mejor_funcion_objetivo_generaciones  # se retorna el mejor individuo

seed = 15
random.seed(seed)

mejores_soluciones_ruleta = []
x_ruleta = []
y_ruleta = []

mejores_soluciones_torneo = []
x_torneo = []
y_torneo = []

TASA_MUTACION = 0.1
TASA_CRUCE = 0.5
GENERACIONES = 10

for _ in range(LANZAMIENTOS):
    solucion_ruleta, _ = algoritmo_genetico(TAMANIO_POBLACION, LONGITUD_CROMOSOMA, TASA_MUTACION, TASA_CRUCE, GENERACIONES, 'ruleta')
    mejores_soluciones_ruleta.append(aptitud(solucion_ruleta))
    x_ruleta.append(binario_a_decimal_X (solucion_ruleta[:CROMOSOMASX])) 
    y_ruleta.append(binario_a_decimal_Y(solucion_ruleta[CROMOSOMASX:])) 

    solucion_torneo, _ = algoritmo_genetico(TAMANIO_POBLACION, LONGITUD_CROMOSOMA, TASA_MUTACION, TASA_CRUCE, GENERACIONES, 'torneo')
    mejores_soluciones_torneo.append(aptitud(solucion_torneo))
    x_torneo.append(binario_a_decimal_X (solucion_ruleta[:CROMOSOMASX])) 
    y_torneo.append(binario_a_decimal_Y(solucion_ruleta[CROMOSOMASX:])) 

mejores_soluciones = {
    'ruleta': mejores_soluciones_ruleta,
    'torneo': mejores_soluciones_torneo,
}

df = pd.DataFrame(mejores_soluciones)

print(df)
print("_________________________________________________________________________________")
print('c)')

fig = plt.figure(figsize = (15,15))
ax = plt.axes(projection='3d')
ax.grid()

x = np.arange(X_MIN, X_MAX, (X_MAX-X_MIN)/50)
y = np.arange(Y_MIN, Y_MAX, (Y_MAX-Y_MIN)/50)
X, Y = np.meshgrid(x, y)
Z = 7.7 + 0.15 * X + 0.22 * Y - 0.05 * X ** 2 - 0.016 * Y ** 2 - 0.007 * X * Y

surf = ax.plot_surface(X, Y, Z, cmap = plt.cm.cividis)

# Set axes label
ax.set_xlabel('x', labelpad=20)
ax.set_ylabel('y', labelpad=20)
ax.set_zlabel('z', labelpad=20)

fig.colorbar(surf, shrink=0.5, aspect=8)

x_torneo = np.array(x_torneo)
y_torneo = np.array(y_torneo)
z_torneo =  7.7 + 0.15 * x_torneo + 0.22 * y_torneo - 0.05 * x_torneo ** 2 - 0.016 * y_torneo ** 2 - 0.007 * x_torneo * y_torneo

ax.plot3D(x_torneo, y_torneo,z_torneo,'r', linewidth=10, label="Torneo")

x_ruleta = np.array(x_ruleta)
y_ruleta = np.array(y_ruleta)
z_ruleta =  7.7 + 0.15 * x_ruleta + 0.22 * y_ruleta - 0.05 * x_ruleta ** 2 - 0.016 * y_ruleta ** 2 - 0.007 * x_ruleta * y_ruleta

ax.plot3D(x_ruleta, y_ruleta,z_ruleta,'b', linewidth=10, label="Ruleta")
plt.legend()
plt.show()
