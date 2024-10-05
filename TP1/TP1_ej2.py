'''
#  -----------------------------------------------------------------
Algoritmo Genético que encuentra el mínimo de la función objetivo x^2 en el
intervalo [-31, 31]
El rango dado por [-31, 31] es 62 es decir: X_MIN - XMAX por tanto el rango
debe ser contenido por un exponente de 2 tal que 2^exponente >= 62, en este
caso exponente = 6, es decir 2^6=64, ( log2(62)=5.95 ) de este modo obtengo la
longitud del cromosoma, luego solo resta mapear los cromosomas binarios en el
rango [-31, 31]
Para mapear se usa la fórmula:
x=X_MIN+decimal(valor_binario)*(X_MAX - X_MIN)/((2 ** LONGITUD_CROMOSOMA) - 1)
#  -----------------------------------------------------------------
Si necesito trabajar con números reales para obtener mas precisión se procede
igual pero se debe multiplicar el rango por 10 si quiero tener un dígito
decimal, x100 para 2 dígitos decimales, etc. En este caso, el rango es 62,
 multiplicado por 10 (para que proporcione 1 dígito decimal) es 620.
Para saber la dimensión del cromosoma debo despejar el exponente de
2^exponente = 620 , log2(620)=9.28
es decir que el valor que contiene a 620 es exponente=10; 2^10=1024
debo por tanto cambiar la constante LONGITUD_CROMOSOMA = 10, de ese modo ahora
el intervalo será real con 1 dígito [-31.0, 31.0]'''
#  -----------------------------------------------------------------

import random
import matplotlib.pyplot as plt
import random
import pandas as pd

plt.style.use('dark_background')


# parametros generales
TAMANIO_POBLACION = 4
LONGITUD_CROMOSOMA = 10 # Debido a que se requiere un dígito decimal
TASA_MUTACION = 0.09
TASA_CRUCE = 0.85
GENERACIONES = 10
X_MIN = -31
X_MAX = 31
EPSILON = 0.001  # Valor pequeño para evitar división por cero en la funcion fitness
TAMANIO_TORNEO = 3  # tamaño del torneo
LANZAMIENTOS = 30 # Cantidad de lanzamientos por método

# parametros específicos de cada modelo
TAMANIO_POBLACION_RULETA = 8
GENERACIONES_RULETA = 20
TASA_MUTACION_RULETA = 0.12
TAMANIO_POBLACION_TORNEO = 16
GENERACIONES_TORNEO = 15
TASA_MUTACION_TORNEO = 0.15
TAMANIO_POBLACION_RANKING = 8
GENERACIONES_RANKING = 17
TASA_MUTACION_RANKING = 0.05


#  -----------------------------------------------------------------
# funcion para mapear el valor binario a un rango [-31, 31]
#  -----------------------------------------------------------------
def binario_a_decimal(cromosoma):
    decimal = int(cromosoma, 2)
    x = X_MIN + decimal * (X_MAX - X_MIN) / ((2 ** LONGITUD_CROMOSOMA) - 1)
    return x


#  -----------------------------------------------------------------
# Aqui en las proximas lineas se puede ver que mi funcion objetivo es
# a veces diferente de mi funcion fitness, depende del problema a resolver
#  -----------------------------------------------------------------


#  -----------------------------------------------------------------
# funcion objetivo x^2
#  -----------------------------------------------------------------
def funcion_objetivo(x):
    return x ** 2


#  -----------------------------------------------------------------
# funcion fitness o tambien llamada funcion de aptitud (1/(x^2 + epsilon))
#  -----------------------------------------------------------------
def aptitud(cromosoma):
    x = binario_a_decimal(cromosoma)
    return 1 / (funcion_objetivo(x) + EPSILON)


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

# -----------------------------------------------------------------
# seleccion por ranking lineal
# -----------------------------------------------------------------
def seleccion_ranking(poblacion):
    # se calcula la aptitud de cada individuo
    aptitudes = []
    for individuo in poblacion:
        aptitudes.append(aptitud(individuo))

    # se ordena la poblacion por aptitud (mayor aptitud es mejor)
    # x es la tupla de zip, de la cual se toma el segundo elemento (x[1]) para
    # ser ordenada la tupla por ese elemento, en este caso por aptitud
    poblacion_ordenada = sorted(zip(poblacion, aptitudes), key=lambda x: x[1])

    # se calcula probabilidades segun el ranking lineal
    N = len(poblacion)
    s = 1.7  # Factor de seleccion comunmente usado
    probabilidades = []
    for i in range(N):
        prob = (2 - s) / N + (2 * i * (s - 1)) / (N * (N - 1))
        probabilidades.append(prob)

    # se selecciona un progenitor basado en las probabilidades
    r = random.random()
    suma = 0
    for i in range(N):
        suma = suma + probabilidades[i]
        if r <= suma:
            return poblacion_ordenada[i][0]  # se retornar el cromosomas


#  -----------------------------------------------------------------
# cruce monopunto con probabilidad de cruza pc = 0.85
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

        #  -----------------------------------------------------------------
        # Selección método para obtener los progenitores
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
        else:
            # seleccion de progenitores con el metodo de ranking lineal
            progenitores = []
            for _ in range(tamanio_poblacion):
                progenitores.append(seleccion_ranking(poblacion))

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

        # Mejor individuo de la generacion
        mejor_individuo = max(poblacion, key=aptitud)  # Buscar el maximo para la aptitud
        mejor_funcion_objetivo_generaciones.append(funcion_objetivo(binario_a_decimal(mejor_individuo)))

    return max(poblacion, key=aptitud), mejor_funcion_objetivo_generaciones


#  -----------------------------------------------------------------
# ejecucion principal del algoritmo genetico
#  -----------------------------------------------------------------
print("_________________________________________________________________________________")
print('a)')

# Se le da un seed arbitrario
seed = 11
random.seed(seed)

mejores_soluciones_ruleta = []
mejores_soluciones_torneo = []
mejores_soluciones_ranking = []

for _ in range(LANZAMIENTOS):
    solucion_ruleta, _ = algoritmo_genetico(TAMANIO_POBLACION, LONGITUD_CROMOSOMA, TASA_MUTACION, TASA_CRUCE, GENERACIONES, 'ruleta')
    mejores_soluciones_ruleta.append(binario_a_decimal(solucion_ruleta))
    solucion_torneo, _ = algoritmo_genetico(TAMANIO_POBLACION, LONGITUD_CROMOSOMA, TASA_MUTACION, TASA_CRUCE, GENERACIONES, 'torneo')
    mejores_soluciones_torneo.append(binario_a_decimal(solucion_torneo))
    solucion_ranking, _ = algoritmo_genetico(TAMANIO_POBLACION, LONGITUD_CROMOSOMA, TASA_MUTACION, TASA_CRUCE, GENERACIONES, 'ranking')
    mejores_soluciones_ranking.append(binario_a_decimal(solucion_ranking))

mejores_soluciones = {
    'Solución ranking': mejores_soluciones_ranking,
    'Solución ruleta': mejores_soluciones_ruleta,
    'Solución torneo': mejores_soluciones_torneo
}

df = pd.DataFrame(mejores_soluciones)

# Crea una columna para los lanzamientos 
df = df.reset_index()
df.rename(columns={'index': 'Lanzamientos'}, inplace=True)
df['Lanzamientos'] += 1

print(df.to_string(index=False))


print("_________________________________________________________________________________")
print('b)')

df_algoritmos_resumen = df[['Solución ranking','Solución ruleta','Solución torneo']].describe().loc[['min','mean','max','std']].transpose()
df_algoritmos_resumen.rename(columns={'min': 'Mínimo', 'mean': 'Media', 'max': 'Máximo', 'std': 'Desv. Est.'}, inplace=True)
print(df_algoritmos_resumen.to_string(index=True))

print("_________________________________________________________________________________")
print('d)')
# Greedy search
tamanios_poblaciones = [10, 12, 14]
tasas_mutaciones = [0.08, 0.09, 0.10, 0.11, 0.12]
generaciones_metodo = [10, 11, 12, 13, 14, 15] #[4, 6, 8, 10, 12]

# Inicializar con el mayor valor del intervalo
mejor_solucion_ruleta = bin(X_MAX)
mejor_solucion_torneo = bin(X_MAX)
mejor_solucion_ranking = bin(X_MAX)
mejor_solucion_ruleta_dec = X_MAX
mejor_solucion_torneo_dec = X_MAX
mejor_solucion_ranking_dec = X_MAX
mejores_params_ruleta = []
mejores_params_torneo = []
mejores_params_ranking = []

print(' De ', len(tamanios_poblaciones) * len(tasas_mutaciones) * len(generaciones_metodo), ' combinaciones de parámetros se encontaron:')

for poblacion_param in tamanios_poblaciones:
    for mutacion_param in tasas_mutaciones:
        for generacion_param in generaciones_metodo:
            
            solucion_ruleta, _ = algoritmo_genetico(poblacion_param, LONGITUD_CROMOSOMA, mutacion_param, TASA_CRUCE, generacion_param, 'ruleta')
            #if abs(binario_a_decimal(solucion_ruleta)) < abs(mejor_solucion_ruleta):
            if aptitud(solucion_ruleta) > aptitud(mejor_solucion_ruleta):
                mejor_solucion_ruleta = solucion_ruleta
                mejor_solucion_ruleta_dec = binario_a_decimal(solucion_ruleta)
                mejores_params_ruleta = [poblacion_param, mutacion_param, generacion_param]
            
            solucion_torneo, _ = algoritmo_genetico(poblacion_param, LONGITUD_CROMOSOMA, mutacion_param, TASA_CRUCE, generacion_param, 'torneo')
            #if abs(binario_a_decimal(solucion_torneo)) < abs(mejor_solucion_torneo):
            if aptitud(solucion_torneo) > aptitud(mejor_solucion_torneo):
                mejor_solucion_torneo = solucion_torneo
                mejor_solucion_torneo_dec = binario_a_decimal(solucion_torneo) 
                mejores_params_torneo = [poblacion_param, mutacion_param, generacion_param]
            
            solucion_ranking, _ = algoritmo_genetico(poblacion_param, LONGITUD_CROMOSOMA, mutacion_param, TASA_CRUCE, generacion_param, 'ranking')
            #if abs(binario_a_decimal(solucion_ranking)) < abs(mejor_solucion_ranking):
            if aptitud(solucion_ranking) > aptitud(mejor_solucion_ranking):
                mejor_solucion_ranking = solucion_ranking
                mejor_solucion_ranking_dec = binario_a_decimal(solucion_ranking)
                mejores_params_ranking = [poblacion_param, mutacion_param, generacion_param]


mejores_soluciones = {
    'Método': ['Ranking', 'Ruleta', 'Torneo'],
    'Solución': [mejor_solucion_ranking_dec, mejor_solucion_ruleta_dec,  mejor_solucion_torneo_dec],
    'Parámetros': [mejores_params_ranking, mejores_params_ruleta, mejores_params_torneo]
}

df2 = pd.DataFrame(mejores_soluciones)

print(df2.to_string(index=False))

print("_________________________________________________________________________________")
print('e)')

mejor_solucion_ruleta_mismos_params, mejor_funcion_objetivo_generaciones_ruleta_mismos_params = algoritmo_genetico(TAMANIO_POBLACION, LONGITUD_CROMOSOMA, TASA_MUTACION, TASA_CRUCE, GENERACIONES, 'ruleta')
mejor_solucion_torneo_mismos_params, mejor_funcion_objetivo_generaciones_torneo_mismos_params = algoritmo_genetico(TAMANIO_POBLACION, LONGITUD_CROMOSOMA, TASA_MUTACION, TASA_CRUCE, GENERACIONES, 'torneo')
mejor_solucion_ranking_mismos_params, mejor_funcion_objetivo_generaciones_ranking_mismos_params = algoritmo_genetico(TAMANIO_POBLACION, LONGITUD_CROMOSOMA, TASA_MUTACION, TASA_CRUCE, GENERACIONES, 'ranking')
print("Mejor solución con método ruleta con parámetros generales:", binario_a_decimal(mejor_solucion_ruleta_mismos_params), "Aptitud:", aptitud(mejor_solucion_ruleta_mismos_params))
print("Mejor solución con método torneo con parámetros generales:", binario_a_decimal(mejor_solucion_torneo_mismos_params), "Aptitud:", aptitud(mejor_solucion_torneo_mismos_params))
print("Mejor solución con método ranking con parámetros generales:", binario_a_decimal(mejor_solucion_ranking_mismos_params), "Aptitud:", aptitud(mejor_solucion_ranking_mismos_params))
print("")
mejor_solucion_ruleta_dif_params, mejor_funcion_objetivo_generaciones_ruleta_dif_params = algoritmo_genetico(mejores_params_ruleta[0], LONGITUD_CROMOSOMA, mejores_params_ruleta[1], TASA_CRUCE, mejores_params_ruleta[2], 'ruleta')
mejor_solucion_torneo_dif_params, mejor_funcion_objetivo_generaciones_torneo_dif_params = algoritmo_genetico(mejores_params_torneo[0], LONGITUD_CROMOSOMA, mejores_params_torneo[1], TASA_CRUCE, mejores_params_torneo[2], 'torneo')
mejor_solucion_ranking_dif_params, mejor_funcion_objetivo_generaciones_ranking_dif_params = algoritmo_genetico(mejores_params_ranking[0], LONGITUD_CROMOSOMA, mejores_params_ranking[1], TASA_CRUCE, mejores_params_ranking[2], 'ranking')
print("Mejor solución con método ruleta con parámetros particulares:", binario_a_decimal(mejor_solucion_ruleta_dif_params), "Aptitud:", aptitud(mejor_solucion_ruleta_dif_params))
print("Mejor solución con método torneo con parámetros particulares:", binario_a_decimal(mejor_solucion_torneo_dif_params), "Aptitud:", aptitud(mejor_solucion_torneo_dif_params))
print("Mejor solución con método ranking con parámetros particulares:", binario_a_decimal(mejor_solucion_ranking_dif_params), "Aptitud:", aptitud(mejor_solucion_ranking_dif_params))


fig, ax = plt.subplots(2, 3, figsize=(10, 5))

# Título general
fig.suptitle('Curvas de Convergencia', fontsize=16)

# Gráfico en el primer subplot (ax[0])
ax[0, 0].plot(range(1, GENERACIONES + 1), mejor_funcion_objetivo_generaciones_ruleta_mismos_params, marker='o')
ax[0, 0].set_xlabel('Generación')  # Cambiado a set_xlabel
ax[0, 0].set_ylabel('Valor de la Función Objetivo')  # Cambiado a set_ylabel
ax[0, 0].set_title('Método Ruleta (mismos parámetros)')  # Cambiado a set_title
ax[0, 0].legend(['Ruleta'])
ax[0, 0].grid(True)  # Añadir la grilla al gráfico

# Gráfico en el segundo subplot (ax[1])
ax[0, 1].plot(range(1, GENERACIONES + 1), mejor_funcion_objetivo_generaciones_torneo_mismos_params, marker='o')
ax[0, 1].set_xlabel('Generación')  # Cambiado a set_xlabel
ax[0, 1].set_ylabel('Valor de la Función Objetivo')  # Cambiado a set_ylabel
ax[0, 1].set_title('Método Torneo (mismos parámetros)')  # Cambiado a set_title
ax[0, 1].legend(['Torneo'])
ax[0, 1].grid(True)  # Añadir la grilla al gráfico

# Gráfico en el segundo subplot (ax[2])
ax[0, 2].plot(range(1, GENERACIONES + 1), mejor_funcion_objetivo_generaciones_ranking_mismos_params, marker='o')
ax[0, 2].set_xlabel('Generación')  # Cambiado a set_xlabel
ax[0, 2].set_ylabel('Valor de la Función Objetivo')  # Cambiado a set_ylabel
ax[0, 2].set_title('Método Ranking (mismos parámetros)')  # Cambiado a set_title
ax[0, 2].legend(['Ranking'])
ax[0, 2].grid(True)  # Añadir la grilla al gráfico

# Gráfico en el primer subplot (ax[3])
ax[1, 0].plot(range(1, mejores_params_ruleta[2] + 1), mejor_funcion_objetivo_generaciones_ruleta_dif_params, marker='o')
ax[1, 0].set_xlabel('Generación')  # Cambiado a set_xlabel
ax[1, 0].set_ylabel('Valor de la Función Objetivo')  # Cambiado a set_ylabel
ax[1, 0].set_title('Método Ruleta (diferentes parámetros)')  # Cambiado a set_title
ax[1, 0].legend(['Ruleta'])
ax[1, 0].grid(True)  # Añadir la grilla al gráfico

# Gráfico en el segundo subplot (ax[4])
ax[1, 1].plot(range(1, mejores_params_torneo[2] + 1), mejor_funcion_objetivo_generaciones_torneo_dif_params, marker='o')
ax[1, 1].set_xlabel('Generación')  # Cambiado a set_xlabel
ax[1, 1].set_ylabel('Valor de la Función Objetivo')  # Cambiado a set_ylabel
ax[1, 1].set_title('Método Torneo (diferentes parámetros)')  # Cambiado a set_title
ax[1, 1].legend(['Torneo'])
ax[1, 1].grid(True)  # Añadir la grilla al gráfico

# Gráfico en el segundo subplot (ax[5])
ax[1, 2].plot(range(1, mejores_params_ranking[2] + 1), mejor_funcion_objetivo_generaciones_ranking_dif_params, marker='o')
ax[1, 2].set_xlabel('Generación')  # Cambiado a set_xlabel
ax[1, 2].set_ylabel('Valor de la Función Objetivo')  # Cambiado a set_ylabel
ax[1, 2].set_title('Método Ranking (diferentes parámetros)')  # Cambiado a set_title
ax[1, 2].legend(['Ranking'])
ax[1, 2].grid(True)  # Añadir la grilla al gráfico

plt.tight_layout()  # Asegura que los subplots no se superpongan
plt.show()

