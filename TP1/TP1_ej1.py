#  -----------------------------------------------------------------
# Algoritmo Genetico que encuentra el maximo de la funcion x^2
# en el intervalo [0, 31]
# Seleccion por ruleta
# Pc = 0.92
# Pm = 0.1
#  -----------------------------------------------------------------


import random
import pandas as pd

# Parametros
TAMANIO_POBLACION = 4 # Es un hiperparámetro experimental
LONGITUD_CROMOSOMA = 5 # porque 2^5=32, cubriendo el rango [0, 31]
TASA_MUTACION = 0.1
TASA_CRUCE = 0.92
GENERACIONES = 10
LANZAMIENTOS = 10

#  -----------------------------------------------------------------
# Aptitud (y = x^2)
#  -----------------------------------------------------------------
def aptitud(cromosoma):

    # Convierto a entero, el 2 indica que esto pasando un número binario
    x = int(cromosoma, 2)
    
    # Elevo al cuadrado el valor entero y devuelvo
    return x ** 2


#  -----------------------------------------------------------------
# Inicializar la población
#  -----------------------------------------------------------------
def inicializar_poblacion(tamanio_poblacion, longitud_cromosoma):
    poblacion = []
    for tp in range(tamanio_poblacion):
        cromosoma = ""
        for lc in range(longitud_cromosoma):
            #  se crean los cromosomas alelo por alelo en forma aleatoria
            #  hasta completar toda la poblacion
            cromosoma = cromosoma + str(random.randint(0, 1))
        poblacion.append(cromosoma)
    return poblacion


#  -----------------------------------------------------------------
# Seleccion por ruleta
#  -----------------------------------------------------------------
def seleccion_ruleta(poblacion, aptitud_total):
    probabilidades = []
    for individuo in poblacion:

        # Divido la aptitudo del individuo por la total
        prob = aptitud(individuo) / aptitud_total

        # Guardo la probabilidad del individuo, que mientras mayor sea más probabilidad tendrá de ser elegido
        probabilidades.append(prob)

    probabilidades_acumuladas = []
    suma = 0

    for prob in probabilidades:

        # Calculo las probabilidades acumuladas
        suma = suma + prob
        probabilidades_acumuladas.append(suma)

    r = random.random()
    # despues de generar un numero aleatorio entre 0 y 1
    # se itera sobre la lista probabilidades_acumuladas
    # y se obtiene el indice (i) del cromosoma que selecciono para que forme parte de la nueva poblacion
    # tambien se obtiene el valor de probabilidad acumulada en la variable "acumulada"
    for i, acumulada in enumerate(probabilidades_acumuladas):
        if r <= acumulada:
            return poblacion[i]


#  -----------------------------------------------------------------
# Cruce monopunto con probabilidad de cruza pc = 0.92
#  -----------------------------------------------------------------
def cruce_mono_punto(progenitor1, progenitor2, tasa_cruce):

    # Si es menor se produce la cruza, el valor de tasa_cruce generalente es alto, cerca de 0.92
    if random.random() < tasa_cruce:
        punto_cruce = random.randint(1, len(progenitor1) - 1)  # elijo aleatoriamente un punto de cruce

        # Elijo la primera mitad de cada uno según corresponda
        descendiente1 = progenitor1[:punto_cruce] + progenitor2[punto_cruce:]
        descendiente2 = progenitor2[:punto_cruce] + progenitor1[punto_cruce:]
    
    # Pasa a la siguiente generación
    else:
        descendiente1, descendiente2 = progenitor1, progenitor2
    return descendiente1, descendiente2


#  -----------------------------------------------------------------
# mutacion
#  -----------------------------------------------------------------
def mutacion(cromosoma, tasa_mutacion):
    cromosoma_mutado = ""
    for bit in cromosoma:  # aqui se itera cada gen del cromosoma recibido
        
        # Si el número aleatorio es generado a la tasa de mutación modifico el alelo
        if random.random() < tasa_mutacion:
            # se produce la mutacion de un alelo si es que el numero aleatorio generado
            # es inferior que tasa_mutacion tambien llamado "pm" (prob.de mutacion)
            # Lo que antes era 0 pasa a ser 1 y lo convierto en string
            cromosoma_mutado = cromosoma_mutado + str(int(not int(bit)))
        else:
            cromosoma_mutado = cromosoma_mutado + bit
    return cromosoma_mutado


#  -----------------------------------------------------------------
# aplicacion de operadores geneticos
#  -----------------------------------------------------------------
def algoritmo_genetico(tamanio_poblacion, longitud_cromosoma, tasa_mutacion, tasa_cruce, generaciones):
    poblacion = inicializar_poblacion(tamanio_poblacion, longitud_cromosoma)

    for generacion in range(generaciones):  # en este caso se definio un maximo de 10 generaciones
        
        # Selecciono un cronomosma aleatoriamente en una varaible llamada elitism
        #print("Generación:", generacion + 1)

        # se calcula aptitud total (suma de evaluaciones de cada cromosoma) para luego
        # poder obtener la Ps (probabilidad de selección) de cada individuo (Ps = f(i) / sumatoria(f(i)) <-ruleta, es una de las operaciones básicas)
        aptitud_total = 0
        for cromosoma in poblacion: # 10011 es la forma de cada cromosoma

            # Acumulado de las aptitudes de la población, le paso el individuo que esto analizando
            aptitud_total = aptitud_total + aptitud(cromosoma)

        #print("Sumatoria de aptitudes total:", aptitud_total)

        #  -----------------------------------------------------------------
        # seleccion de progenitores con el metodo ruleta
        # se crea una lista vacia de progenitores primero y luego se llama
        # a la funcion seleccion_ruleta para que devuelva de a uno los individuos
        # que se convertiran en futuros progenitores
        progenitores = []
        for _ in range(tamanio_poblacion):
            progenitores.append(seleccion_ruleta(poblacion, aptitud_total))

        #  -----------------------------------------------------------------
        # Cruce
        descendientes = []

        # Voy incrementando de a 2
        for i in range(0, tamanio_poblacion, 2):
            #  se llama a cruce_mono_punto y se le envia pares de progenitores secuencialmente
            # para que se produzca la cruza (en este caso monopunto) segun la tasa_cruce (o Pc)
            descendiente1, descendiente2 = cruce_mono_punto(progenitores[i], progenitores[i + 1], tasa_cruce)
            descendientes.extend([descendiente1, descendiente2])

        #  -----------------------------------------------------------------
        # mutacion
        descendientes_mutados = []
        for descendiente in descendientes:

            # Cargo los individuos mutados
            descendientes_mutados.append(mutacion(descendiente, tasa_mutacion))

        # Selecciono un descendiente_mutado y comparo con elitis, si aptitud(elitis) > aptitud(descenciente mutado)
        # descendiente_mutado = elitism       

        # Aqui se aplica elitismo
        # se reemplazar los peores cromosomas con los mejores progenitores
        # Se ordena en forma ascendente a los progenitores
        poblacion.sort(key=aptitud)

        # Se ordena en forma descendente a los descendiente mutados
        descendientes_mutados.sort(key=aptitud, reverse=True)
        for i in range(len(descendientes_mutados)):

            # Si no es mayor conserva el progentior
            if aptitud(descendientes_mutados[i]) > aptitud(poblacion[i]):
                poblacion[i] = descendientes_mutados[i]

        # mostrar el mejor individuo de la generacion
        mejor_individuo = max(poblacion, key=aptitud)
        #print("Mejor individuo:", int(mejor_individuo, 2), "Aptitud:", aptitud(mejor_individuo))
        #print("_________________________________________________________________________________")

    return max(poblacion, key=aptitud)

# Valor óptimo es el z para un x,y o f(x1,x2,...,xn)
# solución (es lo que quiero encontrar) => x,y tal que z es max, o x tal que y es max para el caso de una variable, o valor de x1,x2,..,xn tal que f es máx
#  -----------------------------------------------------------------
# algoritmo genetico ejecucion principal
#  -----------------------------------------------------------------
print("_________________________________________________________________________________")
print()
mejor_solucion = algoritmo_genetico(TAMANIO_POBLACION, LONGITUD_CROMOSOMA, TASA_MUTACION, TASA_CRUCE, GENERACIONES)
#print("Mejor solución:", int(mejor_solucion, 2), "Aptitud:", aptitud(mejor_solucion))

mejores_soluciones_ruleta = []
for _ in range(LANZAMIENTOS):
    mejores_soluciones_ruleta.append(int(algoritmo_genetico(TAMANIO_POBLACION, LONGITUD_CROMOSOMA, TASA_MUTACION, TASA_CRUCE, GENERACIONES), 2))

mejores_soluciones = {
    'Solución encontrada': mejores_soluciones_ruleta,
}

df = pd.DataFrame(mejores_soluciones)

# Crea una columna para los lanzamientos 
df = df.reset_index()
df.rename(columns={'index': 'Lanzamientos'}, inplace=True)
df['Lanzamientos'] += 1

print(df.to_string(index=False))


# 1 - Mejor solución: 31 Aptitud: 961
# 2 - Mejor solución: 31 Aptitud: 961
# 3 - Mejor solución: 31 Aptitud: 961
# 4 - Mejor solución: 31 Aptitud: 961
# 5 - Mejor solución: 29 Aptitud: 841
# 6 - Mejor solución: 31 Aptitud: 961
# 7 - Mejor solución: 31 Aptitud: 961
# 8 - Mejor solución: 30 Aptitud: 900
# 9 - Mejor solución: 31 Aptitud: 961
# 10 - Mejor solución: 31 Aptitud: 961