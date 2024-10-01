
from pyswarm import pso
import numpy as np

# función objetivo
def funcion_objetivo(x):
    x1, x2 = x
    eq1 = 3 * x1 + 2 * x2 - 9
    eq2 = x1 - 5 * x2 - 4
    return np.abs(eq1) + np.abs(eq2)

lb = [-100, -100]  # limite inf
ub = [100, 100]  # limite sup

num_particulas = 10  # numero de particulas
cantidad_iteraciones = 50  # numero maximo de iteraciones
c1 = 2.0  # componente cognitivo
c2 = 2.0  # componente social
w = 0.5  # factor de inercia

# Llamada a la función pso
solucion_optima, valor_optimo = pso(funcion_objetivo, lb, ub, swarmsize=num_particulas, maxiter=cantidad_iteraciones, debug=True,omega=w, phip=c1, phig=c2)

# Resultados
print("\nSolución óptima (x, y):", solucion_optima)
print("Valor óptimo:", valor_optimo)