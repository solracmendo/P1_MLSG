"""
Module: plotting/plotting.py
Description: Contiene funciones para generar gráficas de comparación de algoritmos.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

from typing import List, Dict, Any

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from algorithms import Algorithm, EpsilonGreedy


def get_algorithm_label(algo: Algorithm) -> str:
    """
    Genera una etiqueta descriptiva para el algoritmo incluyendo sus parámetros.

    :param algo: Instancia de un algoritmo.
    :type algo: Algorithm
    :return: Cadena descriptiva para el algoritmo.
    :rtype: str
    """
    label = type(algo).__name__
    if isinstance(algo, EpsilonGreedy):
        label += f" (epsilon={algo.epsilon})"
    # elif isinstance(algo, OtroAlgoritmo):
    #     label += f" (parametro={algo.parametro})"
    # Añadir más condiciones para otros algoritmos aquí
    else:
        raise ValueError("El algoritmo debe ser de la clase Algorithm o una subclase.")
    return label


def plot_average_rewards(steps: int, rewards: np.ndarray, algorithms: List[Algorithm]):
    """
    Genera la gráfica de Recompensa Promedio vs Pasos de Tiempo.

    :param steps: Número de pasos de tiempo.
    :param rewards: Matriz de recompensas promedio.
    :param algorithms: Lista de instancias de algoritmos comparados.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), rewards[idx], label=label, linewidth=2)

    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('Recompensa Promedio', fontsize=14)
    plt.title('Recompensa Promedio vs Pasos de Tiempo', fontsize=16)
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()


def plot_optimal_selections(steps: int, optimal_selections: np.ndarray, algorithms: List[Algorithm]):
    """
    Genera la gráfica de Porcentaje de Selección del Brazo Óptimo vs Pasos de Tiempo.
    
    :param steps: Número de pasos de tiempo.
    :param optimal_selections: Matriz de porcentaje de selecciones óptimas.
    :param algorithms: Lista de instancias de algoritmos comparados.
    """

    plt.figure(figsize=(10, 5))
    
    for idx, algo in enumerate(algorithms):
        plt.plot(range(steps), optimal_selections[idx], label=f"Epsilon = {algo.epsilon}")
    
    plt.xlabel("Pasos de Tiempo")
    plt.ylabel("Porcentaje de selección del brazo Óptimo")
    plt.title("Porcentaje de Selección del brazo Óptimo vs Pasos de Tiempo")
    plt.legend()
    plt.grid()
    plt.show()

def plot_arm_statistics(arm_stats: dict, algorithms: List[Algorithm], optimal_arm: int):
    """
    Muestra un gráfico de barras que representa el promedio de las ganancias de cada brazo,
    junto con el número de veces que fue seleccionado y si es el brazo óptimo, para cada algoritmo.

    Parámetros:
    - arm_stats: Un diccionario que contiene las estadísticas de cada brazo por algoritmo.
    - algorithms: Una lista de instancias de algoritmos utilizados en el experimento.
    - optimal_arm: El índice del brazo óptimo.
    """
    for i, algorithm in enumerate(algorithms):
        # Obtener el nombre del algoritmo
        algo_name = algorithm.__class__.__name__  # Usamos el nombre de la clase como identificador
        
        # Obtener las estadísticas del algoritmo actual
        stats = arm_stats[algo_name]
        
        arms = list(stats.keys())
        avg_rewards = [stats[arm]['avg_reward'] for arm in arms]
        selections = [stats[arm]['selections'] for arm in arms]
        is_optimal = [arm == optimal_arm for arm in arms]

        x = np.arange(len(arms))  # Posiciones en el eje X
        width = 0.35  # Ancho de las barras

        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Barras para el promedio de las ganancias
        color = 'tab:blue'
        ax1.set_xlabel('Brazo')
        ax1.set_ylabel('Promedio de Ganancias', color=color)
        ax1.bar(x - width/2, avg_rewards, width, label='Promedio de Ganancias', color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        # Etiquetas en el eje X
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'Brazo {arm+1}\n {selections[i]}\n{"(Óptimo)" if is_optimal[i] else ""}' 
                             for i, arm in enumerate(arms)])
        
        # Título y leyenda
        plt.title(f'Epsilon = {algorithm.epsilon}')
        plt.xticks(rotation=45)  # Rotar las etiquetas 45 grados
        fig.tight_layout()
        fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

        plt.show()

def plot_regret(steps: int,
                regret_accumulated: np.ndarray,
                algorithms: List[Algorithm]):
    """
    Genera la gráfica de Regret Acumulado vs Pasos de Tiempo.
    
    :param steps: Número de pasos de tiempo.
    :param regret_accumulated: Matriz de regret acumulado (algoritmos x pasos).
    :param algorithms: Lista de instancias de algoritmos comparados.
    :param args: Opcional. Parámetros que consideres. P.e. la cota teórica Cte * ln(T).
    """
    # Verificamos que el número de algoritmos coincida con el número de filas en regret_accumulated
    if regret_accumulated.shape[0] != len(algorithms):
        raise ValueError("El número de algoritmos debe coincidir con el número de filas en regret_accumulated.")
    
    # Creamos la figura
    plt.figure(figsize=(10, 6))
    
    # Iteramos sobre cada algoritmo y su regret acumulado
    for i, algorithm in enumerate(algorithms):
        plt.plot(range(1, steps + 1), regret_accumulated[i], label=f'Algoritmo {algorithm.epsilon}')
    
    # Añadimos etiquetas y título
    plt.xlabel('Pasos de Tiempo (T)')
    plt.ylabel('Regret Acumulado')
    plt.title('Regret Acumulado vs Pasos de Tiempo')
    plt.legend()
    plt.grid(True)
    
    # Mostramos la gráfica
    plt.show()