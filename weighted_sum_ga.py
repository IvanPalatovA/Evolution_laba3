"""Генетический алгоритм со взвешенной суммой критериев (Weighted Sum GA)

для задачи формирования недельного меню (Вариант 6).

Используется для прямого экспериментального сравнения с NSGA-II в соответствии
с требованием трека A:
  "Сравнить с подходом взвешенной суммы критериев либо объяснить, почему
   такое сравнение для конкретной задачи ограничено."
"""

import copy
import json
import random
from pathlib import Path

from nsga2_menu import (
    crossover,
    evaluate,
    load_data,
    mutate,
    random_chromosome,
    dominates,
    calculate_hypervolume,
    calculate_spacing,
)


# Границы для нормализации критериев в диапазон [0, 1]
BOUNDS = {
    "cost": (3200.0, 9000.0),
    "nutrient_error": (13.0, 50.0),
    "monotony": (0.0, 100.0),
}


def normalize_objectives(ind):
    """Нормализует значения критериев в диапазон [0, 1]."""
    c = (ind["cost"] - BOUNDS["cost"][0]) / (BOUNDS["cost"][1] - BOUNDS["cost"][0])
    n = (ind["nutrient_error"] - BOUNDS["nutrient_error"][0]) / (BOUNDS["nutrient_error"][1] - BOUNDS["nutrient_error"][0])
    m = (ind["monotony"] - BOUNDS["monotony"][0]) / (BOUNDS["monotony"][1] - BOUNDS["monotony"][0])
    return max(0.0, c), max(0.0, n), max(0.0, m)


def weighted_fitness(ind, weights):
    """Вычисляет скаляризованный штраф (минимизируется): чем меньше, тем лучше."""
    norm_c, norm_n, norm_m = normalize_objectives(ind)
    w1, w2, w3 = weights
    return w1 * norm_c + w2 * norm_n + w3 * norm_m


def run_weighted_sum_single(data, config, weights, seed=3000):
    """Запуск стандартного одноцелевого ГА с фитнесом в виде взвешенной суммы."""
    rng = random.Random(seed)
    pop_size = config["population_size"]
    generations = config["generations"]
    p_cross = config["crossover_probability"]
    p_mut = config["mutation_probability"]
    tourn_size = config.get("tournament_size", 2)
    elite_count = 2

    # Инициализация
    population = []
    for _ in range(pop_size):
        chrom = random_chromosome(data, rng)
        ind = evaluate(chrom, data)
        ind["scalar_fitness"] = weighted_fitness(ind, weights)
        population.append(ind)

    evaluations = pop_size

    for _ in range(generations):
        # Сортируем по скалярному фитнесу (минимизация)
        population.sort(key=lambda x: x["scalar_fitness"])

        new_population = copy.deepcopy(population[:elite_count])

        while len(new_population) < pop_size:
            # Турнирная селекция по скалярному фитнесу
            cand1 = rng.sample(population, tourn_size)
            p1 = min(cand1, key=lambda x: x["scalar_fitness"])
            cand2 = rng.sample(population, tourn_size)
            p2 = min(cand2, key=lambda x: x["scalar_fitness"])

            c1_chrom, c2_chrom = crossover(p1, p2, rng, p_cross)
            c1_chrom = mutate(c1_chrom, data, rng, p_mut)
            c2_chrom = mutate(c2_chrom, data, rng, p_mut)

            c1 = evaluate(c1_chrom, data)
            c1["scalar_fitness"] = weighted_fitness(c1, weights)
            new_population.append(c1)
            evaluations += 1

            if len(new_population) < pop_size:
                c2 = evaluate(c2_chrom, data)
                c2["scalar_fitness"] = weighted_fitness(c2, weights)
                new_population.append(c2)
                evaluations += 1

        population = new_population

    population.sort(key=lambda x: x["scalar_fitness"])
    best = population[0]
    return {
        "best": best,
        "population": population,
        "evaluations": evaluations,
        "weights": weights,
    }


def run_weighted_sum_experiment(data, config, seed=3000):
    """Запускает серию оптимизаций по сетке весовых коэффициентов

    и объединяет полученные решения в единый Парето-фронт.
    """
    weights_list = config.get("weights_for_wsga", [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [0.34, 0.33, 0.33],
        [0.6, 0.2, 0.2],
        [0.2, 0.6, 0.2],
        [0.2, 0.2, 0.6],
    ])

    collected_solutions = []
    total_evals = 0

    # Распределяем бюджет поколений по числу запусков с разными весами
    adjusted_config = copy.deepcopy(config)
    # Чтобы бюджет вычислений был сопоставим с одним запуском NSGA-II:
    adjusted_config["generations"] = max(20, config["generations"] // len(weights_list))

    for idx, w in enumerate(weights_list):
        run_res = run_weighted_sum_single(data, adjusted_config, w, seed=seed + idx)
        collected_solutions.append(run_res["best"])
        total_evals += run_res["evaluations"]

    # Фильтруем недоминируемые решения из всех найденных WSGA
    non_dominated = []
    for i, p in enumerate(collected_solutions):
        p_dom = False
        for j, q in enumerate(collected_solutions):
            if i != j and dominates(q["objectives"], p["objectives"]):
                p_dom = True
                break
        if not p_dom:
            non_dominated.append(p)

    # Убираем дубликаты по критериям
    unique_front = []
    seen = set()
    for ind in non_dominated:
        key = (round(ind["cost"], 1), round(ind["nutrient_error"], 2), round(ind["monotony"], 2))
        if key not in seen:
            seen.add(key)
            unique_front.append(ind)

    pts = [ind["objectives"] for ind in unique_front]
    ref_pt = config["reference_point"]
    hv = calculate_hypervolume(pts, ref_pt)
    sp = calculate_spacing(unique_front)

    return {
        "pareto_front": unique_front,
        "all_solutions": collected_solutions,
        "evaluations": total_evals,
        "hypervolume": hv,
        "spacing": sp,
    }

