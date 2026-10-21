"""Многокритериальная эволюционная оптимизация (NSGA-II) для задачи

формирования недельного меню (Вариант 6).

Критерии оптимизации:
  f1: Стоимость меню на неделю (руб.) -> min
  f2: Среднесуточное отклонение от целевых нутриентов (%) -> min
  f3: Коэффициент монотонности меню (100% - Разнообразие%) -> min

Реализовано с нуля без использования сторонних фреймворков (DEAP, pymoo).
Используется только стандартная библиотека Python.
"""

import argparse
import copy
import json
import math
import random
from pathlib import Path


def load_data(filepath="data/dishes.json"):
    """Загружает базу блюд и индексирует их по категориям."""
    path = Path(filepath)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    dishes = {d["id"]: d for d in data["dishes"]}
    by_category = {
        "breakfast": [d for d in data["dishes"] if d["category"] == "breakfast"],
        "lunch": [d for d in data["dishes"] if d["category"] == "lunch"],
        "dinner": [d for d in data["dishes"] if d["category"] == "dinner"],
    }
    return {
        "metadata": data["metadata"],
        "dishes": dishes,
        "by_category": by_category,
        "daily_targets": data["metadata"]["daily_targets"],
    }


def random_chromosome(data, rng):
    """Генерирует случайный генотип меню на 7 дней по 3 приема пищи (21 слот).

    Слоты строго упорядочены:
      0, 1, 2: Пн (завтрак, обед, ужин)
      3, 4, 5: Вт (завтрак, обед, ужин)
      ...
      18, 19, 20: Вс (завтрак, обед, ужин)
    """
    chromosome = []
    for _ in range(7):
        b = rng.choice(data["by_category"]["breakfast"])["id"]
        l = rng.choice(data["by_category"]["lunch"])["id"]
        d = rng.choice(data["by_category"]["dinner"])["id"]
        chromosome.extend([b, l, d])
    return chromosome


def evaluate(chromosome, data):
    """Вычисляет три целевые функции и дополнительные показатели решения.

    f1: Суммарная стоимость всех блюд за 7 дней (руб.) -> min
    f2: Среднесуточное относительное отклонение по БЖУ и ккал (%) -> min
    f3: Монотонность меню = (1.0 - Индекс_разнообразия) * 100% -> min
    """
    dishes = data["dishes"]
    targets = data["daily_targets"]

    # 1. Стоимость
    total_cost = sum(dishes[dish_id]["price"] for dish_id in chromosome)

    # 2. Нутриенты по дням
    daily_deviations = []
    day_details = []
    for d in range(7):
        b = dishes[chromosome[3 * d]]
        l = dishes[chromosome[3 * d + 1]]
        din = dishes[chromosome[3 * d + 2]]

        day_cals = b["calories"] + l["calories"] + din["calories"]
        day_prot = b["protein"] + l["protein"] + din["protein"]
        day_fat = b["fat"] + l["fat"] + din["fat"]
        day_carbs = b["carbs"] + l["carbs"] + din["carbs"]

        err_cal = abs(day_cals - targets["calories"]) / targets["calories"]
        err_prot = abs(day_prot - targets["protein"]) / targets["protein"]
        err_fat = abs(day_fat - targets["fat"]) / targets["fat"]
        err_carbs = abs(day_carbs - targets["carbs"]) / targets["carbs"]

        day_err = (err_cal + err_prot + err_fat + err_carbs) / 4.0
        daily_deviations.append(day_err)
        day_details.append({
            "day": d + 1,
            "calories": day_cals,
            "protein": day_prot,
            "fat": day_fat,
            "carbs": day_carbs,
            "mean_error_percent": day_err * 100.0,
        })

    avg_nutrient_error = (sum(daily_deviations) / 7.0) * 100.0

    # 3. Разнообразие (уникальные блюда, отсутствие повторов подряд, разнообразие источников белка)
    unique_dishes = len(set(chromosome))
    consecutive_repeats = 0
    for d in range(6):
        for m in range(3):
            if chromosome[3 * d + m] == chromosome[3 * (d + 1) + m]:
                consecutive_repeats += 1

    protein_sources = set(dishes[dish_id]["protein_source"] for dish_id in chromosome)
    cuisines = set(dishes[dish_id]["cuisine"] for dish_id in chromosome)

    # Нормированный индекс разнообразия D in [0, 1]
    # max unique is 21, min is 3
    norm_unique = (unique_dishes - 3) / 18.0
    norm_no_consec = 1.0 - min(1.0, consecutive_repeats / 6.0)
    norm_sources = min(1.0, len(protein_sources) / 6.0)

    diversity_score = 0.50 * norm_unique + 0.30 * norm_no_consec + 0.20 * norm_sources
    monotony = (1.0 - diversity_score) * 100.0

    objectives = (float(total_cost), float(avg_nutrient_error), float(monotony))

    return {
        "chromosome": list(chromosome),
        "objectives": objectives,
        "cost": float(total_cost),
        "nutrient_error": float(avg_nutrient_error),
        "monotony": float(monotony),
        "diversity_score": float(diversity_score),
        "unique_dishes": unique_dishes,
        "consecutive_repeats": consecutive_repeats,
        "unique_protein_sources": len(protein_sources),
        "unique_cuisines": len(cuisines),
        "day_details": day_details,
        "rank": 0,
        "crowding_distance": 0.0,
    }


def dominates(p_objs, q_objs):
    """Проверяет отношение Парето-доминирования для минимизации: p доминирует q."""
    strictly_better = False
    for p_val, q_val in zip(p_objs, q_objs):
        if p_val > q_val:
            return False
        if p_val < q_val:
            strictly_better = True
    return strictly_better


def fast_non_dominated_sort(population):
    """Быстрая недоминируемая сортировка по К. Дебу O(M * N^2).

    Возвращает список фронтов: [F_1, F_2, ...].
    """
    n = len(population)
    dominated_by = [0] * n
    dominates_list = [[] for _ in range(n)]
    fronts = [[]]

    for p in range(n):
        p_objs = population[p]["objectives"]
        for q in range(p + 1, n):
            q_objs = population[q]["objectives"]
            if dominates(p_objs, q_objs):
                dominates_list[p].append(q)
                dominated_by[q] += 1
            elif dominates(q_objs, p_objs):
                dominates_list[q].append(p)
                dominated_by[p] += 1

        if dominated_by[p] == 0:
            population[p]["rank"] = 1
            fronts[0].append(p)

    current_rank = 1
    while fronts[current_rank - 1]:
        next_front = []
        for p in fronts[current_rank - 1]:
            for q in dominates_list[p]:
                dominated_by[q] -= 1
                if dominated_by[q] == 0:
                    population[q]["rank"] = current_rank + 1
                    next_front.append(q)
        current_rank += 1
        if next_front:
            fronts.append(next_front)
        else:
            break

    # Возвращаем список фронтов как списки самих особей
    result_fronts = []
    for f in fronts:
        result_fronts.append([population[idx] for idx in f])
    return result_fronts


def calculate_crowding_distance(front):
    """Вычисляет расстояние до скопления (crowding distance) для особей фронта."""
    size = len(front)
    if size == 0:
        return
    for ind in front:
        ind["crowding_distance"] = 0.0

    if size <= 2:
        for ind in front:
            ind["crowding_distance"] = float("inf")
        return

    num_objectives = len(front[0]["objectives"])

    for m in range(num_objectives):
        # Сортируем фронт по m-му критерию
        front.sort(key=lambda ind: ind["objectives"][m])

        front[0]["crowding_distance"] = float("inf")
        front[-1]["crowding_distance"] = float("inf")

        f_min = front[0]["objectives"][m]
        f_max = front[-1]["objectives"][m]
        f_range = f_max - f_min

        if f_range <= 1e-9:
            continue

        for i in range(1, size - 1):
            if front[i]["crowding_distance"] != float("inf"):
                distance = (front[i + 1]["objectives"][m] - front[i - 1]["objectives"][m]) / f_range
                front[i]["crowding_distance"] += distance


def crowded_comparison(a, b):
    """Оператор сравнения по скоплению (<_n): -1 если a лучше b, 1 если b лучше a, 0 если равны."""
    if a["rank"] < b["rank"]:
        return -1
    if a["rank"] > b["rank"]:
        return 1
    # Ранги равны: предпочитаем бОльшее расстояние до скопления
    if a["crowding_distance"] > b["crowding_distance"]:
        return -1
    if a["crowding_distance"] < b["crowding_distance"]:
        return 1
    return 0


def tournament_selection(population, k, rng):
    """Турнирная селекция размера k с использованием crowded-comparison."""
    candidates = rng.sample(population, k)
    best = candidates[0]
    for other in candidates[1:]:
        if crowded_comparison(other, best) < 0:
            best = other
    return best


def crossover(parent1, parent2, rng, p_cross=0.9):
    """Кроссовер для недельного меню с сохранением категорий блюд в слотах.

    Используется либо однородный кроссовер по слотам, либо двухточечный по дням.
    """
    if rng.random() > p_cross:
        return copy.deepcopy(parent1["chromosome"]), copy.deepcopy(parent2["chromosome"])

    c1 = list(parent1["chromosome"])
    c2 = list(parent2["chromosome"])

    if rng.random() < 0.5:
        # Однородный кроссовер: для каждого слота с равной вероятностью берем от p1 или p2
        for i in range(21):
            if rng.random() < 0.5:
                c1[i], c2[i] = c2[i], c1[i]
    else:
        # Двухточечный обмен целыми суточными блоками (днями недели)
        day1 = rng.randint(0, 6)
        day2 = rng.randint(0, 6)
        if day1 > day2:
            day1, day2 = day2, day1
        idx1 = day1 * 3
        idx2 = (day2 + 1) * 3
        c1[idx1:idx2], c2[idx1:idx2] = c2[idx1:idx2], c1[idx1:idx2]

    return c1, c2


def mutate(chromosome, data, rng, p_mut=0.3):
    """Специализированная мутация трех типов:

    1. Точечная замена блюда на случайное той же категории.
    2. Перестановка (swap) блюд одной категории между двумя днями.
    3. Разрушение монотонности: замена повторяющегося блюда на ранее не использованное.
    """
    mutated = list(chromosome)
    dishes_by_cat = data["by_category"]

    cat_names = ["breakfast", "lunch", "dinner"]

    if rng.random() < p_mut:
        mutation_type = rng.random()
        if mutation_type < 0.45:
            # Случайная смена 1-2 позиций
            num_slots = rng.randint(1, 2)
            for _ in range(num_slots):
                slot = rng.randint(0, 20)
                cat = cat_names[slot % 3]
                options = dishes_by_cat[cat]
                new_dish = rng.choice(options)["id"]
                mutated[slot] = new_dish

        elif mutation_type < 0.75:
            # Перестановка блюд одного типа между двумя днями
            m = rng.randint(0, 2)  # завтрак, обед или ужин
            d1, d2 = rng.sample(range(7), 2)
            mutated[3 * d1 + m], mutated[3 * d2 + m] = mutated[3 * d2 + m], mutated[3 * d1 + m]

        else:
            # Находим дубликат и заменяем его на отсутствующее в меню блюдо
            seen = set()
            dup_slot = None
            for slot_idx, d_id in enumerate(mutated):
                if d_id in seen:
                    dup_slot = slot_idx
                    break
                seen.add(d_id)

            if dup_slot is not None:
                cat = cat_names[dup_slot % 3]
                all_cat_ids = [d["id"] for d in dishes_by_cat[cat]]
                unused = [i for i in all_cat_ids if i not in seen]
                if unused:
                    mutated[dup_slot] = rng.choice(unused)
                else:
                    mutated[dup_slot] = rng.choice(all_cat_ids)

    return mutated


def hypervolume_2d(points_2d, ref_2d):
    """Точный расчет 2D гипер-объема (площади)."""
    pts = sorted(points_2d, key=lambda p: (p[0], p[1]))
    clean = []
    min_z = float("inf")
    for y, z in pts:
        if z < min_z:
            clean.append((y, z))
            min_z = z
    area = 0.0
    for i in range(len(clean)):
        y_curr, z_curr = clean[i]
        next_y = clean[i + 1][0] if i + 1 < len(clean) else ref_2d[0]
        width = max(0.0, next_y - y_curr)
        height = max(0.0, ref_2d[1] - z_curr)
        area += width * height
    return area


def calculate_hypervolume(points, reference_point):
    """Точный расчет 3D Hypervolume методом разложения на 2D сечения."""
    ref = (reference_point["cost"], reference_point["nutrient_deviation"], reference_point["monotony"])
    valid = [p for p in points if p[0] <= ref[0] and p[1] <= ref[1] and p[2] <= ref[2]]
    if not valid:
        return 0.0

    unique_x = sorted(list(set(p[0] for p in valid)))
    total_vol = 0.0
    for i in range(len(unique_x)):
        x_curr = unique_x[i]
        next_x = unique_x[i + 1] if i + 1 < len(unique_x) else ref[0]
        slice_width = next_x - x_curr
        if slice_width <= 0:
            continue
        active = [(p[1], p[2]) for p in valid if p[0] <= x_curr]
        area = hypervolume_2d(active, (ref[1], ref[2]))
        total_vol += slice_width * area

    # Нормированный гиперобъем относительно максимального бокса
    max_box = ref[0] * ref[1] * ref[2]
    norm_hv = total_vol / max_box if max_box > 0 else 0.0
    return norm_hv


def calculate_spacing(front):
    """Метрика равномерности распределения решений вдоль фронта Парето (Spacing / Spread)."""
    if len(front) <= 1:
        return 0.0

    points = [ind["objectives"] for ind in front]
    dists = []
    for i, p in enumerate(points):
        min_d = float("inf")
        for j, q in enumerate(points):
            if i != j:
                # Манхэттенское расстояние в нормализованных координатах
                d = sum(abs(p[m] - q[m]) for m in range(len(p)))
                if d < min_d:
                    min_d = d
        dists.append(min_d)

    d_mean = sum(dists) / len(dists)
    var = sum((d - d_mean) ** 2 for d in dists) / len(dists)
    return math.sqrt(var)


def run_nsga2(data, config, seed=3000, track_history=True, verbose=False):
    """Основной запуск алгоритма NSGA-II."""
    rng = random.Random(seed)
    pop_size = config["population_size"]
    generations = config["generations"]
    p_cross = config["crossover_probability"]
    p_mut = config["mutation_probability"]
    tourn_size = config.get("tournament_size", 2)
    ref_pt = config["reference_point"]

    # Инициализация популяции
    population = []
    for _ in range(pop_size):
        chrom = random_chromosome(data, rng)
        ind = evaluate(chrom, data)
        population.append(ind)

    # Недоминируемая сортировка начальной популяции
    fronts = fast_non_dominated_sort(population)
    for f in fronts:
        calculate_crowding_distance(f)

    # Внешний архив лучших недоминируемых решений за всю эволюцию
    archive = copy.deepcopy(fronts[0])

    history = []
    evaluations = pop_size

    for gen in range(generations):
        # 1. Генерация потомков Q_t размера pop_size
        offspring = []
        while len(offspring) < pop_size:
            p1 = tournament_selection(population, tourn_size, rng)
            p2 = tournament_selection(population, tourn_size, rng)
            c1_chrom, c2_chrom = crossover(p1, p2, rng, p_cross)
            c1_chrom = mutate(c1_chrom, data, rng, p_mut)
            c2_chrom = mutate(c2_chrom, data, rng, p_mut)
            offspring.append(evaluate(c1_chrom, data))
            evaluations += 1
            if len(offspring) < pop_size:
                offspring.append(evaluate(c2_chrom, data))
                evaluations += 1

        # 2. Объединение R_t = P_t U Q_t
        combined = population + offspring

        # 3. Недоминируемая сортировка объединенной популяции
        all_fronts = fast_non_dominated_sort(combined)

        # 4. Формирование P_{t+1}
        new_population = []
        for front in all_fronts:
            calculate_crowding_distance(front)
            if len(new_population) + len(front) <= pop_size:
                new_population.extend(front)
            else:
                # Сортируем последний фронт по crowding_distance убывающе
                remaining = pop_size - len(new_population)
                front.sort(key=lambda x: x["crowding_distance"], reverse=True)
                new_population.extend(front[:remaining])
                break

        population = new_population

        # Обновление внешнего архива Парето-решений
        combined_archive = archive + [population[0]] + all_fronts[0]
        arch_fronts = fast_non_dominated_sort(combined_archive)
        archive = arch_fronts[0]

        # Ограничиваем размер архива при необходимости (сохраняя максимальное разнообразие)
        if len(archive) > pop_size * 2:
            calculate_crowding_distance(archive)
            archive.sort(key=lambda x: x["crowding_distance"], reverse=True)
            archive = archive[: pop_size * 2]

        if track_history:
            current_pf = all_fronts[0]
            pts = [ind["objectives"] for ind in current_pf]
            hv = calculate_hypervolume(pts, ref_pt)
            best_cost = min(p[0] for p in pts)
            best_nutr = min(p[1] for p in pts)
            best_monotony = min(p[2] for p in pts)
            history.append({
                "generation": gen + 1,
                "front_size": len(current_pf),
                "hypervolume": hv,
                "best_cost": best_cost,
                "best_nutrient_error": best_nutr,
                "best_monotony": best_monotony,
            })

        if verbose and (gen + 1) % 25 == 0:
            print(f"Поколение {gen+1:3d} | Фронт F1: {len(all_fronts[0]):2d} | HV: {history[-1]['hypervolume']:.4f}")

    # Финальная сортировка и расчет расстояния
    final_fronts = fast_non_dominated_sort(population)
    for f in final_fronts:
        calculate_crowding_distance(f)
    pareto_front = final_fronts[0]

    # Итоговый архив уникальных недоминируемых решений
    final_archive_fronts = fast_non_dominated_sort(archive)
    clean_archive = final_archive_fronts[0]

    pts = [ind["objectives"] for ind in pareto_front]
    final_hv = calculate_hypervolume(pts, ref_pt)
    spacing = calculate_spacing(pareto_front)

    return {
        "pareto_front": pareto_front,
        "archive": clean_archive,
        "population": population,
        "history": history,
        "evaluations": evaluations,
        "hypervolume": final_hv,
        "spacing": spacing,
        "seed": seed,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="NSGA-II для формирования недельного меню (Вариант 6)")
    parser.add_argument("--seed", type=int, default=3000, help="Начальное случайное зерно (seed)")
    parser.add_argument("--pop", type=int, default=100, help="Размер популяции")
    parser.add_argument("--gen", type=int, default=150, help="Количество поколений")
    parser.add_argument("--data", type=str, default="data/dishes.json", help="Путь к файлу блюд")
    parser.add_argument("--config", type=str, default="config.json", help="Путь к файлу конфигурации")
    parser.add_argument("--verbose", action="store_true", help="Подробный вывод процесса")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data = load_data(args.data)
    with Path(args.config).open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["population_size"] = args.pop
    cfg["generations"] = args.gen

    print(f"Запуск NSGA-II: seed={args.seed}, pop={args.pop}, gen={args.gen}...")
    res = run_nsga2(data, cfg, seed=args.seed, verbose=args.verbose)
    pf = res["pareto_front"]
    print(f"Успешно завершено! Найдено решений на фронте Парето: {len(pf)}")
    print(f"Итоговый нормированный Hypervolume: {res['hypervolume']:.5f}")
    print(f"Равномерность фронта (Spacing): {res['spacing']:.2f}")

    # Вывод репрезентативных решений
    costs = [x["cost"] for x in pf]
    nutrs = [x["nutrient_error"] for x in pf]
    monots = [x["monotony"] for x in pf]
    print(f"Диапазон стоимости: {min(costs):.0f} - {max(costs):.0f} руб.")
    print(f"Диапазон отклонения нутриентов: {min(nutrs):.1f}% - {max(nutrs):.1f}%")
    print(f"Диапазон монотонности: {min(monots):.1f}% - {max(monots):.1f}% (разнообразие {100-max(monots):.1f}% - {100-min(monots):.1f}%)")

