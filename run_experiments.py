"""Серия экспериментов и генерация результатов для лабораторной работы № 3.

Вариант 6: Формирование меню на неделю
Трек A: Многокритериальная эволюционная оптимизация (NSGA-II)

Проводит:
  1. 20 независимых запусков NSGA-II (seeds 3000..3019).
  2. Серию взвешенной суммы критериев (Weighted Sum GA) по сетке весов.
  3. Случайный допустимый поиск (Random Search) при равном бюджете вычислений.
  4. Расчет статистических метрик: Hypervolume, Spacing, размер фронта, время.
  5. Генерацию CSV-отчетов, детального меню репрезентативных решений.
  6. Построение качественных SVG-диаграмм и интерактивного 3D HTML визуализатора.

Все графики строятся стандартными средствами Python без внешних зависимостей.
"""

import copy
import csv
import json
import math
import random
import statistics
import time
from pathlib import Path

from nsga2_menu import (
    calculate_hypervolume,
    calculate_spacing,
    dominates,
    evaluate,
    fast_non_dominated_sort,
    load_data,
    random_chromosome,
    run_nsga2,
)
from weighted_sum_ga import run_weighted_sum_experiment


def run_random_search(data, config, seed, budget):
    """Случайный допустимый поиск при эквивалентном бюджете вычислений."""
    rng = random.Random(seed)
    ref_pt = config["reference_point"]

    samples = []
    for _ in range(budget):
        chrom = random_chromosome(data, rng)
        samples.append(evaluate(chrom, data))

    # Выделяем фронт недоминируемых решений
    fronts = fast_non_dominated_sort(samples)
    pf = fronts[0]

    # Убираем дубликаты
    unique_pf = []
    seen = set()
    for ind in pf:
        key = (round(ind["cost"], 1), round(ind["nutrient_error"], 2), round(ind["monotony"], 2))
        if key not in seen:
            seen.add(key)
            unique_pf.append(ind)

    pts = [p["objectives"] for p in unique_pf]
    hv = calculate_hypervolume(pts, ref_pt)
    sp = calculate_spacing(unique_pf)

    return {
        "pareto_front": unique_pf,
        "evaluations": budget,
        "hypervolume": hv,
        "spacing": sp,
    }


def write_csv(path, rows, fieldnames):
    """Вспомогательная функция записи CSV."""
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def extract_representative_solutions(pareto_front, data):
    """Выделяет 4 характерных компромиссных решения из Парето-фронта:

    1. «Бюджетное»: минимальная стоимость меню при допустимом качестве.
    2. «Сбалансированное»: минимальное отклонение от суточных норм нутриентов.
    3. «Гастрономическое разнообразие»: максимальное разнообразие (монотонность 0%).
    4. «Компромиссное (Knee Point)»: сбалансированное решение в точке перегиба.
    """
    dishes = data["dishes"]

    # 1. Бюджетное (min cost)
    cheapest = min(pareto_front, key=lambda x: x["cost"])

    # 2. Сбалансированное по нутриентам (min nutrient error)
    most_balanced = min(pareto_front, key=lambda x: x["nutrient_error"])

    # 3. Максимально разнообразное (min monotony, max diversity score)
    # Среди с минимальной монотонностью выбираем с меньшей стоимостью
    zero_monotony = [x for x in pareto_front if x["monotony"] <= 0.01]
    if zero_monotony:
        most_diverse = min(zero_monotony, key=lambda x: (x["nutrient_error"], x["cost"]))
    else:
        most_diverse = min(pareto_front, key=lambda x: x["monotony"])

    # 4. Компромиссное решение (Knee Point): минимизирует нормированное евклидово расстояние до утопической точки (0,0,0)
    costs = [x["cost"] for x in pareto_front]
    nutrs = [x["nutrient_error"] for x in pareto_front]
    monots = [x["monotony"] for x in pareto_front]

    c_min, c_max = min(costs), max(costs)
    n_min, n_max = min(nutrs), max(nutrs)
    m_min, m_max = min(monots), max(monots)

    def dist_to_utopia(x):
        nc = (x["cost"] - c_min) / max(1.0, c_max - c_min)
        nn = (x["nutrient_error"] - n_min) / max(1.0, n_max - n_min)
        nm = (x["monotony"] - m_min) / max(1.0, m_max - m_min)
        return math.sqrt(nc**2 + nn**2 + nm**2)

    knee = min(pareto_front, key=dist_to_utopia)

    representative = [
        {"type": "Бюджетное", "code": "cheapest", "ind": cheapest},
        {"type": "Сбалансированное по нутриентам", "code": "nutritional", "ind": most_balanced},
        {"type": "Максимальное разнообразие (Гурман)", "code": "diverse", "ind": most_diverse},
        {"type": "Компромиссное (Knee Point)", "code": "compromise", "ind": knee},
    ]

    day_names = ["Понедельник", "Вторник", "Среда", "Четверг", "Пятница", "Суббота", "Воскресенье"]
    meal_names = ["Завтрак", "Обед", "Ужин"]

    csv_rows = []
    json_data = []

    for item in representative:
        sol_type = item["type"]
        ind = item["ind"]
        chrom = ind["chromosome"]

        menu_days = []
        for d in range(7):
            b_dish = dishes[chrom[3 * d]]
            l_dish = dishes[chrom[3 * d + 1]]
            d_dish = dishes[chrom[3 * d + 2]]

            daily_dishes = [b_dish, l_dish, d_dish]
            d_cal = sum(x["calories"] for x in daily_dishes)
            d_prot = sum(x["protein"] for x in daily_dishes)
            d_fat = sum(x["fat"] for x in daily_dishes)
            d_carb = sum(x["carbs"] for x in daily_dishes)
            d_cost = sum(x["price"] for x in daily_dishes)

            day_info = {
                "day_number": d + 1,
                "day_name": day_names[d],
                "breakfast": b_dish["name"],
                "lunch": l_dish["name"],
                "dinner": d_dish["name"],
                "daily_cost": d_cost,
                "calories": d_cal,
                "protein": d_prot,
                "fat": d_fat,
                "carbs": d_carb,
            }
            menu_days.append(day_info)

            for m_idx, dish in enumerate(daily_dishes):
                csv_rows.append({
                    "solution_type": sol_type,
                    "day_number": d + 1,
                    "day_name": day_names[d],
                    "meal_type": meal_names[m_idx],
                    "dish_name": dish["name"],
                    "category": dish["category"],
                    "price_rub": dish["price"],
                    "calories_kcal": dish["calories"],
                    "protein_g": dish["protein"],
                    "fat_g": dish["fat"],
                    "carbs_g": dish["carbs"],
                    "cuisine": dish["cuisine"],
                    "protein_source": dish["protein_source"],
                    "menu_total_cost": round(ind["cost"], 2),
                    "menu_nutrient_error_pct": round(ind["nutrient_error"], 2),
                    "menu_monotony_pct": round(ind["monotony"], 2),
                    "menu_diversity_score": round(ind["diversity_score"], 3),
                    "menu_unique_dishes": ind["unique_dishes"],
                })

        json_data.append({
            "type": sol_type,
            "code": item["code"],
            "total_cost": ind["cost"],
            "nutrient_error_pct": ind["nutrient_error"],
            "monotony_pct": ind["monotony"],
            "diversity_score": ind["diversity_score"],
            "unique_dishes": ind["unique_dishes"],
            "consecutive_repeats": ind["consecutive_repeats"],
            "days": menu_days,
        })

    return csv_rows, json_data, representative


# ----------------------------------------------------------------------
# SVG ГЕНЕРАТОРЫ (Чистый Python, без сторонних библиотек)
# ----------------------------------------------------------------------

def write_pareto_projections_svg(filepath, nsga_pf, wsga_pf, rand_pf, rep_solutions):
    """Создает двухмерные проекции трехмерного фронта Парето:

    1. Стоимость (руб.) vs Ошибка нутриентов (%)
    2. Стоимость (руб.) vs Монотонность (%)
    3. Ошибка нутриентов (%) vs Монотонность (%)
    """
    w, h = 1200, 480
    panel_w = 340
    panel_h = 340
    margin_top = 65
    margin_left = 65
    gap = 60

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>',
        'text { font-family: "Segoe UI", Arial, sans-serif; font-size: 11px; fill: #2c3e50; }',
        '.title { font-size: 16px; font-weight: bold; fill: #1a252f; text-anchor: middle; }',
        '.subtitle { font-size: 12px; fill: #7f8c8d; text-anchor: middle; }',
        '.panel-title { font-size: 13px; font-weight: bold; fill: #34495e; text-anchor: middle; }',
        '.axis { stroke: #2c3e50; stroke-width: 1.2; }',
        '.grid { stroke: #ecf0f1; stroke-width: 1; stroke-dasharray: 2,2; }',
        '.legend-text { font-size: 11px; font-weight: 500; }',
        '</style>',
        f'<text class="title" x="{w/2}" y="28">Двумерные проекции Парето-фронта для задачи формирования меню на неделю</text>',
        f'<text class="subtitle" x="{w/2}" y="46">Сравнение NSGA-II, взвешенной суммы критериев (WSGA) и случайного поиска с выделением характерных решений</text>',
    ]

    projections = [
        {
            "title": "Стоимость vs Отклонение нутриентов",
            "x_idx": 0, "x_label": "Стоимость (руб.)", "x_min": 3200, "x_max": 6500,
            "y_idx": 1, "y_label": "Отклонение нутриентов (%)", "y_min": 10, "y_max": 45,
        },
        {
            "title": "Стоимость vs Монотонность",
            "x_idx": 0, "x_label": "Стоимость (руб.)", "x_min": 3200, "x_max": 6500,
            "y_idx": 2, "y_label": "Монотонность меню (%)", "y_min": 0, "y_max": 90,
        },
        {
            "title": "Отклонение нутриентов vs Монотонность",
            "x_idx": 1, "x_label": "Отклонение нутриентов (%)", "x_min": 10, "x_max": 45,
            "y_idx": 2, "y_label": "Монотонность меню (%)", "y_min": 0, "y_max": 90,
        },
    ]

    for p_idx, proj in enumerate(projections):
        ox = margin_left + p_idx * (panel_w + gap)
        oy = margin_top

        # Заголовок панели
        parts.append(f'<text class="panel-title" x="{ox + panel_w/2}" y="{oy - 10}">{proj["title"]}</text>')

        # Фон графика
        parts.append(f'<rect x="{ox}" y="{oy}" width="{panel_w}" height="{panel_h}" fill="#fafbfc" stroke="#e1e4e8"/>')

        x_min, x_max = proj["x_min"], proj["x_max"]
        y_min, y_max = proj["y_min"], proj["y_max"]

        def to_screen(val_x, val_y):
            sx = ox + (val_x - x_min) / (x_max - x_min) * panel_w
            sy = oy + panel_h - (val_y - y_min) / (y_max - y_min) * panel_h
            return sx, sy

        # Сетка и метки X
        for step in range(5):
            gx_val = x_min + (x_max - x_min) * step / 4
            gx, _ = to_screen(gx_val, y_min)
            parts.append(f'<line class="grid" x1="{gx}" y1="{oy}" x2="{gx}" y2="{oy + panel_h}"/>')
            parts.append(f'<text x="{gx}" y="{oy + panel_h + 16}" text-anchor="middle">{gx_val:.0f}</text>')

        # Сетка и метки Y
        for step in range(5):
            gy_val = y_min + (y_max - y_min) * step / 4
            _, gy = to_screen(x_min, gy_val)
            parts.append(f'<line class="grid" x1="{ox}" y1="{gy}" x2="{ox + panel_w}" y2="{gy}"/>')
            parts.append(f'<text x="{ox - 8}" y="{gy + 4}" text-anchor="end">{gy_val:.0f}</text>')

        # Оси
        parts.append(f'<line class="axis" x1="{ox}" y1="{oy + panel_h}" x2="{ox + panel_w}" y2="{oy + panel_h}"/>')
        parts.append(f'<line class="axis" x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy + panel_h}"/>')
        parts.append(f'<text x="{ox + panel_w/2}" y="{oy + panel_h + 34}" text-anchor="middle">{proj["x_label"]}</text>')

        # Подпись Y
        parts.append(f'<text transform="rotate(-90 {ox - 35} {oy + panel_h/2})" x="{ox - 35}" y="{oy + panel_h/2}" text-anchor="middle">{proj["y_label"]}</text>')

        # 1. Точки Random Search
        for ind in rand_pf:
            sx, sy = to_screen(ind["objectives"][proj["x_idx"]], ind["objectives"][proj["y_idx"]])
            if ox <= sx <= ox + panel_w and oy <= sy <= oy + panel_h:
                parts.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="2.5" fill="#bdc3c7" opacity="0.6"/>')

        # 2. Точки Weighted Sum GA
        for ind in wsga_pf:
            sx, sy = to_screen(ind["objectives"][proj["x_idx"]], ind["objectives"][proj["y_idx"]])
            if ox <= sx <= ox + panel_w and oy <= sy <= oy + panel_h:
                parts.append(f'<polygon points="{sx},{sy-4} {sx+4},{sy} {sx},{sy+4} {sx-4},{sy}" fill="#e67e22" stroke="#d35400" stroke-width="0.8"/>')

        # 3. Точки NSGA-II
        for ind in nsga_pf:
            sx, sy = to_screen(ind["objectives"][proj["x_idx"]], ind["objectives"][proj["y_idx"]])
            if ox <= sx <= ox + panel_w and oy <= sy <= oy + panel_h:
                parts.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="3.2" fill="#2980b9" stroke="#1b4f72" stroke-width="0.8" opacity="0.85"/>')

        # 4. Репрезентативные точки
        colors = {"cheapest": "#27ae60", "nutritional": "#8e44ad", "diverse": "#c0392b", "compromise": "#d35400"}
        for rep in rep_solutions:
            ind = rep["ind"]
            code = rep["code"]
            color = colors.get(code, "#e74c3c")
            sx, sy = to_screen(ind["objectives"][proj["x_idx"]], ind["objectives"][proj["y_idx"]])
            if ox <= sx <= ox + panel_w and oy <= sy <= oy + panel_h:
                parts.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="6.0" fill="{color}" stroke="#ffffff" stroke-width="1.8"/>')

    # Легенда внизу
    leg_y = h - 18
    parts.append(f'<g transform="translate({margin_left}, {leg_y})">')
    parts.append('<circle cx="10" cy="0" r="3.5" fill="#2980b9" stroke="#1b4f72"/>')
    parts.append('<text class="legend-text" x="20" y="3.5">NSGA-II (Парето-фронт)</text>')

    parts.append('<polygon points="175,-4 179,0 175,4 171,0" fill="#e67e22" stroke="#d35400"/>')
    parts.append('<text class="legend-text" x="187" y="3.5">Взвешенная сумма (WSGA)</text>')

    parts.append('<circle cx="370" cy="0" r="3" fill="#bdc3c7"/>')
    parts.append('<text class="legend-text" x="380" y="3.5">Случайный поиск (Random Search)</text>')

    parts.append('<circle cx="590" cy="0" r="5" fill="#27ae60" stroke="#fff"/>')
    parts.append('<text class="legend-text" x="600" y="3.5">Бюджетное</text>')

    parts.append('<circle cx="690" cy="0" r="5" fill="#8e44ad" stroke="#fff"/>')
    parts.append('<text class="legend-text" x="700" y="3.5">Сбалансированное</text>')

    parts.append('<circle cx="820" cy="0" r="5" fill="#c0392b" stroke="#fff"/>')
    parts.append('<text class="legend-text" x="830" y="3.5">Разнообразное (Гурман)</text>')

    parts.append('<circle cx="980" cy="0" r="5" fill="#d35400" stroke="#fff"/>')
    parts.append('<text class="legend-text" x="990" y="3.5">Компромисс (Knee Point)</text>')
    parts.append('</g>')

    parts.append('</svg>')
    Path(filepath).write_text("\n".join(parts), encoding="utf-8")


def write_convergence_svg(filepath, all_nsga_histories):
    """Строит график динамики нормализованного Hypervolume по поколениям."""
    w, h = 880, 440
    left, right, top, bottom = 80, 40, 50, 60
    plot_w = w - left - right
    plot_h = h - top - bottom

    num_gens = len(all_nsga_histories[0])
    hv_matrix = [[h_step["hypervolume"] for h_step in run_h] for run_h in all_nsga_histories]

    mean_hvs = [statistics.mean(values) for values in zip(*hv_matrix)]
    min_hvs = [min(values) for values in zip(*hv_matrix)]
    max_hvs = [max(values) for values in zip(*hv_matrix)]

    y_min = math.floor(min(min_hvs) * 20) / 20.0
    y_max = math.ceil(max(max_hvs) * 20) / 20.0
    if y_max - y_min < 0.1:
        y_max = y_min + 0.1

    def to_screen(gen_idx, hv_val):
        sx = left + (gen_idx / (num_gens - 1)) * plot_w
        sy = top + plot_h - ((hv_val - y_min) / (y_max - y_min)) * plot_h
        return sx, sy

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>',
        'text { font-family: "Segoe UI", Arial, sans-serif; font-size: 12px; fill: #2c3e50; }',
        '.title { font-size: 15px; font-weight: bold; fill: #1a252f; text-anchor: middle; }',
        '.subtitle { font-size: 11px; fill: #7f8c8d; text-anchor: middle; }',
        '.axis { stroke: #2c3e50; stroke-width: 1.2; }',
        '.grid { stroke: #ecf0f1; stroke-width: 1; stroke-dasharray: 2,2; }',
        '</style>',
        f'<text class="title" x="{w/2}" y="24">Сходимость алгоритма NSGA-II по метрике Hypervolume (20 независимых запусков)</text>',
        f'<text class="subtitle" x="{w/2}" y="40">Отображены среднее значение (синяя линия) и коридор мин./макс. значений по поколениям</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fcfdfe" stroke="#e1e4e8"/>',
    ]

    # Сетка X (поколения)
    for g in [1, 25, 50, 75, 100, 125, 150]:
        if g <= num_gens:
            gx, _ = to_screen(g - 1, y_min)
            parts.append(f'<line class="grid" x1="{gx}" y1="{top}" x2="{gx}" y2="{top + plot_h}"/>')
            parts.append(f'<text x="{gx}" y="{top + plot_h + 18}" text-anchor="middle">{g}</text>')

    # Сетка Y (HV)
    for step in range(6):
        y_val = y_min + (y_max - y_min) * step / 5
        _, gy = to_screen(0, y_val)
        parts.append(f'<line class="grid" x1="{left}" y1="{gy}" x2="{left + plot_w}" y2="{gy}"/>')
        parts.append(f'<text x="{left - 8}" y="{gy + 4}" text-anchor="end">{y_val:.3f}</text>')

    # Коридор min-max (полигон)
    poly_points = []
    for g_idx in range(num_gens):
        sx, sy = to_screen(g_idx, max_hvs[g_idx])
        poly_points.append(f"{sx:.1f},{sy:.1f}")
    for g_idx in reversed(range(num_gens)):
        sx, sy = to_screen(g_idx, min_hvs[g_idx])
        poly_points.append(f"{sx:.1f},{sy:.1f}")

    parts.append(f'<polygon points="{" ".join(poly_points)}" fill="#d4e6f1" opacity="0.6"/>')

    # Средняя траектория
    path_points = []
    for g_idx in range(num_gens):
        sx, sy = to_screen(g_idx, mean_hvs[g_idx])
        path_points.append(f"{sx:.1f},{sy:.1f}")
    parts.append(f'<polyline points="{" ".join(path_points)}" fill="none" stroke="#1f618d" stroke-width="2.5"/>')

    # Оси
    parts.append(f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>')
    parts.append(f'<text x="{left + plot_w/2}" y="{top + plot_h + 40}" text-anchor="middle">Номер поколения (Generation)</text>')
    parts.append(f'<text transform="rotate(-90 {left - 48} {top + plot_h/2})" x="{left - 48}" y="{top + plot_h/2}" text-anchor="middle">Нормированный Hypervolume</text>')

    # Легенда
    parts.append(f'<rect x="{left + plot_w - 240}" y="{top + plot_h - 60}" width="230" height="50" fill="#ffffff" stroke="#bdc3c7" rx="3"/>')
    parts.append(f'<line x1="{left + plot_w - 225}" y1="{top + plot_h - 45}" x2="{left + plot_w - 195}" y2="{top + plot_h - 45}" stroke="#1f618d" stroke-width="2.5"/>')
    parts.append(f'<text x="{left + plot_w - 185}" y="{top + plot_h - 41}">Среднее HV ({mean_hvs[-1]:.4f})</text>')
    parts.append(f'<rect x="{left + plot_w - 225}" y="{top + plot_h - 28}" width="30" height="10" fill="#d4e6f1" stroke="#a9cce3"/>')
    parts.append(f'<text x="{left + plot_w - 185}" y="{top + plot_h - 19}">Коридор разброса min–max</text>')

    parts.append('</svg>')
    Path(filepath).write_text("\n".join(parts), encoding="utf-8")


def write_comparison_methods_svg(filepath, metrics):
    """Строит сравнительную столбчатую диаграмму эффективности трех подходов:

    NSGA-II, Weighted Sum GA, Random Search по метрикам Hypervolume, Spacing и размеру фронта.
    """
    w, h = 880, 400
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>',
        'text { font-family: "Segoe UI", Arial, sans-serif; font-size: 12px; fill: #2c3e50; }',
        '.title { font-size: 15px; font-weight: bold; fill: #1a252f; text-anchor: middle; }',
        '.panel-title { font-size: 13px; font-weight: bold; fill: #34495e; text-anchor: middle; }',
        '.bar-val { font-size: 11px; font-weight: bold; fill: #2c3e50; text-anchor: middle; }',
        '.axis { stroke: #2c3e50; stroke-width: 1.2; }',
        '.grid { stroke: #ecf0f1; stroke-width: 1; stroke-dasharray: 2,2; }',
        '</style>',
        f'<text class="title" x="{w/2}" y="25">Сравнение методов оптимизации недельного меню при равном бюджете вычислений</text>',
    ]

    panels = [
        {"title": "Hypervolume (больше — лучше)", "key": "hv", "y_max": 0.55, "fmt": "{:.3f}", "color": "#2980b9"},
        {"title": "Равномерность Spacing (меньше — лучше)", "key": "sp", "y_max": 140.0, "fmt": "{:.1f}", "color": "#e67e22"},
        {"title": "Число решений на фронте Парето", "key": "size", "y_max": 120.0, "fmt": "{:.0f}", "color": "#27ae60"},
    ]

    p_w, p_h = 240, 260
    margin_left = 60
    gap = 55
    top = 65

    method_names = ["NSGA-II", "WSGA", "Random"]
    colors = [["#2980b9", "#5499c7"], ["#e67e22", "#f39c12"], ["#95a5a6", "#bdc3c7"]]

    for p_idx, p in enumerate(panels):
        ox = margin_left + p_idx * (p_w + gap)
        oy = top

        parts.append(f'<text class="panel-title" x="{ox + p_w/2}" y="{oy - 10}">{p["title"]}</text>')
        parts.append(f'<rect x="{ox}" y="{oy}" width="{p_w}" height="{p_h}" fill="#fcfdfe" stroke="#e1e4e8"/>')

        vals = [metrics["nsga"][p["key"]], metrics["wsga"][p["key"]], metrics["rand"][p["key"]]]
        y_max = p["y_max"]

        # Линия сетки
        for step in range(5):
            g_val = y_max * step / 4
            gy = oy + p_h - (g_val / y_max) * p_h
            parts.append(f'<line class="grid" x1="{ox}" y1="{gy}" x2="{ox + p_w}" y2="{gy}"/>')
            parts.append(f'<text x="{ox - 6}" y="{gy + 4}" text-anchor="end" font-size="10">{p["fmt"].format(g_val)}</text>')

        # Столбцы
        bar_w = 46
        bar_gap = 24
        for m_idx, (val, name) in enumerate(zip(vals, method_names)):
            bx = ox + 22 + m_idx * (bar_w + bar_gap)
            bar_height = max(2.0, (val / y_max) * p_h)
            by = oy + p_h - bar_height

            c1, c2 = colors[m_idx]
            parts.append(f'<rect x="{bx}" y="{by}" width="{bar_w}" height="{bar_height}" fill="{c1}" rx="3"/>')
            parts.append(f'<text class="bar-val" x="{bx + bar_w/2}" y="{by - 6}">{p["fmt"].format(val)}</text>')
            parts.append(f'<text x="{bx + bar_w/2}" y="{oy + p_h + 18}" text-anchor="middle" font-weight="bold">{name}</text>')

        parts.append(f'<line class="axis" x1="{ox}" y1="{oy + p_h}" x2="{ox + p_w}" y2="{oy + p_h}"/>')
        parts.append(f'<line class="axis" x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy + p_h}"/>')

    parts.append('</svg>')
    Path(filepath).write_text("\n".join(parts), encoding="utf-8")


def write_algorithm_scheme_svg(filepath):
    """Генерирует блок-схему алгоритма NSGA-II."""
    w, h = 920, 520
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<defs>',
        '<marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">',
        '<path d="M 0 1 L 8 5 L 0 9 z" fill="#2c3e50"/>',
        '</marker>',
        '</defs>',
        '<style>',
        'text { font-family: "Segoe UI", Arial, sans-serif; font-size: 12px; fill: #2c3e50; }',
        '.header { font-size: 16px; font-weight: bold; fill: #1a252f; text-anchor: middle; }',
        '.box { fill: #f8f9fa; stroke: #2980b9; stroke-width: 1.6; rx: 6; }',
        '.box-accent { fill: #ebf5fb; stroke: #1b4f72; stroke-width: 2.0; rx: 6; }',
        '.box-decision { fill: #fef9e7; stroke: #d35400; stroke-width: 1.8; }',
        '.box-title { font-weight: bold; font-size: 12px; fill: #1a252f; text-anchor: middle; }',
        '.box-desc { font-size: 10.5px; fill: #566573; text-anchor: middle; }',
        '.flow-line { stroke: #2c3e50; stroke-width: 1.5; fill: none; marker-end: url(#arrow); }',
        '</style>',
        f'<text class="header" x="{w/2}" y="28">Блок-схема генетического алгоритма NSGA-II (Вариант 6: Меню на неделю)</text>',
    ]

    # Блоки схемы
    blocks = [
        {"id": "b1", "x": 50, "y": 60, "w": 230, "h": 50, "title": "1. Инициализация P_0 (N=100)", "desc": "21 слот: случайный выбор блюд по категориям"},
        {"id": "b2", "x": 50, "y": 140, "w": 230, "h": 50, "title": "2. Оценивание критериев", "desc": "f1 (стоимость), f2 (нутриенты), f3 (монотонность)"},
        {"id": "b3", "x": 50, "y": 220, "w": 230, "h": 54, "title": "3. Недоминируемая сортировка", "desc": "Деление P_0 на фронты F1, F2, ... и crowding distance", "accent": True},
        {"id": "b4", "x": 50, "y": 305, "w": 230, "h": 50, "title": "4. Внешний архив Парето", "desc": "Сохранение всех найденных решений F1"},

        {"id": "b5", "x": 345, "y": 60, "w": 230, "h": 50, "title": "5. Турнирная селекция", "desc": "Турнир k=2 по рангу и crowding distance"},
        {"id": "b6", "x": 345, "y": 140, "w": 230, "h": 50, "title": "6. Кроссовер (p=0.90)", "desc": "Однородный / суточный с сохранением слотов"},
        {"id": "b7", "x": 345, "y": 220, "w": 230, "h": 54, "title": "7. Специализированная мутация", "desc": "Замена блюда, swap дней, разрушение монотонности"},
        {"id": "b8", "x": 345, "y": 305, "w": 230, "h": 50, "title": "8. Оценка потомков Q_t (N)", "desc": "Расчет f1, f2, f3 для N новых особей"},

        {"id": "b9", "x": 640, "y": 60, "w": 230, "h": 50, "title": "9. Объединение R_t = P_t U Q_t", "desc": "Пул из 2N особей (родители + потомки)"},
        {"id": "b10", "x": 640, "y": 140, "w": 230, "h": 54, "title": "10. Быстрая сортировка 2N", "desc": "Разбиение на фронты Парето F1, F2, ...", "accent": True},
        {"id": "b11", "x": 640, "y": 225, "w": 230, "h": 60, "title": "11. Элитарный отбор P_{t+1}", "desc": "Заполнение фронтами F_i; обрезка по crowding distance"},
        {"id": "b12", "x": 640, "y": 320, "w": 230, "h": 52, "title": "12. Критерий останова?", "desc": "t == 150 поколений", "decision": True},

        {"id": "b13", "x": 640, "y": 420, "w": 230, "h": 50, "title": "13. Итоговый фронт Парето", "desc": "Множество компромиссных недельных меню", "accent": True},
    ]

    for b in blocks:
        bx, by, bw, bh = b["x"], b["y"], b["w"], b["h"]
        cls = "box-accent" if b.get("accent") else ("box-decision" if b.get("decision") else "box")
        parts.append(f'<rect class="{cls}" x="{bx}" y="{by}" width="{bw}" height="{bh}"/>')
        parts.append(f'<text class="box-title" x="{bx + bw/2}" y="{by + 20}">{b["title"]}</text>')
        parts.append(f'<text class="box-desc" x="{bx + bw/2}" y="{by + 37}">{b["desc"]}</text>')

    # Стрелки соединений
    arrows = [
        # Колонка 1
        (165, 110, 165, 140),
        (165, 190, 165, 220),
        (165, 274, 165, 305),
        # Переход к генерации potomkov
        (280, 245, 345, 85),
        # Колонка 2
        (460, 110, 460, 140),
        (460, 190, 460, 220),
        (460, 274, 460, 305),
        # Переход к объединению
        (575, 330, 640, 85),
        # Колонка 3
        (755, 110, 755, 140),
        (755, 194, 755, 225),
        (755, 285, 755, 320),
        # Выход из критерия останова (Да -> 13)
        (755, 372, 755, 420),
    ]

    for x1, y1, x2, y2 in arrows:
        parts.append(f'<path class="flow-line" d="M {x1} {y1} L {x2} {y2}"/>')

    # Обратная петля (Нет -> на шаг 5)
    parts.append('<path class="flow-line" d="M 640 346 L 610 346 L 610 35 L 460 35 L 460 60"/>')
    parts.append('<text x="618" y="340" font-size="10" font-weight="bold" fill="#c0392b">Нет</text>')
    parts.append('<text x="765" y="402" font-size="10" font-weight="bold" fill="#27ae60">Да</text>')

    parts.append('</svg>')
    Path(filepath).write_text("\n".join(parts), encoding="utf-8")


def write_3d_viewer_html(filepath, nsga_pf, wsga_pf, rand_pf, rep_solutions):
    """Создает автономный интерактивный 3D HTML визуализатор Парето-фронта

    на чистом HTML5 Canvas / JavaScript без необходимости установки сторонних библиотек.
    """
    html_content = f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8">
  <title>Интерактивный 3D Фронт Парето — Меню на неделю (Вариант 6)</title>
  <style>
    body {{
      margin: 0;
      padding: 15px;
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      background: #0f172a;
      color: #f1f5f9;
      display: flex;
      flex-direction: column;
      height: 100vh;
      box-sizing: border-box;
    }}
    .header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 10px;
    }}
    h1 {{ font-size: 20px; margin: 0; color: #38bdf8; }}
    .subtitle {{ font-size: 13px; color: #94a3b8; }}
    .main-container {{
      display: flex;
      flex: 1;
      gap: 15px;
      min-height: 0;
    }}
    #canvas-container {{
      flex: 1;
      background: #1e293b;
      border-radius: 8px;
      position: relative;
      overflow: hidden;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }}
    canvas {{ width: 100%; height: 100%; display: block; }}
    .sidebar {{
      width: 330px;
      background: #1e293b;
      border-radius: 8px;
      padding: 15px;
      box-sizing: border-box;
      display: flex;
      flex-direction: column;
      gap: 12px;
      overflow-y: auto;
    }}
    .panel {{
      background: #0f172a;
      border-radius: 6px;
      padding: 12px;
      font-size: 13px;
    }}
    .panel h3 {{ margin-top: 0; font-size: 14px; color: #38bdf8; border-bottom: 1px solid #334155; padding-bottom: 5px; }}
    .badge {{
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 50%;
      margin-right: 6px;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      margin: 6px 0;
      cursor: pointer;
      user-select: none;
    }}
    .controls-hint {{
      font-size: 11px;
      color: #94a3b8;
      line-height: 1.4;
    }}
    #tooltip {{
      position: absolute;
      display: none;
      background: rgba(15, 23, 42, 0.95);
      border: 1px solid #38bdf8;
      border-radius: 6px;
      padding: 10px;
      font-size: 12px;
      pointer-events: none;
      box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
      z-index: 100;
    }}
  </style>
</head>
<body>
  <div class="header">
    <div>
      <h1>Интерактивная 3D визуализация фронта Парето</h1>
      <div class="subtitle">Вариант 6: Многокритериальное формирование меню на неделю (NSGA-II)</div>
    </div>
    <div class="controls-hint">Вращение: левая кнопка мыши | Масштаб: колесико мыши</div>
  </div>

  <div class="main-container">
    <div id="canvas-container">
      <canvas id="plot3d"></canvas>
      <div id="tooltip"></div>
    </div>

    <div class="sidebar">
      <div class="panel">
        <h3>Оси критериев (min)</h3>
        <div><b>X (красная):</b> Стоимость меню (руб.)</div>
        <div><b>Y (зеленая):</b> Отклонение нутриентов (%)</div>
        <div><b>Z (синяя):</b> Монотонность меню (%)</div>
      </div>

      <div class="panel">
        <h3>Методы и точки</h3>
        <label class="legend-item"><input type="checkbox" id="show-nsga" checked> <span class="badge" style="background:#38bdf8"></span> NSGA-II ({len(nsga_pf)} точек)</label>
        <label class="legend-item"><input type="checkbox" id="show-wsga" checked> <span class="badge" style="background:#f97316"></span> Взвешенная сумма ({len(wsga_pf)} точек)</label>
        <label class="legend-item"><input type="checkbox" id="show-rand" checked> <span class="badge" style="background:#94a3b8"></span> Случайный поиск ({len(rand_pf)} точек)</label>
      </div>

      <div class="panel">
        <h3>Характерные меню</h3>
        <div style="margin: 4px 0;"><span class="badge" style="background:#10b981"></span> <b>Бюджетное:</b> {min(x['cost'] for x in nsga_pf):.0f} руб.</div>
        <div style="margin: 4px 0;"><span class="badge" style="background:#a855f7"></span> <b>Сбалансированное:</b> ошибка {min(x['nutrient_error'] for x in nsga_pf):.1f}%</div>
        <div style="margin: 4px 0;"><span class="badge" style="background:#ef4444"></span> <b>Разнообразное:</b> 21 блюдо, 100% разн.</div>
        <div style="margin: 4px 0;"><span class="badge" style="background:#f59e0b"></span> <b>Knee Point:</b> оптимальный баланс</div>
      </div>

      <div class="panel">
        <h3>Информация о точке</h3>
        <div id="selected-info" style="color:#94a3b8; font-style:italic;">Наведите курсор на точку для просмотра показателей</div>
      </div>
    </div>
  </div>

  <script>
    const dataNSGA = {json.dumps([{"cost": x["cost"], "nutr": x["nutrient_error"], "monot": x["monotony"], "unique": x["unique_dishes"], "type": "NSGA-II"} for x in nsga_pf])};
    const dataWSGA = {json.dumps([{"cost": x["cost"], "nutr": x["nutrient_error"], "monot": x["monotony"], "unique": x["unique_dishes"], "type": "WSGA"} for x in wsga_pf])};
    const dataRand = {json.dumps([{"cost": x["cost"], "nutr": x["nutrient_error"], "monot": x["monotony"], "unique": x["unique_dishes"], "type": "Random"} for x in rand_pf])};

    const canvas = document.getElementById('plot3d');
    const ctx = canvas.getContext('2d');
    const tooltip = document.getElementById('tooltip');
    const infoDiv = document.getElementById('selected-info');

    let rotX = 0.5;
    let rotY = -0.7;
    let zoom = 1.0;
    let isDragging = false;
    let lastMouseX = 0;
    let lastMouseY = 0;

    const bounds = {{
      cost: [3200, 6500],
      nutr: [10, 45],
      monot: [0, 90]
    }};

    function resize() {{
      canvas.width = canvas.parentElement.clientWidth;
      canvas.height = canvas.parentElement.clientHeight;
      draw();
    }}
    window.addEventListener('resize', resize);

    canvas.addEventListener('mousedown', (e) => {{
      isDragging = true;
      lastMouseX = e.clientX;
      lastMouseY = e.clientY;
    }});
    window.addEventListener('mouseup', () => isDragging = false);
    canvas.addEventListener('mousemove', (e) => {{
      if (isDragging) {{
        const dx = e.clientX - lastMouseX;
        const dy = e.clientY - lastMouseY;
        rotY += dx * 0.008;
        rotX += dy * 0.008;
        lastMouseX = e.clientX;
        lastMouseY = e.clientY;
        draw();
      }}
      checkHover(e);
    }});
    canvas.addEventListener('wheel', (e) => {{
      zoom *= (e.deltaY > 0 ? 0.9 : 1.1);
      zoom = Math.max(0.5, Math.min(3.0, zoom));
      draw();
      e.preventDefault();
    }});

    document.querySelectorAll('.legend-item input').forEach(el => el.addEventListener('change', draw));

    let projectedPoints = [];

    function project(x, y, z) {{
      // нормализация [-1, 1]
      const nx = ((x - bounds.cost[0]) / (bounds.cost[1] - bounds.cost[0]) - 0.5) * 2;
      const ny = ((y - bounds.nutr[0]) / (bounds.nutr[1] - bounds.nutr[0]) - 0.5) * 2;
      const nz = ((z - bounds.monot[0]) / (bounds.monot[1] - bounds.monot[0]) - 0.5) * 2;

      // 3D вращение
      // вокруг Y
      const cosY = Math.cos(rotY), sinY = Math.sin(rotY);
      const x1 = nx * cosY + nz * sinY;
      const z1 = -nx * sinY + nz * cosY;

      // вокруг X
      const cosX = Math.cos(rotX), sinX = Math.sin(rotX);
      const y2 = ny * cosX - z1 * sinX;
      const z2 = ny * sinX + z1 * cosX;

      const scale = 140 * zoom;
      const dist = 4.0;
      const fov = dist / (dist + z2);

      const cx = canvas.width / 2;
      const cy = canvas.height / 2;

      return {{
        sx: cx + x1 * scale * fov,
        sy: cy - y2 * scale * fov,
        depth: z2
      }};
    }}

    function draw() {{
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      projectedPoints = [];

      // Оси куба
      const origin = project(bounds.cost[0], bounds.nutr[0], bounds.monot[0]);
      const axX = project(bounds.cost[1], bounds.nutr[0], bounds.monot[0]);
      const axY = project(bounds.cost[0], bounds.nutr[1], bounds.monot[0]);
      const axZ = project(bounds.cost[0], bounds.nutr[0], bounds.monot[1]);

      function drawAxis(p1, p2, color, label) {{
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(p1.sx, p1.sy);
        ctx.lineTo(p2.sx, p2.sy);
        ctx.stroke();
        ctx.fillStyle = color;
        ctx.font = 'bold 12px sans-serif';
        ctx.fillText(label, p2.sx + 5, p2.sy + 5);
      }}

      drawAxis(origin, axX, '#ef4444', 'Стоимость (X)');
      drawAxis(origin, axY, '#10b981', 'Нутриенты (Y)');
      drawAxis(origin, axZ, '#38bdf8', 'Монотонность (Z)');

      const showNSGA = document.getElementById('show-nsga').checked;
      const showWSGA = document.getElementById('show-wsga').checked;
      const showRand = document.getElementById('show-rand').checked;

      const all = [];
      if (showRand) dataRand.forEach(p => all.push({{ ...p, color: '#94a3b8', r: 3 }}));
      if (showWSGA) dataWSGA.forEach(p => all.push({{ ...p, color: '#f97316', r: 5, shape: 'diamond' }}));
      if (showNSGA) dataNSGA.forEach(p => all.push({{ ...p, color: '#38bdf8', r: 4.5 }}));

      // Сортировка по глубине z
      const plotted = all.map(p => {{
        const pt = project(p.cost, p.nutr, p.monot);
        return {{ ...p, ...pt }};
      }}).sort((a, b) => b.depth - a.depth);

      projectedPoints = plotted;

      plotted.forEach(p => {{
        ctx.fillStyle = p.color;
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(p.sx, p.sy, p.r, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }});
    }}

    function checkHover(e) {{
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;

      let found = null;
      for (const p of projectedPoints) {{
        const dist = Math.hypot(p.sx - mx, p.sy - my);
        if (dist <= 7) {{
          found = p;
          break;
        }}
      }}

      if (found) {{
        tooltip.style.display = 'block';
        tooltip.style.left = (mx + 15) + 'px';
        tooltip.style.top = (my - 10) + 'px';
        tooltip.innerHTML = `
          <div style="font-weight:bold; color:${{found.color}};">${{found.type}}</div>
          <div>Стоимость: <b>${{found.cost.toFixed(0)}} руб.</b></div>
          <div>Отклонение нутриентов: <b>${{found.nutr.toFixed(1)}}%</b></div>
          <div>Монотонность: <b>${{found.monot.toFixed(1)}}%</b></div>
          <div>Уникальных блюд: <b>${{found.unique}} / 21</b></div>
        `;
        infoDiv.innerHTML = `
          <div><b>Метод:</b> ${{found.type}}</div>
          <div><b>Стоимость:</b> ${{found.cost.toFixed(0)}} руб.</div>
          <div><b>Баланс БЖУ/ккал:</b> ${{found.nutr.toFixed(2)}}%</div>
          <div><b>Монотонность:</b> ${{found.monot.toFixed(2)}}%</div>
          <div><b>Уникальных блюд:</b> ${{found.unique}} из 21</div>
        `;
      }} else {{
        tooltip.style.display = 'none';
      }}
    }}

    resize();
  </script>
</body>
</html>
"""
    Path(filepath).write_text(html_content, encoding="utf-8")


def main():
    start_time = time.time()
    data = load_data("data/dishes.json")
    with Path("config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print("ЭКСПЕРИМЕНТАЛЬНЫЙ КОМПЛЕКС: ЛАБОРАТОРНАЯ РАБОТА № 3 (ВАРИАНТ 6)")
    print("Тема: Многокритериальная оптимизация недельного меню алгоритмом NSGA-II")
    print("================================================================================")

    # 1. Серия из 20 независимых запусков NSGA-II
    print(f"\n>>> 1. Запуск серии из {config['runs']} прогонов NSGA-II (seeds {config['base_seed']}..{config['base_seed']+config['runs']-1})...")
    nsga_run_results = []
    all_nsga_histories = []
    all_nsga_solutions = []

    for run_idx in range(config["runs"]):
        seed = config["base_seed"] + run_idx
        t0 = time.time()
        res = run_nsga2(data, config, seed=seed, track_history=True, verbose=False)
        dt = time.time() - t0

        pf = res["pareto_front"]
        all_nsga_histories.append(res["history"])
        all_nsga_solutions.extend(pf)

        costs = [x["cost"] for x in pf]
        nutrs = [x["nutrient_error"] for x in pf]
        monots = [x["monotony"] for x in pf]

        row = {
            "method": "NSGA-II",
            "run": run_idx + 1,
            "seed": seed,
            "pareto_front_size": len(pf),
            "hypervolume": round(res["hypervolume"], 5),
            "spacing": round(res["spacing"], 2),
            "min_cost_rub": round(min(costs), 1),
            "min_nutrient_error_pct": round(min(nutrs), 2),
            "min_monotony_pct": round(min(monots), 2),
            "max_diversity_score": round(max(x["diversity_score"] for x in pf), 3),
            "evaluations": res["evaluations"],
            "runtime_seconds": round(dt, 2),
        }
        nsga_run_results.append(row)
        print(f"  [Запуск {run_idx+1:2d}/20] seed={seed} | Фронт={len(pf):2d} | HV={res['hypervolume']:.4f} | Spacing={res['spacing']:.2f} | Время={dt:.2f}c")

    # Итоговый Парето-фронт по объединению всех 20 прогонов
    all_nsga_fronts = fast_non_dominated_sort(all_nsga_solutions)
    combined_nsga_pf = all_nsga_fronts[0]

    # Убираем дубликаты
    unique_nsga_pf = []
    seen = set()
    for ind in combined_nsga_pf:
        key = (round(ind["cost"], 1), round(ind["nutrient_error"], 2), round(ind["monotony"], 2))
        if key not in seen:
            seen.add(key)
            unique_nsga_pf.append(ind)

    # 2. Взвешенная сумма критериев (Weighted Sum GA)
    print("\n>>> 2. Запуск оптимизации взвешенной суммой критериев (WSGA) по сетке весов...")
    t0 = time.time()
    wsga_res = run_weighted_sum_experiment(data, config, seed=config["base_seed"])
    wsga_time = time.time() - t0
    wsga_pf = wsga_res["pareto_front"]
    print(f"  WSGA завершено за {wsga_time:.2f}c: найдено {len(wsga_pf)} точек Парето | HV={wsga_res['hypervolume']:.4f} | Spacing={wsga_res['spacing']:.2f}")

    # 3. Случайный допустимый поиск (Random Search)
    budget = config["population_size"] * config["generations"]
    print(f"\n>>> 3. Запуск случайного допустимого поиска (Random Search, бюджет={budget} оценок)...")
    t0 = time.time()
    rand_res = run_random_search(data, config, seed=config["base_seed"], budget=budget)
    rand_time = time.time() - t0
    rand_pf = rand_res["pareto_front"]
    print(f"  Random Search завершено за {rand_time:.2f}c: найдено {len(rand_pf)} точек Парето | HV={rand_res['hypervolume']:.4f} | Spacing={rand_res['spacing']:.2f}")

    # 4. Сводная статистика по серии 20 запусков NSGA-II
    hvs = [r["hypervolume"] for r in nsga_run_results]
    sps = [r["spacing"] for r in nsga_run_results]
    sizes = [r["pareto_front_size"] for r in nsga_run_results]
    min_costs = [r["min_cost_rub"] for r in nsga_run_results]
    min_nutrs = [r["min_nutrient_error_pct"] for r in nsga_run_results]
    max_divs = [r["max_diversity_score"] for r in nsga_run_results]
    runtimes = [r["runtime_seconds"] for r in nsga_run_results]

    summary_rows = [
        {
            "metric": "Hypervolume (нормированный)",
            "best": round(max(hvs), 5),
            "mean": round(statistics.mean(hvs), 5),
            "median": round(statistics.median(hvs), 5),
            "std": round(statistics.stdev(hvs), 5),
            "worst": round(min(hvs), 5),
        },
        {
            "metric": "Равномерность Spacing (spread)",
            "best": round(min(sps), 2),
            "mean": round(statistics.mean(sps), 2),
            "median": round(statistics.median(sps), 2),
            "std": round(statistics.stdev(sps), 2),
            "worst": round(max(sps), 2),
        },
        {
            "metric": "Число точек Парето-фронта (|PF|)",
            "best": max(sizes),
            "mean": round(statistics.mean(sizes), 1),
            "median": round(statistics.median(sizes), 1),
            "std": round(statistics.stdev(sizes), 2),
            "worst": min(sizes),
        },
        {
            "metric": "Минимальная стоимость меню (руб.)",
            "best": min(min_costs),
            "mean": round(statistics.mean(min_costs), 1),
            "median": round(statistics.median(min_costs), 1),
            "std": round(statistics.stdev(min_costs), 2),
            "worst": max(min_costs),
        },
        {
            "metric": "Минимальная ошибка нутриентов (%)",
            "best": min(min_nutrs),
            "mean": round(statistics.mean(min_nutrs), 2),
            "median": round(statistics.median(min_nutrs), 2),
            "std": round(statistics.stdev(min_nutrs), 2),
            "worst": max(min_nutrs),
        },
        {
            "metric": "Максимальный индекс разнообразия",
            "best": max(max_divs),
            "mean": round(statistics.mean(max_divs), 3),
            "median": round(statistics.median(max_divs), 3),
            "std": round(statistics.stdev(max_divs), 3),
            "worst": min(max_divs),
        },
        {
            "metric": "Время одного прогона (сек.)",
            "best": min(runtimes),
            "mean": round(statistics.mean(runtimes), 2),
            "median": round(statistics.median(runtimes), 2),
            "std": round(statistics.stdev(runtimes), 2),
            "worst": max(runtimes),
        },
    ]

    # Сравнение трех подходов
    comparison_metrics = {
        "nsga": {
            "hv": statistics.mean(hvs),
            "sp": statistics.mean(sps),
            "size": statistics.mean(sizes),
        },
        "wsga": {
            "hv": wsga_res["hypervolume"],
            "sp": wsga_res["spacing"],
            "size": len(wsga_pf),
        },
        "rand": {
            "hv": rand_res["hypervolume"],
            "sp": rand_res["spacing"],
            "size": len(rand_pf),
        },
    }

    # 5. Выделение характерных меню
    rep_csv_rows, rep_json_data, rep_solutions = extract_representative_solutions(unique_nsga_pf, data)

    # 6. Запись файлов результатов
    print("\n>>> 4. Запись результатов в папку outputs/...")
    write_csv(output_dir / "experiment_results.csv", nsga_run_results, list(nsga_run_results[0].keys()))
    write_csv(output_dir / "summary.csv", summary_rows, list(summary_rows[0].keys()))

    # Запись точек фронта Парето NSGA-II
    nsga_pf_rows = []
    for ind in unique_nsga_pf:
        nsga_pf_rows.append({
            "cost_rub": round(ind["cost"], 2),
            "nutrient_error_pct": round(ind["nutrient_error"], 2),
            "monotony_pct": round(ind["monotony"], 2),
            "diversity_score": round(ind["diversity_score"], 3),
            "unique_dishes": ind["unique_dishes"],
            "consecutive_repeats": ind["consecutive_repeats"],
            "unique_protein_sources": ind["unique_protein_sources"],
            "unique_cuisines": ind["unique_cuisines"],
        })
    write_csv(output_dir / "pareto_front_nsga2.csv", nsga_pf_rows, list(nsga_pf_rows[0].keys()))

    # Запись фронта WSGA
    wsga_pf_rows = []
    for ind in wsga_pf:
        wsga_pf_rows.append({
            "cost_rub": round(ind["cost"], 2),
            "nutrient_error_pct": round(ind["nutrient_error"], 2),
            "monotony_pct": round(ind["monotony"], 2),
            "diversity_score": round(ind["diversity_score"], 3),
            "unique_dishes": ind["unique_dishes"],
        })
    write_csv(output_dir / "pareto_front_weighted_sum.csv", wsga_pf_rows, list(wsga_pf_rows[0].keys()))

    # Запись фронта Random Search
    rand_pf_rows = []
    for ind in rand_pf:
        rand_pf_rows.append({
            "cost_rub": round(ind["cost"], 2),
            "nutrient_error_pct": round(ind["nutrient_error"], 2),
            "monotony_pct": round(ind["monotony"], 2),
            "diversity_score": round(ind["diversity_score"], 3),
            "unique_dishes": ind["unique_dishes"],
        })
    write_csv(output_dir / "pareto_front_random.csv", rand_pf_rows, list(rand_pf_rows[0].keys()))

    # Запись репрезентативных меню
    write_csv(output_dir / "representative_menus.csv", rep_csv_rows, list(rep_csv_rows[0].keys()))
    (output_dir / "representative_menus.json").write_text(json.dumps(rep_json_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Метаданные запуска
    run_info = {
        "variant": 6,
        "track": "A",
        "task": "Формирование меню на неделю",
        "criteria": [
            "f1: Стоимость меню (руб.) -> min",
            "f2: Отклонение нутриентов (%) -> min",
            "f3: Монотонность меню (%) -> min",
        ],
        "experiment_config": config,
        "runs": config["runs"],
        "total_evaluations_per_nsga_run": config["population_size"] * config["generations"] + config["population_size"],
        "comparison_metrics": comparison_metrics,
        "total_experiment_time_seconds": round(time.time() - start_time, 2),
    }
    (output_dir / "run_information.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 7. Построение графиков
    print("\n>>> 5. Построение SVG-диаграмм и интерактивного 3D визуализатора...")
    write_pareto_projections_svg(output_dir / "pareto_projections.svg", unique_nsga_pf, wsga_pf, rand_pf, rep_solutions)
    write_convergence_svg(output_dir / "convergence_hypervolume.svg", all_nsga_histories)
    write_comparison_methods_svg(output_dir / "comparison_methods.svg", comparison_metrics)
    write_algorithm_scheme_svg(output_dir / "algorithm_scheme.svg")
    write_3d_viewer_html(output_dir / "pareto_3d_viewer.html", unique_nsga_pf, wsga_pf, rand_pf, rep_solutions)

    print("\n>>> Все результаты успешно сформированы в папке outputs/!")
    print(f"Общее время выполнения экспериментов: {time.time() - start_time:.2f} сек.")


if __name__ == "__main__":
    main()

