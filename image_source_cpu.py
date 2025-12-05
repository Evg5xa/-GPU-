import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.optimize import curve_fit
import time


def get_virtual_sources(source, room_width, room_height, max_order, reflection_coeffs_freq):
    start_time = time.time()

    # reflection_coeffs_freq = {
    #     'left': [r1, r2, r3, r4, r5, r6],  # для 6 частотных диапазонов
    #     'right': [r1, r2, r3, r4, r5, r6],
    #     'top': [r1, r2, r3, r4, r5, r6],
    #     'bottom': [r1, r2, r3, r4, r5, r6]
    # }
    # где r от 0 (полное поглощение) до 1 (полное отражение) для каждой частоты
    virtual_sources = []

    for n in range(-max_order, max_order + 1):
        for m in range(-max_order, max_order + 1):
            order = abs(n) + abs(m)
            if order == 0 or order > max_order:
                continue

            # Определяем количество отражений от каждой стены
            left_count = abs(n) if n < 0 else 0
            right_count = n if n > 0 else 0
            bottom_count = abs(m) if m < 0 else 0
            top_count = m if m > 0 else 0

            # Для каждого частотного диапазона вычисляем эффективный коэффициент
            effective_energy_factors = []
            for freq_idx in range(6):  # 6 частотных диапазонов
                factor = (
                        (reflection_coeffs_freq['left'][freq_idx] ** left_count) *
                        (reflection_coeffs_freq['right'][freq_idx] ** right_count) *
                        (reflection_coeffs_freq['bottom'][freq_idx] ** bottom_count) *
                        (reflection_coeffs_freq['top'][freq_idx] ** top_count)
                )
                effective_energy_factors.append(factor)

            if all(f == 0 for f in effective_energy_factors):
                continue  # Если энергия нулевая для всех частот, пропускаем

            # Вычисляем координаты виртуального источника
            x = n * room_width + (source[0] if n % 2 == 0 else room_width - source[0])
            y = m * room_height + (source[1] if m % 2 == 0 else room_height - source[1])
            virtual_sources.append({
                'pos': [x, y],
                'order': order,
                'energy_factors': effective_energy_factors,  # 6 значений для 6 частот
                'left_refl': left_count,
                'right_refl': right_count,
                'top_refl': top_count,
                'bottom_refl': bottom_count
            })

    elapsed = time.time() - start_time
    print(f"[get_virtual_sources] Выполнено за {elapsed:.4f} секунд")
    print(f"[get_virtual_sources] Найдено виртуальных источников: {len(virtual_sources)}")
    return virtual_sources


def find_wall_intersections(p1, p2, room_width, room_height):
    start_time = time.time()

    intersections = []
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    if dx != 0:
        t = (0 - p1[0]) / dx
        if 0 <= t <= 1:
            y = p1[1] + t * dy
            if 0 <= y <= room_height:
                intersections.append((0, y, 'left', t))
        t = (room_width - p1[0]) / dx
        if 0 <= t <= 1:
            y = p1[1] + t * dy
            if 0 <= y <= room_height:
                intersections.append((room_width, y, 'right', t))

    if dy != 0:
        t = (0 - p1[1]) / dy
        if 0 <= t <= 1:
            x = p1[0] + t * dx
            if 0 <= x <= room_width:
                intersections.append((x, 0, 'bottom', t))
        t = (room_height - p1[1]) / dy
        if 0 <= t <= 1:
            x = p1[0] + t * dx
            if 0 <= x <= room_width:
                intersections.append((x, room_height, 'top', t))

    elapsed = time.time() - start_time
    if elapsed > 0.001:  # Показываем только если операция долгая
        print(f"[find_wall_intersections] Выполнено за {elapsed:.6f} секунд")

    return sorted(intersections, key=lambda inter: inter[3])


def plot_reflection_path(ax, points, color):
    xs, ys = zip(*points)
    ax.plot(xs, ys, linestyle='-', linewidth=1.5, color=color, alpha=0.7)
    # Не отмечаем начальную и конечную точки
    for p in points[1:-1]:
        ax.plot(p[0], p[1], 'bo', markersize=6, alpha=0.7)


def draw_room_and_sources(room_width, room_height, source, receiver, max_order, reflection_coeffs_freq):
    total_start_time = time.time()

    fig, ax = plt.subplots(figsize=(12, 10))
    room_patch = patches.Rectangle((0, 0), room_width, room_height, linewidth=2,
                                   edgecolor='orange', facecolor='none', linestyle='-')
    ax.add_patch(room_patch)
    ax.plot(source[0], source[1], 'ro', markersize=10, label='Источник')
    ax.plot(receiver[0], receiver[1], 'go', markersize=10, label='Приёмник')
    ax.plot([source[0], receiver[0]], [source[1], receiver[1]],
            color='purple', linestyle='-', linewidth=2, alpha=0.8, label='Прямой путь')

    speed_of_sound = 343

    # Генерация виртуальных источников
    virtual_sources = get_virtual_sources(source, room_width, room_height, max_order, reflection_coeffs_freq)

    # Для каждой частоты будем собирать свои массивы
    freq_labels = ['125 Гц', '250 Гц', '500 Гц', '1 кГц', '2 кГц', '4 кГц']
    times_freq = [[] for _ in range(6)]
    energies_freq = [[] for _ in range(6)]
    distances = []

    # Обработка виртуальных источников
    process_sources_start = time.time()
    for vs in virtual_sources:
        pos = vs['pos']
        order = vs['order']
        energy_factors = vs['energy_factors']  # [r1, r2, ..., r6] для 6 частот

        # Поиск пересечений со стенами
        intersections_start = time.time()
        reflections = find_wall_intersections(pos, receiver, room_width, room_height)
        intersections_time = time.time() - intersections_start

        # Построение пути
        path_points = [pos] + [(ix[0], ix[1]) for ix in reflections] + [receiver]

        # Вычисление длины пути
        distance_start = time.time()
        r = 0
        for i in range(len(path_points) - 1):
            seg_length = np.linalg.norm(np.array(path_points[i + 1]) - np.array(path_points[i]))
            r += seg_length
        distances.append(r)
        distance_time = time.time() - distance_start

        # Вычисление энергии для каждой частоты
        energy_start = time.time()
        for freq_idx in range(6):
            base_energy = 1 / r ** 2
            energy = base_energy * energy_factors[freq_idx]  # Учитываем коэффициенты отражения для конкретной частоты
            time_sec = r / speed_of_sound
            if energy > 0:
                times_freq[freq_idx].append(time_sec)
                energies_freq[freq_idx].append(energy)
        energy_time = time.time() - energy_start

        # Вывод информации о луче
        print(f"Луч порядка {order}: длина = {r:.2f} м")
        for freq_idx in range(6):
            energy = (1 / r ** 2) * energy_factors[freq_idx]
            time_sec = r / speed_of_sound
            if energy > 0:
                print(f"  {freq_labels[freq_idx]}: время = {time_sec:.4f} с, энергия = {energy:.5f}")

        # Визуализация пути
        plot_reflection_path(ax, path_points, color='y')
        ax.plot(pos[0], pos[1], 'ro', markersize=5, alpha=0.7)
        offset_x = (pos[0] // room_width) * room_width
        offset_y = (pos[1] // room_height) * room_height
        virtual_room = patches.Rectangle(
            (offset_x, offset_y), room_width, room_height,
            linewidth=1, edgecolor='black', facecolor='none',
            linestyle='--', alpha=0.5
        )
        ax.add_patch(virtual_room)

    process_sources_time = time.time() - process_sources_start
    print(f"\n[Обработка виртуальных источников] Выполнено за {process_sources_time:.4f} секунд")
    print(
        f"[Обработка виртуальных источников] Среднее время на источник: {process_sources_time / len(virtual_sources) * 1000:.2f} мс")

    # Настройка графика
    ax.set_xlim(-room_width, 2 * room_width)
    ax.set_ylim(-room_height, 2 * room_height)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    ax.set_title(f'Моделирование акустики помещения\n'
                 f'Коэффициенты отражения зависят от частоты\n'
                 f'Всего источников: {len(virtual_sources)}, Время вычислений: {process_sources_time:.2f} с')
    plt.show()

    # --- Аппроксимация для каждой частоты: E(r) = C / r^2 ---
    approximation_start = time.time()
    for freq_idx in range(6):
        if len(energies_freq[freq_idx]) > 1:
            plt.figure(figsize=(10, 6))
            plt.scatter(distances[:len(energies_freq[freq_idx])], energies_freq[freq_idx],
                        c='green', alpha=0.7, label=f'Точки: E(r) для {freq_labels[freq_idx]}')

            def inv_r_squared(r, C):
                return C / (r ** 2)

            try:
                fit_start = time.time()
                popt, pcov = curve_fit(inv_r_squared, distances[:len(energies_freq[freq_idx])],
                                       energies_freq[freq_idx], p0=(1), maxfev=5000)
                fit_time = time.time() - fit_start
                C_fit = popt[0]
                print(f"\nПараметр аппроксимации для {freq_labels[freq_idx]} E(r) = C / r^2: C = {C_fit:.4f}")
                print(f"Время аппроксимации: {fit_time:.4f} секунд")

                r_smooth = np.linspace(min(distances[:len(energies_freq[freq_idx])]),
                                       max(distances[:len(energies_freq[freq_idx])]), 500)
                e_smooth = inv_r_squared(r_smooth, C_fit)
                plt.plot(r_smooth, e_smooth, 'b-', linewidth=2)
            except RuntimeError:
                print(f"\nНе удалось подобрать параметры для {freq_labels[freq_idx]} E(r) = C / r^2.")

            plt.title(f'Зависимость энергии отражённых лучей от расстояния\n'
                      f'для виртуальных источников до {max_order} порядка ({freq_labels[freq_idx]})')
            plt.xlabel('Расстояние (м)')
            plt.ylabel('Энергия')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            plt.show()

    approximation_time = time.time() - approximation_start
    print(f"\n[Аппроксимация кривых] Выполнено за {approximation_time:.4f} секунд")

    total_time = time.time() - total_start_time
    print(f"\n{'=' * 60}")
    print(f"ОБЩЕЕ ВРЕМЯ ВЫПОЛНЕНИЯ: {total_time:.4f} секунд")
    print(f"{'=' * 60}")

    # Итоговая статистика
    print(f"\nИТОГОВАЯ СТАТИСТИКА:")
    print(f"Всего виртуальных источников: {len(virtual_sources)}")
    print(f"Максимальный порядок отражений: {max_order}")
    print(f"Размер помещения: {room_width} x {room_height} м")
    print(f"Скорость звука: {speed_of_sound} м/с")


# Параметры
room_width = 10
room_height = 8
source = [2, 3]
receiver = [5, 5]

print("=" * 60)
print("МОДЕЛИРОВАНИЕ АКУСТИКИ ПОМЕЩЕНИЯ")
print("=" * 60)

max_reflection_order = int(input('\nВведите максимальный порядок источников: '))

print("\nВведите коэффициенты отражения для стен для 6 частотных диапазонов (от 0 до 1):")
print("Частоты: 125 Гц, 250 Гц, 500 Гц, 1 кГц, 2 кГц, 4 кГц")

# Для каждой стены вводим 6 значений
print("\nЛевая стена:")
left_freq = [float(input(f'  {freq} Гц: ')) for freq in [125, 250, 500, 1000, 2000, 4000]]

print("\nПравая стена:")
right_freq = [float(input(f'  {freq} Гц: ')) for freq in [125, 250, 500, 1000, 2000, 4000]]

print("\nВерхняя стена:")
top_freq = [float(input(f'  {freq} Гц: ')) for freq in [125, 250, 500, 1000, 2000, 4000]]

print("\nНижняя стена:")
bottom_freq = [float(input(f'  {freq} Гц: ')) for freq in [125, 250, 500, 1000, 2000, 4000]]

reflection_coeffs_freq = {
    'left': left_freq,
    'right': right_freq,
    'top': top_freq,
    'bottom': bottom_freq
}

print("\n" + "=" * 60)
print("НАЧАЛО ВЫЧИСЛЕНИЙ...")
print("=" * 60)

draw_room_and_sources(room_width, room_height, source, receiver, max_reflection_order, reflection_coeffs_freq)

print("\n" + "=" * 60)
print("ВЫЧИСЛЕНИЯ ЗАВЕРШЕНЫ")
print("=" * 60)