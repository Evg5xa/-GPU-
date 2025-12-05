import matplotlib.pyplot as plt
import numpy as np
import math
import time
import random
from matplotlib.path import Path


import matplotlib.pyplot as plt
import numpy as np
import math
import time
import random
from matplotlib.path import Path


# Настройка параметров ввода для удобства
def get_float_input(prompt, default=None):
    if default is not None:
        prompt += f" [по умолчанию: {default}]: "
    else:
        prompt += ": "
    try:
        user_input = input(prompt).strip()
        if user_input == "" and default is not None:
            print(f"Используется значение по умолчанию: {default}")
            return default
        return float(user_input)
    except ValueError:
        if default is not None:
            print(f"Используется значение по умолчанию: {default}")
            return default
        return 0.0


def get_int_input(prompt, default=None):
    if default is not None:
        prompt += f" [по умолчанию: {default}]: "
    else:
        prompt += ": "
    try:
        user_input = input(prompt).strip()
        if user_input == "" and default is not None:
            print(f"Используется значение по умолчанию: {default}")
            return default
        return int(user_input)
    except ValueError:
        if default is not None:
            print(f"Используется значение по умолчанию: {default}")
            return default
        return 0


def get_bool_input(prompt, default=True):
    default_str = "y" if default else "n"
    prompt += f" (y/n) [по умолчанию: {default_str}]: "

    user_input = input(prompt).strip().lower()

    if user_input == "":
        print(f"Используется значение по умолчанию: {default_str}")
        return default

    return user_input in ['y', 'yes', 'да', '1', 'true']


# Ввод параметров комнаты
print("=== ПАРАМЕТРЫ КОМНАТЫ ===")
roomEdges = get_int_input("Количество стен", 4)

# Автоматическая генерация прямоугольной комнаты или ручной ввод
auto_generate = get_bool_input("Автоматически сгенерировать прямоугольную комнату?", True)

if auto_generate:
    room_width = get_float_input("Ширина комнаты", 10.0)
    room_height = get_float_input("Высота комнаты", 8.0)
    center_x = get_float_input("Центр комнаты X", 0.0)
    center_y = get_float_input("Центр комнаты Y", 0.0)

    wallVerticesX = np.array([
        center_x - room_width / 2, center_x + room_width / 2,
        center_x + room_width / 2, center_x - room_width / 2
    ])
    wallVerticesY = np.array([
        center_y - room_height / 2, center_y - room_height / 2,
        center_y + room_height / 2, center_y + room_height / 2
    ])

    # Создаем параметры стен для прямоугольной комнаты
    wall_absorption = np.array([0, 0, 0, 0])  # Нулевое поглощение для лучших отражений
    wall_diffraction = np.array([0.05, 0.05, 0.05, 0.05])  # Коэффициенты диффракции по умолчанию
else:
    wallVerticesX = np.array([])
    wallVerticesY = np.array([])
    wall_absorption = np.array([])
    wall_diffraction = np.array([])

    for i in range(roomEdges):
        print(f"Стена {i + 1} из {roomEdges}:")
        wallVerticesX = np.append(wallVerticesX, get_float_input(f"  X координата"))
        wallVerticesY = np.append(wallVerticesY, get_float_input(f"  Y координата"))
        absorption = get_float_input(f"  Коэффициент поглощения (0-1)", 0.0)
        diffraction = get_float_input(f"  Коэффициент диффракции (0-1)", 0.05)

        # Проверка, что сумма коэффициентов не превышает 1
        if absorption + diffraction > 1.0:
            print(f"  Внимание: сумма коэффициентов поглощения и диффракции превышает 1.0!")
            print(f"  Установлены значения: поглощение={absorption:.2f}, диффракция={1.0-absorption:.2f}")
            diffraction = 1.0 - absorption

        wall_absorption = np.append(wall_absorption, absorption)
        wall_diffraction = np.append(wall_diffraction, diffraction)

# Параметры лучей
print("\n=== ПАРАМЕТРЫ ЛУЧЕЙ ===")
rayCount = get_int_input("Количество лучей", 360)
maxCollisions = get_int_input("Максимальное количество отражений", 10)
startEnergy = get_float_input("Начальная энергия луча", 5.0)
energyLoss = get_float_input("Потеря энергии при отражении (0-1)", 0.05)
speed_of_sound = get_float_input("Скорость звука:", 343)

# Параметры распространения звука
print("\n=== ПАРАМЕТРЫ РАСПРОСТРАНЕНИЯ ЗВУКА ===")
distance_attenuation = get_bool_input("Учитывать затухание с расстоянием?", False)
distance_loss_factor = 0.0
if distance_attenuation:
    distance_loss_factor = get_float_input("Коэффициент затухания с расстоянием (дБ/м)", 0.1)
    distance_loss_linear = 10 ** (-distance_loss_factor / 10)

# Источник
print("\n=== ИСТОЧНИК ЗВУКА ===")
sourceX = get_float_input("Позиция источника X", 0.0)
sourceY = get_float_input("Позиция источника Y", 0.0)

# Автоматическая настройка углов или ручной ввод
auto_angles = get_bool_input("Использовать полный круг для лучей?", True)
if auto_angles:
    minAngle = 0.0
    maxAngle = 2 * math.pi
else:
    minAngle = get_float_input("Минимальный угол (0-2pi)", 0.0)
    maxAngle = get_float_input("Максимальный угол (0-2pi, >=minAngle)", 2 * math.pi)

# Микрофоны
print("\n=== МИКРОФОНЫ ===")
micCount = get_int_input("Количество микрофонов", 2)
microphones = []
for i in range(micCount):
    print(f"Микрофон {i + 1}:")
    micX = get_float_input("  Позиция X", 2.0 * (i + 1))
    micY = get_float_input("  Позиция Y", 1.0)
    micRadius = get_float_input("  Радиус", 0.5)
    microphones.append({
        'position': (micX, micY),
        'radius': micRadius,
        'absorbed_energy': 0.0,
        'ray_count': 0,
        'energy_history': [],
        'time_history': [],
        'distance_history': [],
        'bounce_history': []
    })

wallVertices = np.array([wallVerticesX, wallVerticesY])

# Параметры энергетической карты
print("\n=== ПАРАМЕТРЫ ЭНЕРГЕТИЧЕСКОЙ КАРТЫ ===")
create_energy_map = get_bool_input("Создать энергетическую карту?", True)
grid_size = 0.2
if create_energy_map:
    grid_size = get_float_input("Размер ячейки сетки", 0.2)

# Визуализация
print("\n=== НАСТРОЙКИ ВИЗУАЛИЗАЦИИ ===")
show_energy_color = get_bool_input("Показывать энергию цветом?", True)
show_bounce_points = get_bool_input("Показывать точки отражения?", True)
show_energy_text = get_bool_input("Показывать энергию текстом?", True)
show_ray_paths = get_bool_input("Показывать пути лучей?", True)

# Функции для расчёта энергии с учётом расстояния
def calculate_distance_energy(initial_energy, distance, distance_loss_linear):
    """Расчёт энергии с учётом затухания на расстоянии"""
    if distance_attenuation:
        distance_effect = 1.0 / (1.0 + distance)
        attenuation = distance_loss_linear ** distance
        return initial_energy * distance_effect * attenuation
    else:
        return initial_energy / (1.0 + distance ** 2)


def calculate_total_distance(path_x, path_y):
    """Расчёт общего расстояния, пройденного лучом"""
    total_dist = 0.0
    for i in range(len(path_x) - 1):
        dx = path_x[i + 1] - path_x[i]
        dy = path_y[i + 1] - path_y[i]
        total_dist += math.sqrt(dx ** 2 + dy ** 2)
    return total_dist


# Функции геометрии
def to_angle(sin_a, cos_a):
    angle = math.atan2(sin_a, cos_a)
    if angle < 0:
        angle += 2 * math.pi
    return angle


def reflect_ray(normal_x, normal_y, incident_sin, incident_cos):
    normal_length = math.sqrt(normal_x ** 2 + normal_y ** 2)
    if normal_length > 0:
        normal_x /= normal_length
        normal_y /= normal_length

    dot_product = incident_cos * normal_x + incident_sin * normal_y
    reflect_cos = incident_cos - 2 * dot_product * normal_x
    reflect_sin = incident_sin - 2 * dot_product * normal_y

    reflect_length = math.sqrt(reflect_sin ** 2 + reflect_cos ** 2)
    if reflect_length > 0:
        reflect_sin /= reflect_length
        reflect_cos /= reflect_length

    return reflect_sin, reflect_cos


def line_intersection(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    if 0 <= t <= 1 and 0 <= u <= 1:
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return (x, y)
    return None


def point_in_microphone(x, y, microphones):
    for mic in microphones:
        mic_x, mic_y = mic['position']
        distance = math.sqrt((x - mic_x) ** 2 + (y - mic_y) ** 2)
        if distance <= mic['radius']:
            return mic
    return None


def find_next_intersection(x, y, sin_a, cos_a, wall_vertices, microphones, wall_absorption, wall_diffraction):
    ray_end_x = x + cos_a * 1000
    ray_end_y = y + sin_a * 1000

    closest_intersection = None
    closest_distance = float('inf')
    wall_normal = None
    intersection_type = None
    hit_microphone = None
    wall_index = None

    # Проверка пересечения с микрофонами
    for mic in microphones:
        mic_x, mic_y = mic['position']
        radius = mic['radius']

        dx = mic_x - x
        dy = mic_y - y
        proj = dx * cos_a + dy * sin_a

        closest_x = x + cos_a * proj
        closest_y = y + sin_a * proj
        dist_to_center = math.sqrt((closest_x - mic_x) ** 2 + (closest_y - mic_y) ** 2)

        if dist_to_center <= radius:
            if dist_to_center == radius:
                t = proj
                if t >= 0:
                    intersection = (closest_x, closest_y)
                    dist = math.sqrt((intersection[0] - x) ** 2 + (intersection[1] - y) ** 2)
                    if dist > 1e-8 and dist < closest_distance:
                        closest_distance = dist
                        closest_intersection = intersection
                        hit_microphone = mic
                        intersection_type = 'microphone'
            else:
                chord_half_length = math.sqrt(radius ** 2 - dist_to_center ** 2)
                t1 = proj - chord_half_length
                t2 = proj + chord_half_length

                t = None
                if t1 >= 0 and t2 >= 0:
                    t = min(t1, t2)
                elif t1 >= 0:
                    t = t1
                elif t2 >= 0:
                    t = t2

                if t is not None:
                    intersection = (x + cos_a * t, y + sin_a * t)
                    dist = math.sqrt((intersection[0] - x) ** 2 + (intersection[1] - y) ** 2)
                    if dist > 1e-8 and dist < closest_distance:
                        closest_distance = dist
                        closest_intersection = intersection
                        hit_microphone = mic
                        intersection_type = 'microphone'

    # Проверка пересечения со стенами
    n_walls = wall_vertices.shape[1]
    for i in range(n_walls):
        p1 = (wall_vertices[0, i], wall_vertices[1, i])
        p2 = (wall_vertices[0, (i + 1) % n_walls], wall_vertices[1, (i + 1) % n_walls])

        intersection = line_intersection((x, y), (ray_end_x, ray_end_y), p1, p2)

        if intersection:
            dist = math.sqrt((intersection[0] - x) ** 2 + (intersection[1] - y) ** 2)
            if dist > 1e-8 and dist < closest_distance:
                closest_distance = dist
                closest_intersection = intersection
                intersection_type = 'wall'
                wall_index = i

                wall_dx = p2[0] - p1[0]
                wall_dy = p2[1] - p1[1]
                normal_x = wall_dy
                normal_y = -wall_dx
                wall_normal = (normal_x, normal_y)

    return closest_intersection, wall_normal, intersection_type, hit_microphone, wall_index


def get_diffraction_angle(normal_x, normal_y, current_sin, current_cos):
    """
    Генерирует случайный угол для диффракции
    """
    normal_length = math.sqrt(normal_x ** 2 + normal_y ** 2)
    if normal_length > 0:
        normal_x /= normal_length
        normal_y /= normal_length

    normal_angle = math.atan2(normal_y, normal_x)
    incident_angle = math.atan2(current_sin, current_cos)

    angle_variation = random.uniform(-math.pi/2, math.pi/2)
    diffracted_angle = normal_angle + angle_variation
    diffracted_angle = diffracted_angle % (2 * math.pi)

    return math.sin(diffracted_angle), math.cos(diffracted_angle)


def handle_wall_interaction(current_energy, wall_index, wall_absorption, wall_diffraction, normal_x, normal_y, current_sin, current_cos):
    """
    Обработка взаимодействия луча со стеной
    """
    absorption_prob = wall_absorption[wall_index]
    diffraction_prob = wall_diffraction[wall_index]
    reflection_prob = 1.0 - absorption_prob - diffraction_prob

    rand_val = random.random()

    if rand_val < absorption_prob:
        return 'absorbed', 0.0, 0.0, 0.0
    elif rand_val < absorption_prob + diffraction_prob:
        diffract_sin, diffract_cos = get_diffraction_angle(normal_x, normal_y, current_sin, current_cos)
        return 'diffracted', current_energy * (1.0 - energyLoss), diffract_sin, diffract_cos
    else:
        reflect_sin, reflect_cos = reflect_ray(normal_x, normal_y, current_sin, current_cos)
        return 'reflected', current_energy * (1.0 - energyLoss), reflect_sin, reflect_cos


def trace_ray(x, y, sin_a, cos_a, initial_energy, wall_vertices, max_bounces, attenuation, microphones, ray_id, wall_absorption, wall_diffraction):
    current_x, current_y = x, y
    current_sin, current_cos = sin_a, cos_a
    current_energy = initial_energy

    ray_path_x = [current_x]
    ray_path_y = [current_y]
    energies = [current_energy]
    bounce_count = 0
    total_distance = 0.0

    for bounce in range(max_bounces):
        if current_energy < 0.001:
            break

        intersection, wall_normal, intersection_type, hit_microphone, wall_index = find_next_intersection(
            current_x, current_y, current_sin, current_cos, wall_vertices, microphones, wall_absorption, wall_diffraction)

        if intersection is None:
            break

        segment_distance = math.sqrt((intersection[0] - current_x) ** 2 + (intersection[1] - current_y) ** 2)
        total_distance += segment_distance

        segment_energy = calculate_distance_energy(current_energy, segment_distance,
                                                   distance_loss_linear if distance_attenuation else 1)

        ray_path_x.append(intersection[0])
        ray_path_y.append(intersection[1])

        if intersection_type == 'microphone' and hit_microphone:
            arrival_time = total_distance / speed_of_sound
            hit_microphone['absorbed_energy'] += segment_energy
            hit_microphone['ray_count'] += 1
            hit_microphone['energy_history'].append(segment_energy)
            hit_microphone['time_history'].append(arrival_time)
            hit_microphone['distance_history'].append(total_distance)
            hit_microphone['bounce_history'].append(bounce_count)
            energies.append(segment_energy)
            break

        current_x, current_y = intersection

        if intersection_type == 'wall' and wall_normal:
            interaction_type, new_energy, new_sin, new_cos = handle_wall_interaction(
                segment_energy, wall_index, wall_absorption, wall_diffraction,
                wall_normal[0], wall_normal[1], current_sin, current_cos)

            if interaction_type == 'absorbed':
                energies.append(new_energy)
                break
            elif interaction_type == 'diffracted':
                current_sin, current_cos = new_sin, new_cos
                current_energy = new_energy
                bounce_count += 1
            elif interaction_type == 'reflected':
                current_sin, current_cos = new_sin, new_cos
                current_energy = new_energy
                bounce_count += 1

        energies.append(current_energy)

    return ray_path_x, ray_path_y, energies, bounce_count, total_distance


# ФУНКЦИИ ДЛЯ СОЗДАНИЯ ЭНЕРГЕТИЧЕСКОЙ КАРТЫ
def analyze_ray_paths(ray_paths):
    """Анализ путей лучей для отладки"""
    print(f"\n=== АНАЛИЗ ПУТЕЙ ЛУЧЕЙ ===")
    multi_segment_rays = 0
    total_segments = 0
    max_segments = 0

    for i, (ray_x, ray_y, energies) in enumerate(ray_paths):
        segments = len(ray_x) - 1
        total_segments += segments
        if segments > 1:
            multi_segment_rays += 1
        if segments > max_segments:
            max_segments = segments

    print(f"Всего лучей: {len(ray_paths)}")
    print(f"Лучей с отражениями: {multi_segment_rays}")
    print(f"Прямых лучей: {len(ray_paths) - multi_segment_rays}")
    print(f"Максимум сегментов в луче: {max_segments}")
    print(f"Среднее сегментов на луч: {total_segments / len(ray_paths):.2f}")


def create_energy_map_numpy(vertices, ray_paths, grid_size=0.2):
    """
    Создает энергетическую карту помещения
    """
    poly_x, poly_y = vertices[0], vertices[1]

    min_x, max_x = np.floor(np.min(poly_x)), np.ceil(np.max(poly_x))
    min_y, max_y = np.floor(np.min(poly_y)), np.ceil(np.max(poly_y))

    x_coords = np.arange(min_x, max_x + grid_size, grid_size)
    y_coords = np.arange(min_y, max_y + grid_size, grid_size)

    energy_matrix = np.zeros((len(y_coords), len(x_coords)))

    polygon = np.column_stack((poly_x, poly_y))
    path = Path(polygon)

    xx, yy = np.meshgrid(x_coords, y_coords)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    inside_mask = path.contains_points(grid_points)
    inside_mask = inside_mask.reshape(xx.shape)

    for ray_path in ray_paths:
        if ray_path is None:
            continue

        ray_x, ray_y, energies = ray_path
        if len(ray_x) < 2:
            continue

        for i in range(len(ray_x) - 1):
            x1, y1 = ray_x[i], ray_y[i]
            x2, y2 = ray_x[i + 1], ray_y[i + 1]
            segment_energy = energies[i] if i < len(energies) else energies[-1]

            energy_matrix = add_energy_to_cells(x1, y1, x2, y2, segment_energy,
                                              x_coords, y_coords, energy_matrix,
                                              grid_size, inside_mask)

    return energy_matrix, x_coords, y_coords, inside_mask


def add_energy_to_cells(x1, y1, x2, y2, energy, x_coords, y_coords, energy_matrix, grid_size, inside_mask):
    """
    Распределение энергии по ячейкам с высокой чувствительностью
    """
    segment_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if segment_length == 0:
        return energy_matrix

    cells = find_cells_along_ray(x1, y1, x2, y2, x_coords, y_coords, grid_size)

    if cells:
        # Высокое усиление для лучшей видимости
        amplified_energy = energy
        energy_per_cell = amplified_energy / len(cells)

        for i, j in cells:
            if (0 <= i < len(y_coords) and 0 <= j < len(x_coords) and
                inside_mask[i, j]):
                energy_matrix[i, j] += energy_per_cell

    return energy_matrix


def find_cells_along_ray(x1, y1, x2, y2, x_coords, y_coords, grid_size):
    """
    Находит все ячейки, через которые проходит луч
    """
    cells = set()

    dx = x2 - x1
    dy = y2 - y1
    length = np.sqrt(dx*dx + dy*dy)

    if length == 0:
        return cells

    # Высокое разрешение для лучшего покрытия
    step = grid_size / 4
    steps = max(1, int(length / step) + 1)

    for t in np.linspace(0, 1, steps * 4):
        x = x1 + t * dx
        y = y1 + t * dy

        i = np.searchsorted(y_coords, y) - 1
        j = np.searchsorted(x_coords, x) - 1

        if 0 <= i < len(y_coords) and 0 <= j < len(x_coords):
            cells.add((i, j))

    return cells


# ФУНКЦИЯ ДЛЯ ВИЗУАЛИЗАЦИИ ПУТЕЙ ЛУЧЕЙ
def plot_ray_paths(ray_paths, wall_vertices, microphones, source_position):
    """
    Визуализация путей лучей на отдельном графике
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Рисуем контур помещения
    wall_x = np.append(wall_vertices[0], wall_vertices[0][0])
    wall_y = np.append(wall_vertices[1], wall_vertices[1][0])
    ax.plot(wall_x, wall_y, 'k-', linewidth=3, label='Помещение')

    # Рисуем источник
    ax.scatter(source_position[0], source_position[1], c='red', s=200, marker='*',
               label='Источник', edgecolors='white', linewidth=2, zorder=10)

    # Рисуем микрофоны
    for i, mic in enumerate(microphones):
        mic_x, mic_y = mic['position']
        circle = plt.Circle((mic_x, mic_y), mic['radius'], color='blue', alpha=0.3,
                          label=f'Микрофон {i+1}' if i == 0 else "")
        ax.add_patch(circle)
        ax.scatter(mic_x, mic_y, c='blue', s=100, marker='s',
                  edgecolors='white', linewidth=2, zorder=5)

    # Рисуем пути лучей с цветовой кодировкой по энергии
    all_energies = []
    for ray_x, ray_y, energies in ray_paths:
        if len(energies) > 0:
            all_energies.extend(energies)

    if all_energies:
        max_energy = max(all_energies)
        min_energy = min(all_energies)
    else:
        max_energy = 1.0
        min_energy = 0.0

    # Ограничиваем количество отображаемых лучей для лучшей читаемости
    max_rays_to_show = min(360, len(ray_paths))
    step = max(1, len(ray_paths) // max_rays_to_show)

    rays_shown = 0
    for i in range(0, len(ray_paths), step):
        ray_x, ray_y, energies = ray_paths[i]
        if len(ray_x) < 2:
            continue

        # Используем среднюю энергию луча для цвета
        avg_energy = np.mean(energies) if len(energies) > 0 else 0
        normalized_energy = (avg_energy - min_energy) / (max_energy - min_energy + 1e-8)

        # Цвет от синего (низкая энергия) к красному (высокая энергия)
        color = plt.cm.plasma(normalized_energy)

        # Рисуем путь луча
        ax.plot(ray_x, ray_y, '-', color=color, alpha=0.7, linewidth=1.5)

        # Рисуем точки отражения
        if show_bounce_points and len(ray_x) > 2:
            ax.scatter(ray_x[1:-1], ray_y[1:-1], c='green', s=20, alpha=0.6, marker='o')

        rays_shown += 1
        if rays_shown >= max_rays_to_show:
            break

    # Добавляем colorbar
    if all_energies:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma,
                                 norm=plt.Normalize(min_energy, max_energy))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Энергия луча', fontsize=12)

    ax.set_xlabel('X координата (м)', fontsize=12)
    ax.set_ylabel('Y координата (м)', fontsize=12)
    ax.set_title('Визуализация путей звуковых лучей', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Статистика
    stats_text = f'Всего лучей: {len(ray_paths)}\nПоказано лучей: {rays_shown}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    return fig, ax


# ЗАПУСК МОДЕЛИРОВАНИЯ
start_time = time.time()

ray_paths = []

# Сброс статистики микрофонов
for mic in microphones:
    mic['absorbed_energy'] = 0.0
    mic['ray_count'] = 0
    mic['energy_history'] = []
    mic['time_history'] = []
    mic['distance_history'] = []
    mic['bounce_history'] = []

# Трассировка всех лучей
print("Запуск трассировки лучей...")
for i in range(rayCount):
    angle = minAngle + i * (maxAngle - minAngle) / max(1, rayCount - 1)
    ray_x, ray_y, energies, bounces, total_dist = trace_ray(
        sourceX, sourceY, math.sin(angle), math.cos(angle),
        startEnergy, wallVertices, maxCollisions, energyLoss,
        microphones, i, wall_absorption, wall_diffraction
    )

    # Сохраняем ВСЕ пути лучей
    if create_energy_map or show_ray_paths:
        ray_paths.append((ray_x, ray_y, energies))

print(f"Трассировка завершена за {time.time() - start_time:.2f} сек")
print(f"Собрано путей лучей: {len(ray_paths)}")

# ВИЗУАЛИЗАЦИЯ ПУТЕЙ ЛУЧЕЙ
if show_ray_paths and ray_paths:
    print("\nСоздание визуализации путей лучей...")
    fig_rays, ax_rays = plot_ray_paths(ray_paths, wallVertices, microphones, (sourceX, sourceY))
    plt.show()

# СОЗДАНИЕ ЭНЕРГЕТИЧЕСКОЙ КАРТЫ
if create_energy_map and ray_paths:
    print("Создание энергетической карты...")
    energy_start_time = time.time()

    # Анализ путей лучей
    analyze_ray_paths(ray_paths)

    # Создаем энергетическую карту
    energy_matrix, x_coords, y_coords, inside_mask = create_energy_map_numpy(
        [wallVerticesX, wallVerticesY], ray_paths, grid_size
    )

    # Высокое усиление для лучшей видимости
    amplification_factor = 100.0
    energy_matrix = np.log10(energy_matrix * amplification_factor)

    print(f"Энергетическая карта создана за {time.time() - energy_start_time:.2f} сек")
    print(f"Размер матрицы: {energy_matrix.shape}")
    print(f"Суммарная энергия: {np.sum(energy_matrix):.2f}")
    print(f"Максимальная энергия в ячейке: {np.max(energy_matrix):.2f}")

    # ВИЗУАЛИЗАЦИЯ
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # График 1: 2D энергетическая карта
    plt.sca(ax1)

    if np.max(energy_matrix) > 0:
        vmax = np.percentile(energy_matrix, 95)
        if vmax == 0:
            vmax = np.max(energy_matrix)

        im = plt.imshow(energy_matrix,
                       extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]],
                       origin='lower', cmap='hot', aspect='auto',
                       vmin=0, vmax=vmax)
        plt.colorbar(im, ax=ax1, label='Энергия')

    # Рисуем контур помещения
    plt.plot(np.append(wallVerticesX, wallVerticesX[0]),
             np.append(wallVerticesY, wallVerticesY[0]), 'w-', linewidth=3, label='Помещение')

    # Рисуем источник и микрофоны
    plt.scatter(sourceX, sourceY, c='red', s=150, marker='*', label='Источник',
               edgecolors='white', linewidth=2, zorder=5)

    for i, mic in enumerate(microphones):
        mic_x, mic_y = mic['position']
        plt.scatter(mic_x, mic_y, c='blue', s=80, marker='s',
                   label=f'Mic{i+1}' if i == 0 else "",
                   edgecolors='white', linewidth=2, zorder=5)

    plt.xlabel('X координата (м)')
    plt.ylabel('Y координата (м)')
    plt.title('Энергетическая карта помещения\n(все лучи, включая отраженные)', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # График 2: Гистограмма распределения энергии
    plt.sca(ax2)

    energy_inside = energy_matrix[inside_mask]

    if len(energy_inside) > 0 and np.max(energy_inside) > 0:
        non_zero_energy = energy_inside[energy_inside > 0]

        if len(non_zero_energy) > 0:
            plt.hist(non_zero_energy, bins=50, alpha=0.7,
                    color='orange', edgecolor='black', linewidth=0.5)

            mean_energy = np.mean(non_zero_energy)
            plt.axvline(mean_energy, color='red', linestyle='--', linewidth=2,
                       label=f'Среднее: {mean_energy:.2f}')

            plt.xlabel('Энергия в ячейке')
            plt.ylabel('Количество ячеек')
            plt.title('Распределение энергии по ячейкам', fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.legend()

            stats_text = f'Всего ячеек: {len(energy_inside):,}\n' \
                        f'Не нулевых: {len(non_zero_energy):,}\n' \
                        f'Максимум: {np.max(non_zero_energy):.2f}\n' \
                        f'Усиление: ×{amplification_factor}'

            plt.text(0.95, 0.95, stats_text, transform=ax2.transAxes,
                    ha='right', va='top', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

    plt.tight_layout()
    plt.show()

    # СТАТИСТИКА
    energy_inside = energy_matrix[inside_mask]
    if len(energy_inside) > 0:
        print(f"\n=== СТАТИСТИКА ЭНЕРГЕТИЧЕСКОЙ КАРТЫ ===")
        print(f"Общая энергия в помещении: {np.sum(energy_inside):.2f}")
        print(f"Средняя энергия на ячейку: {np.mean(energy_inside):.4f}")
        print(f"Максимальная энергия в ячейке: {np.max(energy_inside):.2f}")
        print(f"Количество ячеек с энергией: {np.sum(energy_inside > 0):,}")
        print(f"Общее количество ячеек: {len(energy_inside):,}")

else:
    print("Энергетическая карта не создана")

# Вывод результатов по микрофонам
print(f"\n=== РЕЗУЛЬТАТЫ ПО МИКРОФОНАМ ===")
for i, mic in enumerate(microphones):
    efficiency = mic['absorbed_energy'] / (rayCount * startEnergy) * 100
    print(f"Микрофон {i + 1}:")
    print(f"  Поглощенная энергия: {mic['absorbed_energy']:.4f}")
    print(f"  Принято лучей: {mic['ray_count']}")
    print(f"  Эффективность: {efficiency:.1f}%")