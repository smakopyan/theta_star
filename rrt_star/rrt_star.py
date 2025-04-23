import numpy as np
import random
import math
import matplotlib.pyplot as plt
from skimage.draw import line

# ------------------------------
# Определение узла (Node)
# ------------------------------
class Node:
    def __init__(self, point):
        self.point = point      # Точка в виде (x, y)
        self.parent = None      # Родительский узел
        self.cost = 0.0         # Стоимость от старта до данного узла

# ------------------------------
# Класс RRTStar
# ------------------------------
class RRTStar:
    def __init__(self, start, goal, grid_map, max_iter=10000, step_size=0.6, goal_sample_rate=0.2, search_radius=None):
        """
        :param start: Стартовая точка (x, y)
        :param goal: Целевая точка (x, y)
        :param grid_map: 2D numpy-массив, где 1 – препятствие, 0 – свободно
        :param max_iter: максимальное число итераций
        :param step_size: шаг продвижения дерева
        :param goal_sample_rate: вероятность выбрать целевую точку как случайную
        :param search_radius: радиус поиска для переобучения; если None, берётся как step_size * 5
        """
        self.start = Node(start)
        self.goal = Node(goal)
        self.grid_map = grid_map
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.node_list = [self.start]
        self.search_radius = search_radius if search_radius is not None else step_size * 5

    # ------------------------------
    # Генерация случайной точки
    # ------------------------------
    def get_random_point(self):
        if random.random() < self.goal_sample_rate:
            return self.goal.point
        else:
            h, w = self.grid_map.shape  # h – количество строк (y), w – количество столбцов (x)
            return (random.randint(0, w - 1), random.randint(0, h - 1))

    # ------------------------------
    # Поиск ближайшего узла по евклидовой дистанции
    # ------------------------------
    def nearest(self, random_point):
        return min(self.node_list, key=lambda node: np.linalg.norm(np.array(node.point) - np.array(random_point)))

    # ------------------------------
    # Функция "steer": продвигается от from_point к to_point на расстояние step_size
    # ------------------------------
    def steer(self, from_point, to_point):
        from_arr = np.array(from_point, dtype=float)
        to_arr = np.array(to_point, dtype=float)
        direction = to_arr - from_arr
        length = np.linalg.norm(direction)
        if length == 0:
            return from_point
        direction = direction / length
        new_arr = from_arr + self.step_size * direction
        # Округляем до целых, так как координаты в grid_map дискретные
        new_point = (int(round(new_arr[0])), int(round(new_arr[1])))
        return new_point

    # ------------------------------
    # Проверка столкновения для одной точки
    # ------------------------------
    def is_collision(self, point):
        x, y = int(point[0]), int(point[1])
        h, w = self.grid_map.shape
        if x < 0 or y < 0 or x >= w or y >= h:
            return True

        # Стандартная проверка
        if self.grid_map[y, x] == 1:
            return True

        # Доп. проверка на близость к препятствию
        neighborhood = self.grid_map[max(0, y-1):y+2, max(0, x-1):x+2]
        if np.any(neighborhood == 1):
            return True  # Близко к стене — тоже считается столкновением

        return False


    # ------------------------------
    # Проверка столкновения вдоль линии между p1 и p2
    # ------------------------------
    def is_line_collision(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        rr, cc = line(y1, x1, y2, x2)  # Получаем координаты линии (y, x)
        for y, x in zip(rr, cc):
            if self.is_collision((x, y)):
                return True
        return False

    # ------------------------------
    # Поиск узлов, находящихся в радиусе search_radius от new_node
    # ------------------------------
    def get_near_nodes(self, new_node):
        near_nodes = []
        for node in self.node_list:
            if np.linalg.norm(np.array(node.point) - np.array(new_node.point)) <= self.search_radius:
                near_nodes.append(node)
        return near_nodes

    def find_safest_path(self, paths, lambda_penalty=0.1, min_distance=3):
        best_path = None
        best_score = float('inf')

        for path in paths:
            total_penalty = sum(self.compute_obstacle_cost(point, min_distance) for point in path)
            path_length = len(path)  # Длина пути
            score = total_penalty + lambda_penalty * path_length  # Балансируем штраф и длину пути

            if score < best_score:
                best_score = score
                best_path = path

        return best_path

    def compute_obstacle_cost(self, point, min_distance=5):
        x, y = point
        if self.grid_map[y, x] == 1:
            return float('inf')  # Если точка в препятствии - бесконечный штраф
        
        # Оценка окрестности вокруг точки
        local_area = self.grid_map[max(0, y - min_distance): min(self.grid_map.shape[0], y + min_distance),
                                max(0, x - min_distance): min(self.grid_map.shape[1], x + min_distance)]
        penalty = np.sum(local_area)  # Чем больше препятствий рядом, тем выше штраф
        
        # Увеличиваем штраф за близость к стенам
        distance_weight = 50 / (1 + np.exp(-penalty / 2))  # Нелинейный штраф
        
        return distance_weight

    def choose_parent(self, new_node, near_nodes):
        best_cost = float('inf')
        best_parent = None

        for near_node in near_nodes:
            if not self.is_line_collision(near_node.point, new_node.point):
                cost = (near_node.cost + np.linalg.norm(np.array(near_node.point) - np.array(new_node.point)) +
                        self.compute_obstacle_cost(new_node.point))  # Добавляем штраф за близость к стенам
                
                if cost < best_cost:
                    best_cost = cost
                    best_parent = near_node

        if best_parent is not None:
            new_node.parent = best_parent
            new_node.cost = best_cost

    # ------------------------------
    # Ревайринг: проверка, можно ли улучшить путь до уже существующих узлов через new_node
    # ------------------------------
    def rewire(self, new_node, near_nodes):
        for near_node in near_nodes:
            if near_node == new_node.parent:
                continue
            potential_cost = new_node.cost + np.linalg.norm(np.array(new_node.point) - np.array(near_node.point))
            if potential_cost < near_node.cost:
                if not self.is_line_collision(new_node.point, near_node.point):
                    near_node.parent = new_node
                    near_node.cost = potential_cost

    # ------------------------------
    # Основной метод планирования пути
    # ------------------------------
    def plan(self):
        best_paths = []

        for i in range(self.max_iter):
            random_point = self.get_random_point()
            nearest_node = self.nearest(random_point)
            new_point = self.steer(nearest_node.point, random_point)

            if self.is_collision(new_point) or self.is_line_collision(nearest_node.point, new_point):
                continue

            new_node = Node(new_point)
            new_node.cost = nearest_node.cost + np.linalg.norm(np.array(nearest_node.point) - np.array(new_point))
            new_node.parent = nearest_node

            near_nodes = self.get_near_nodes(new_node)
            self.choose_parent(new_node, near_nodes)
            self.node_list.append(new_node)
            self.rewire(new_node, near_nodes)

            if np.linalg.norm(np.array(new_node.point) - np.array(self.goal.point)) < self.step_size:
                if not self.is_line_collision(new_node.point, self.goal.point):
                    self.goal.parent = new_node
                    self.goal.cost = new_node.cost + np.linalg.norm(np.array(new_node.point) - np.array(self.goal.point))
                    self.node_list.append(self.goal)
                    best_paths.append(self.extract_path())

                    if len(best_paths) >= 5:  # Соберём хотя бы 5 путей
                        break

        if not best_paths:
            return None

        return self.find_safest_path(best_paths)


    # ------------------------------
    # Извлечение пути от цели до старта
    # ------------------------------
    def extract_path(self):
        path = []
        node = self.goal
        while node is not None:
            if node.point != self.start.point:  # Исключаем стартовую точку
                path.append(node.point)
            node = node.parent
        path.reverse()
        return path
