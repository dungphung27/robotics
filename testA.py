import cv2
import numpy as np
import heapq
import math

# ---------------- A* Planner ----------------
class AStarPlanner:
    def __init__(self, map_img, clearance=30):
        self.original_map = map_img
        self.clearance = clearance
        self.map = self.inflate_obstacles(map_img, clearance)
        self.h, self.w = self.map.shape

        self.moves = [
            (1,0),(-1,0),(0,1),(0,-1),
            (1,1),(1,-1),(-1,1),(-1,-1)
        ]

    def inflate_obstacles(self, map_img, clearance):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (2*clearance+1, 2*clearance+1))
        inflated = cv2.dilate(255 - map_img, kernel)
        return 255 - inflated

    def in_bounds(self, x, y):
        return 0 <= x < self.w and 0 <= y < self.h

    def is_free(self, x, y):
        return self.map[y, x] > 200

    def heuristic(self, p1, p2):
        return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

    def plan(self, start, goal):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start:0}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                return self.reconstruct_path(came_from, current)

            for dx, dy in self.moves:
                nx, ny = current[0]+dx, current[1]+dy
                if not self.in_bounds(nx, ny): continue
                if not self.is_free(nx, ny): continue

                new_cost = g_score[current] + math.hypot(dx, dy)
                if (nx,ny) not in g_score or new_cost < g_score[(nx,ny)]:
                    g_score[(nx,ny)] = new_cost
                    f = new_cost + self.heuristic((nx,ny), goal)
                    heapq.heappush(open_set, (f, (nx,ny)))
                    came_from[(nx,ny)] = current
        return None

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

    # Trích waypoint mượt hơn
    def extract_waypoints(self, path, tol=2.0):
        if not path or len(path) < 2:
            return path

        waypoints = [path[0]]
        start_idx = 0

        while start_idx < len(path) - 1:
            end_idx = start_idx + 1
            while end_idx < len(path):
                x0, y0 = path[start_idx]
                x1, y1 = path[end_idx]

                # kiểm tra độ lệch max
                max_dist = 0
                for i in range(start_idx + 1, end_idx):
                    x, y = path[i]
                    if x1 == x0:
                        dist = abs(x - x0)
                    else:
                        a = (y1 - y0) / (x1 - x0)
                        b = y0 - a * x0
                        dist = abs(a*x - y + b) / math.sqrt(a*a + 1)
                    max_dist = max(max_dist, dist)

                if max_dist > tol:
                    break
                end_idx += 1

            waypoints.append(path[end_idx - 1])
            start_idx = end_idx - 1

        return waypoints


# --------------- MAIN -----------------

map_img = cv2.imread("testPx2.png", cv2.IMREAD_GRAYSCALE)
color = cv2.cvtColor(map_img, cv2.COLOR_GRAY2BGR)

start = None
goal = None
click_count = 0

planner = AStarPlanner(map_img, clearance=10)


def mouse_callback(event, x, y, flags, param):
    global start, goal, click_count, color

    if event == cv2.EVENT_LBUTTONDOWN:
        click_count += 1

        if click_count == 1:
            start = (x, y)
            print("Start =", start)
            cv2.circle(color, start, 5, (0,255,0), -1)

        elif click_count == 2:
            goal = (x, y)
            print("Goal =", goal)
            cv2.circle(color, goal, 5, (0,0,255), -1)

            # Start + Goal đã đủ → chạy A*
            print("Running A*...")
            path = planner.plan(start, goal)

            # Reset vẽ
            color = cv2.cvtColor(map_img, cv2.COLOR_GRAY2BGR)
            cv2.circle(color, start, 5, (0,255,0), -1)
            cv2.circle(color, goal, 5, (0,0,255), -1)

            if path:
                waypoints = planner.extract_waypoints(path)
                print("Waypoints:", waypoints)

                for (px,py) in path:
                    cv2.circle(color, (px,py), 1, (0,0,255), -1)

                for (wx,wy) in waypoints:
                    cv2.circle(color, (wx,wy), 5, (255,0,0), -1)
            else:
                print("No path found!")

            click_count = 0  # reset để chọn lại điểm mới

        cv2.imshow("A* Pathfinding", color)


cv2.namedWindow("A* Pathfinding")
cv2.setMouseCallback("A* Pathfinding", mouse_callback)

cv2.imshow("A* Pathfinding", color)
cv2.waitKey(0)
cv2.destroyAllWindows()