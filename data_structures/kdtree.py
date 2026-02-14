class Node:
    def __init__(self, point):
        self.point = point
        self.left = None
        self.right = None

class KDTree:
    def __init__(self, dim):
        self.dim = dim
        self.root = None

    def _insert(self, node, point, depth=0):
        if node is None:
            return Node(point)
        
        # Calculate current dimension
        cd = depth % self.dim
        
        # Compare and recurse
        if point[cd] < node.point[cd]:
            node.left = self._insert(node.left, point, depth + 1)
        else:
            node.right = self._insert(node.right, point, depth + 1)
        
        return node
    
    def insert(self, point):
        if len(point) != self.dim:
            raise ValueError(f"Point must have {self.dim} dimensions")
        self.root = self._insert(self.root, point, 0)

    def _search(self, node, point, depth=0):
        if node is None:
            return None
        if node.point == point:
            return node
        
        cd = depth % self.dim
        
        if point[cd] < node.point[cd]:
            return self._search(node.left, point, depth + 1)
        else:
            return self._search(node.right, point, depth + 1)
    
    def search(self, point):
        if len(point) != self.dim:
            raise ValueError(f"Point must have {self.dim} dimensions")
        return self._search(self.root, point, 0)
    
    def _find_min(self, node, d, depth=0):
        if node is None:
            return None
        
        cd = depth % self.dim
        
        if cd == d:
            if node.left is None:
                return node
            return self._find_min(node.left, d, depth + 1)
        
        left_min = self._find_min(node.left, d, depth + 1)
        right_min = self._find_min(node.right, d, depth + 1)
        
        return self._min_node(node, left_min, right_min, d)
    
    def _min_node(self, x, y, z, d):
        res = x
        if y is not None and (res is None or y.point[d] < res.point[d]):
            res = y
        if z is not None and (res is None or z.point[d] < res.point[d]):
            res = z
        return res
    
    def _delete(self, node, point, depth=0):
        if node is None:
            return None
        
        cd = depth % self.dim
        
        if node.point == point:
            if node.right is not None:
                min_node = self._find_min(node.right, cd, depth + 1)
                node.point = min_node.point
                node.right = self._delete(node.right, min_node.point, depth + 1)
            elif node.left is not None:
                min_node = self._find_min(node.left, cd, depth + 1)
                node.point = min_node.point
                node.right = self._delete(node.left, min_node.point, depth + 1)
                node.left = None
            else:
                return None
            return node
        
        if point[cd] < node.point[cd]:
            node.left = self._delete(node.left, point, depth + 1)
        else:
            node.right = self._delete(node.right, point, depth + 1)
        
        return node
    
    def delete(self, point):
        if len(point) != self.dim:
            raise ValueError(f"Point must have {self.dim} dimensions")
        self.root = self._delete(self.root, point, 0)
    
    def _distance(self, p1, p2):
        return sum((a - b) ** 2 for a, b in zip(p1, p2)) ** 0.5
    
    def _nearest_neighbor(self, node, point, depth=0, best=None, best_dist=float('inf')):
        if node is None:
            return best, best_dist
        
        dist = self._distance(node.point, point)
        if dist < best_dist:
            best = node
            best_dist = dist
        
        cd = depth % self.dim
        
        if point[cd] < node.point[cd]:
            best, best_dist = self._nearest_neighbor(node.left, point, depth + 1, best, best_dist)
            if point[cd] + best_dist >= node.point[cd]:
                best, best_dist = self._nearest_neighbor(node.right, point, depth + 1, best, best_dist)
        else:
            best, best_dist = self._nearest_neighbor(node.right, point, depth + 1, best, best_dist)
            if point[cd] - best_dist <= node.point[cd]:
                best, best_dist = self._nearest_neighbor(node.left, point, depth + 1, best, best_dist)
        
        return best, best_dist
    
    def nearest_neighbor(self, point):
        if len(point) != self.dim:
            raise ValueError(f"Point must have {self.dim} dimensions")
        node, dist = self._nearest_neighbor(self.root, point, 0)
        return node.point if node else None
    
    def _range_search(self, node, lower, upper, depth=0, result=None):
        if result is None:
            result = []
        
        if node is None:
            return result
        
        # Check if point is in range
        if all(l <= p <= u for l, p, u in zip(lower, node.point, upper)):
            result.append(node.point)
        
        cd = depth % self.dim
        
        if lower[cd] <= node.point[cd]:
            self._range_search(node.left, lower, upper, depth + 1, result)
        if upper[cd] >= node.point[cd]:
            self._range_search(node.right, lower, upper, depth + 1, result)
        
        return result
    
    def range_search(self, lower, upper):
        if len(lower) != self.dim or len(upper) != self.dim:
            raise ValueError(f"Points must have {self.dim} dimensions")
        return self._range_search(self.root, lower, upper, 0)
        
    def _pre_order(self, node):
        if node is not None:
            print(node.point)
            self._pre_order(node.left)
            self._pre_order(node.right)
        
    def pre_order(self):
        self._pre_order(self.root)

import random
import time

def generate_random_points(n, dim, min_val=0, max_val=1000):
    return [tuple(random.uniform(min_val, max_val) for _ in range(dim)) for _ in range(n)]

def naive_nearest_neighbor(points, target):
    min_dist = float('inf')
    nearest = None
    for p in points:
        dist = sum((a - b) ** 2 for a, b in zip(p, target)) ** 0.5
        if dist < min_dist:
            min_dist = dist
            nearest = p
    return nearest

def naive_range_search(points, lower, upper):
    result = []
    for p in points:
        if all(l <= coord <= u for l, coord, u in zip(lower, p, upper)):
            result.append(p)
    return result

def benchmark_insertion(n, dim):
    points = generate_random_points(n, dim)
    
    # KDTree insertion
    start = time.time()
    kdtree = KDTree(dim)
    for p in points:
        kdtree.insert(p)
    end = time.time()
    kdtree_time = (end - start) * 1000
    
    # List insertion
    start = time.time()
    lst = []
    for p in points:
        lst.append(p)
    end = time.time()
    list_time = (end - start) * 1000
    
    print(f"Insertion ({n} points, {dim}D):")
    print(f"  KDTree: {kdtree_time:10.2f} ms")
    print(f"  List:   {list_time:10.2f} ms")
    print(f"  Ratio (KDTree/List): {kdtree_time / list_time:.2f}x\n")

def benchmark_search(n, dim):
    points = generate_random_points(n, dim)
    
    # Build KDTree
    kdtree = KDTree(dim)
    for p in points:
        kdtree.insert(p)
    
    # Build list
    lst = points.copy()
    
    # Search for existing points
    search_points = random.sample(points, min(100, n))
    
    # KDTree search
    start = time.time()
    for p in search_points:
        kdtree.search(p)
    end = time.time()
    kdtree_time = (end - start) * 1000
    
    # List search
    start = time.time()
    for p in search_points:
        p in lst
    end = time.time()
    list_time = (end - start) * 1000
    
    print(f"Search ({len(search_points)} searches in {n} points, {dim}D):")
    print(f"  KDTree: {kdtree_time:10.2f} ms")
    print(f"  List:   {list_time:10.2f} ms")
    print(f"  Ratio (KDTree/List): {kdtree_time / list_time:.2f}x\n")

def benchmark_nearest_neighbor(n, dim):
    points = generate_random_points(n, dim)
    
    # Build KDTree
    kdtree = KDTree(dim)
    for p in points:
        kdtree.insert(p)
    
    # Build list
    lst = points.copy()
    
    # Query points
    queries = generate_random_points(100, dim)
    
    # KDTree nearest neighbor
    start = time.time()
    for q in queries:
        kdtree.nearest_neighbor(q)
    end = time.time()
    kdtree_time = (end - start) * 1000
    
    # Naive nearest neighbor
    start = time.time()
    for q in queries:
        naive_nearest_neighbor(lst, q)
    end = time.time()
    naive_time = (end - start) * 1000
    
    print(f"Nearest Neighbor ({len(queries)} queries in {n} points, {dim}D):")
    print(f"  KDTree: {kdtree_time:10.2f} ms")
    print(f"  Naive:  {naive_time:10.2f} ms")
    print(f"  KDTree speedup: {naive_time / kdtree_time:.2f}x\n")

def benchmark_range_search(n, dim):
    points = generate_random_points(n, dim)
    
    # Build KDTree
    kdtree = KDTree(dim)
    for p in points:
        kdtree.insert(p)
    
    # Build list
    lst = points.copy()
    
    # Define range queries (10% of space)
    num_queries = 50
    queries = []
    for _ in range(num_queries):
        center = tuple(random.uniform(0, 1000) for _ in range(dim))
        radius = 50
        lower = tuple(max(0, c - radius) for c in center)
        upper = tuple(min(1000, c + radius) for c in center)
        queries.append((lower, upper))
    
    # KDTree range search
    start = time.time()
    for lower, upper in queries:
        kdtree.range_search(lower, upper)
    end = time.time()
    kdtree_time = (end - start) * 1000
    
    # Naive range search
    start = time.time()
    for lower, upper in queries:
        naive_range_search(lst, lower, upper)
    end = time.time()
    naive_time = (end - start) * 1000
    
    print(f"Range Search ({len(queries)} queries in {n} points, {dim}D):")
    print(f"  KDTree: {kdtree_time:10.2f} ms")
    print(f"  Naive:  {naive_time:10.2f} ms")
    print(f"  KDTree speedup: {naive_time / kdtree_time:.2f}x\n")

def benchmark_delete(n, dim):
    points = generate_random_points(n, dim)
    
    # Build KDTree
    kdtree = KDTree(dim)
    for p in points:
        kdtree.insert(p)
    
    # Build list
    lst = points.copy()
    
    # Delete some points
    delete_points = random.sample(points, min(100, n))
    
    # KDTree delete
    start = time.time()
    for p in delete_points:
        kdtree.delete(p)
    end = time.time()
    kdtree_time = (end - start) * 1000
    
    # List delete
    start = time.time()
    for p in delete_points:
        if p in lst:
            lst.remove(p)
    end = time.time()
    list_time = (end - start) * 1000
    
    print(f"Delete ({len(delete_points)} deletions from {n} points, {dim}D):")
    print(f"  KDTree: {kdtree_time:10.2f} ms")
    print(f"  List:   {list_time:10.2f} ms")
    print(f"  Ratio (KDTree/List): {kdtree_time / list_time:.2f}x\n")

def benchmark_dimensional_scaling(n):
    dims = [2, 3, 5, 10]
    
    print(f"Dimensional Scaling ({n} points):")
    for dim in dims:
        points = generate_random_points(n, dim)
        kdtree = KDTree(dim)
        
        start = time.time()
        for p in points:
            kdtree.insert(p)
        end = time.time()
        insert_time = (end - start) * 1000
        
        queries = generate_random_points(100, dim)
        start = time.time()
        for q in queries:
            kdtree.nearest_neighbor(q)
        end = time.time()
        nn_time = (end - start) * 1000
        
        print(f"  {dim}D: Insert={insert_time:6.2f}ms, NN={nn_time:6.2f}ms")
    print()

if __name__ == '__main__':
    print("=" * 50)
    print("      KDTREE vs NAIVE LIST BENCHMARK")
    print("=" * 50)
    print()
    
    sizes = [100, 500, 1000, 2000]
    dim = 2
    
    for n in sizes:
        print("=" * 50)
        print(f"  Dataset Size: {n} points, {dim}D")
        print("=" * 50)
        print()
        
        benchmark_insertion(n, dim)
        benchmark_search(n, dim)
        benchmark_nearest_neighbor(n, dim)
        benchmark_range_search(n, dim)
        benchmark_delete(n, dim)
    
    print("=" * 50)
    print("  Testing dimensional scaling")
    print("=" * 50)
    print()
    benchmark_dimensional_scaling(1000)
    
    print("=" * 50)
    print("         BENCHMARK COMPLETED")
    print("=" * 50)
    print()
    
    print("Key Observations:")
    print("1. KDTree insertion: O(log n) average - Tree construction overhead")
    print("2. KDTree search: O(log n) average - Much faster than O(n) list")
    print("3. KDTree nearest neighbor: O(log n) average - Huge speedup vs O(n)")
    print("4. KDTree range search: O(sqrt(n) + k) - Very efficient for spatial queries")
    print("5. Performance degrades with higher dimensions (curse of dimensionality)")
    print("6. KDTree excels at spatial/geometric queries")