class Heap:
    """
    Min Heap Implementation

    Variables:
        - heap: just a list
        - size: the size of the heap
        - index: the index of the node in the heap list

    Public Functions:
        - insert: inserts a key into the heap
        - extract_min: extracts the minimum element from the heap
        - build_heap: builds a heap from an array
        - display: displays the heap
        - heap_sort: returns a sorted array
    """
    def __init__(self):
        self.heap = []
        self.size = 0

    def _display_helper(self, index, indent):
        if index < self.size:
            print(' ' * indent + '-->', self.heap[index])

    def _parent(self, index): 
        return (index - 1) // 2
    
    def _left(self, index): 
        return 2 * index + 1
    
    def _right(self, index): 
        return 2 * index + 2
    
    def _swap(self, i, j): 
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

    def _heapify_up(self, index):
        if index == 0: return
        curr_val, parent_val = self.heap[index], self.heap[self._parent(index)]
        if curr_val < parent_val:
            self._swap(index, self._parent(index))
            self._heapify_up(self._parent(index))
        else: return

    def _heapify_down(self, index):
        if index >= self.size: return
        smallest = index
        left_idx = self._left(index)
        right_idx = self._right(index)
        
        if left_idx < self.size and self.heap[left_idx] < self.heap[smallest]:
            smallest = left_idx
        
        if right_idx < self.size and self.heap[right_idx] < self.heap[smallest]:
            smallest = right_idx
        
        if smallest != index:
            self._swap(index, smallest)
            self._heapify_down(smallest)

    def insert(self, key):
        self.heap.append(key)
        self.size += 1
        self._heapify_up(self.size - 1)

    def extract_min(self):
        if self.size == 0: return None
        if self.size == 1: return self.heap.pop()
        min_val = self.heap[0]
        self.heap[0] = self.heap.pop()
        self.size -= 1
        self._heapify_down(0)
        return min_val

    def build_heap(self, arr):
        self.heap = arr
        self.size = len(arr)
        for i in range(self.size // 2 - 1, -1, -1):
            self._heapify_down(i)
    
    def display(self, indent=0):
        if self.size == 0: return
        if indent == 0: print(self.heap[0])
        else: print(' ' * indent + '-->', self.heap[0])
        for i in range(1, self.size // 2):
            self._display_helper(self._left(i), indent + 1)
            self._display_helper(self._right(i), indent + 1)

    def heap_sort(self):
        sorted = []
        while self.size > 0:
            sorted.append(self.extract_min())
        return sorted
        

import random
import time

def generate_random_data(n):
    return [random.randint(1, 100000) for _ in range(n)]

def benchmark_insertion(n):
    data = generate_random_data(n)
    
    # Heap insertion
    heap = Heap()
    start = time.time()
    for val in data:
        heap.insert(val)
    end = time.time()
    heap_time = (end - start) * 1000
    
    # List insertion (unsorted)
    lst = []
    start = time.time()
    for val in data:
        lst.append(val)
    end = time.time()
    list_time = (end - start) * 1000
    
    # Sorted list insertion (maintaining sorted order)
    sorted_list = []
    start = time.time()
    for val in data:
        sorted_list.append(val)
        sorted_list.sort()
    end = time.time()
    sorted_list_time = (end - start) * 1000
    
    print(f"Insertion ({n} elements):")
    print(f"  Heap:            {heap_time:10.2f} ms")
    print(f"  List (unsorted): {list_time:10.2f} ms")
    print(f"  List (sorted):   {sorted_list_time:10.2f} ms")
    print(f"  Heap speedup vs sorted list: {sorted_list_time / heap_time:.2f}x\n")

def benchmark_find_min(n):
    data = generate_random_data(n)
    
    # Build heap
    heap = Heap()
    for val in data:
        heap.insert(val)
    
    # Build list
    lst = data.copy()
    
    # Heap get min
    iterations = 1000
    start = time.time()
    for _ in range(iterations):
        min_val = heap.heap[0] if heap.size > 0 else None
    end = time.time()
    heap_time = ((end - start) * 1000000) / iterations
    
    # List min
    start = time.time()
    for _ in range(iterations):
        min_val = min(lst) if lst else None
    end = time.time()
    list_time = ((end - start) * 1000000) / iterations
    
    print(f"Find Minimum ({n} elements, averaged over {iterations} calls):")
    print(f"  Heap: {heap_time:10.2f} ns")
    print(f"  List: {list_time:10.2f} ns")
    print(f"  Heap speedup: {list_time / heap_time:.2f}x\n")

def benchmark_extract_min(n):
    data = generate_random_data(n)
    
    # Build heap
    heap = Heap()
    for val in data:
        heap.insert(val)
    
    # Build list
    lst = data.copy()
    
    # Heap extract_min
    start = time.time()
    while heap.size > 0:
        heap.extract_min()
    end = time.time()
    heap_time = (end - start) * 1000
    
    # List extract min
    start = time.time()
    while lst:
        min_val = min(lst)
        lst.remove(min_val)
    end = time.time()
    list_time = (end - start) * 1000
    
    print(f"Extract All Minimum Elements ({n} elements):")
    print(f"  Heap: {heap_time:10.2f} ms")
    print(f"  List: {list_time:10.2f} ms")
    print(f"  Heap speedup: {list_time / heap_time:.2f}x\n")

def benchmark_build_heap(n):
    data = generate_random_data(n)
    
    # Build heap from array
    start = time.time()
    heap = Heap()
    heap.build_heap(data.copy())
    end = time.time()
    heap_time = (end - start) * 1000
    
    # Sort list
    start = time.time()
    lst = data.copy()
    lst.sort()
    end = time.time()
    list_time = (end - start) * 1000
    
    print(f"Build/Sort from {n} elements:")
    print(f"  Heap (build_heap): {heap_time:10.2f} ms")
    print(f"  List (sort):       {list_time:10.2f} ms")
    print(f"  Heap speedup: {list_time / heap_time:.2f}x\n")

def benchmark_priority_queue(n):
    data = generate_random_data(n)
    
    # Heap: insert all, then extract top k elements
    k = min(100, n)
    start = time.time()
    heap = Heap()
    for val in data:
        heap.insert(val)
    top_k = []
    for _ in range(k):
        top_k.append(heap.extract_min())
    end = time.time()
    heap_time = (end - start) * 1000
    
    # List: sort and get top k
    start = time.time()
    lst = data.copy()
    lst.sort()
    top_k_list = lst[:k]
    end = time.time()
    list_time = (end - start) * 1000
    
    print(f"Priority Queue: Get top {k} from {n} elements:")
    print(f"  Heap: {heap_time:10.2f} ms")
    print(f"  List: {list_time:10.2f} ms")
    print(f"  Heap speedup: {list_time / heap_time:.2f}x\n")

if __name__ == '__main__':
    print("=" * 40)
    print("    HEAP vs LIST BENCHMARK")
    print("=" * 40)
    print()
    
    sizes = [100, 1000, 5000, 10000]
    
    for n in sizes:
        print("=" * 40)
        print(f"  Dataset Size: {n} elements")
        print("=" * 40)
        print()
        
        benchmark_insertion(n)
        benchmark_find_min(n)
        benchmark_extract_min(n)
        benchmark_build_heap(n)
        benchmark_priority_queue(n)
    
    print("=" * 40)
    print("       BENCHMARK COMPLETED")
    print("=" * 40)
    print()
    
    print("Key Observations:")
    print("1. Heap insertion: O(log n) - Good for dynamic data")
    print("2. Heap get_min: O(1) - Instant access to minimum")
    print("3. Heap extract_min: O(log n) - Efficient priority queue")
    print("4. Heap build_heap: O(n) - Faster than sorting")
    print("5. List is better for simple append operations")
    print("6. Heap is significantly better for priority queue operations")