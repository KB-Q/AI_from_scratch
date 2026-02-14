import random
import time
import string
import numpy as np
import sys
import argparse
random.seed(0)

class HashTable:
    """
    HashTable implementation using separate chaining.

    Attributes:
        size (int): The size of the hash table.
        table (list[list[tuple]]): The hash table.

    Public Methods:
        insert(self, key, value): Inserts a key-value pair into the hash table.
        search(self, key): Searches for a key in the hash table.
        delete(self, key): Deletes a key from the hash table.
        display(self): Displays the hash table.
    """
    def __init__(self, size=100):
        self.size: int = size
        self.table: list[list[tuple]] = [None] * size
    
    def __str__(self):
        return '\n'.join([f'{i}: {self.table[i]}' for i in range(self.size) if self.table[i] is not None])
        
    def __repr__(self):
        return str(self)
        
    def _hash(self, key):
        return hash(key) % self.size
        
    def insert(self, key, value):
        index = self._hash(key)
        if self.table[index] is None:
            self.table[index] = []
        self.table[index].append((key, value))
        
    def search(self, key):
        index = self._hash(key)
        bucket = self.table[index] # NOTE: list indexing is O(1)
        if bucket is not None:
            for k, v in bucket: # NOTE: sub list iteration is O(k) (k << n)
                if k == key:
                    return v
        return None
        
    def delete(self, key):
        index = self._hash(key)
        bucket = self.table[index]
        if bucket is None: return
        for i, (k, v) in enumerate(bucket):
            if k == key:
                del bucket[i]
                return
    
    def display(self):
        for i in range(self.size):
            bucket = self.table[i]
            if bucket is None: continue
            print(i, end=': ')
            for k, v in bucket:
                print(f'({k}, {v})', end=' ')
            print()

def generate_random_keys(n):
    l = []
    for _ in range(n):
        c = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
        l.append(c)
    return l

def benchmark_insertion(n, repeats):
    keys = generate_random_keys(n)
    values = [random.randint(1, 100000) for _ in range(n)]
    
    ht_time = []
    list_time = []
    
    for _ in range(repeats):
        # HashTable insertion
        start = time.time()
        ht = HashTable(size=n)
        for k, v in zip(keys, values):
            ht.insert(k, v)
        end = time.time()
        ht_time.append((end - start) * 1000)
        
        # List insertion (storing tuples)
        start = time.time()
        lst = []
        for k, v in zip(keys, values):
            lst.append((k, v))
        end = time.time()
        list_time.append((end - start) * 1000)
        
    ht_time_mean, ht_time_std = np.mean(ht_time), np.std(ht_time)
    list_time_mean, list_time_std = np.mean(list_time), np.std(list_time)
    
    print(f"Insertion ({n} elements):")
    print(f"  HashTable: {ht_time_mean:10.2f} ± {ht_time_std:10.2f} ms")
    print(f"  List:      {list_time_mean:10.2f} ± {list_time_std:10.2f} ms")
    print(f"  Ratio (HT/List): {ht_time_mean / list_time_mean:.2f}x\n")

def benchmark_search(n, repeats):
    keys = generate_random_keys(n)
    values = [random.randint(1, 100000) for _ in range(n)]
    
    # Build HashTable
    ht = HashTable(size=n)
    for k, v in zip(keys, values):
        ht.insert(k, v)
    
    # Build List
    lst = [(k, v) for k, v in zip(keys, values)]
    
    # Search keys
    search_keys = random.choices(keys, k=min(1000, n))
    
    ht_time = []
    list_time = []
    
    for _ in range(repeats):
        # HashTable search
        start = time.time()
        for k in search_keys:
            val = ht.search(k)
        end = time.time()
        ht_time.append((end - start) * 1000)
        
        # List search (linear search)
        start = time.time()
        for k in search_keys:
            for key, _ in lst:
                if key == k: break
        end = time.time()
        list_time.append((end - start) * 1000)
    
    ht_time_mean, ht_time_std = np.mean(ht_time), np.std(ht_time)
    list_time_mean, list_time_std = np.mean(list_time), np.std(list_time)
    
    print(f"Search ({len(search_keys)} searches in {n} elements):")
    print(f"  HashTable: {ht_time_mean:10.2f} ± {ht_time_std:10.2f} ms")
    print(f"  List:      {list_time_mean:10.2f} ± {list_time_std:10.2f} ms")
    print(f"  HashTable speedup: {list_time_mean / ht_time_mean:.2f}x\n")

def benchmark_delete(n, repeats):
    keys = generate_random_keys(n)
    values = [random.randint(1, 100000) for _ in range(n)]
    
    # Build HashTable
    ht = HashTable(size=n)
    for k, v in zip(keys, values):
        ht.insert(k, v)
    
    # Build List
    lst = [(k, v) for k, v in zip(keys, values)]
    
    # Delete keys
    delete_keys = random.choices(keys, k=min(100, n))
    
    ht_time = []
    list_time = []
    
    for _ in range(repeats):
        # HashTable delete
        start = time.time()
        for k in delete_keys:
            ht.delete(k)
        end = time.time()
        ht_time.append((end - start) * 1000)
        
        # List delete (find and remove)
        start = time.time()
        for k in delete_keys:
            for i, (key, _) in enumerate(lst):
                if key == k:
                    del lst[i]
                    break
        end = time.time()
        list_time.append((end - start) * 1000)
    
    ht_time_mean, ht_time_std = np.mean(ht_time), np.std(ht_time)
    list_time_mean, list_time_std = np.mean(list_time), np.std(list_time)
    
    print(f"Delete ({len(delete_keys)} deletions from {n} elements):")
    print(f"  HashTable: {ht_time_mean:10.2f} ± {ht_time_std:10.2f} ms")
    print(f"  List:      {list_time_mean:10.2f} ± {list_time_std:10.2f} ms")
    print(f"  HashTable speedup: {list_time_mean / ht_time_mean:.2f}x\n")

def benchmark_collision_analysis(n):
    keys = generate_random_keys(n)
    values = [random.randint(1, 100000) for _ in range(n)]
    
    # Build HashTable with different sizes to analyze collisions
    sizes = [n // 4, n // 2, n, n * 2]
    
    print(f"Collision Analysis ({n} elements):")
    for size in sizes:
        ht = HashTable(size=size)
        for k, v in zip(keys, values):
            ht.insert(k, v)
        
        # Count collisions
        collisions = sum(1 for bucket in ht.table if bucket is not None and len(bucket) > 1)
        max_chain = max((len(bucket) if bucket is not None else 0 for bucket in ht.table), default=0)
        
        print(f"  Size={size:6d}: Collisions={collisions:4d}, Max Chain={max_chain:2d}, Load Factor={n/size:.2f}")
    print()

def benchmark_mixed_operations(n, repeats):
    keys = generate_random_keys(n)
    values = [random.randint(1, 100000) for _ in range(n)]
    
    ht_time = []
    list_time = []
    
    for _ in range(repeats):
        # HashTable mixed operations
        start = time.time()
        ht = HashTable(size=n)
        for i, (k, v) in enumerate(zip(keys, values)):
            ht.insert(k, v)
            if i % 3 == 0:
                ht.search(k)
            if i % 5 == 0 and i > 0:
                ht.delete(keys[i // 2])
        end = time.time()
        ht_time.append((end - start) * 1000)
        
        # List mixed operations
        start = time.time()
        lst = []
        for i, (k, v) in enumerate(zip(keys, values)):
            lst.append((k, v))
            if i % 3 == 0:
                # Search
                for key, _ in lst:
                    if key == k: break
            if i % 5 == 0 and i > 0:
                # Delete
                for j, (key, _) in enumerate(lst):
                    if key == keys[i // 2]:
                        del lst[j]
                        break
        end = time.time()
        list_time.append((end - start) * 1000)
    
    ht_time_mean, ht_time_std = np.mean(ht_time), np.std(ht_time)
    list_time_mean, list_time_std = np.mean(list_time), np.std(list_time)
    
    print(f"Mixed Operations ({n} elements with inserts, searches, deletes):")
    print(f"  HashTable: {ht_time_mean:10.2f} ± {ht_time_std:10.2f} ms")
    print(f"  List:      {list_time_mean:10.2f} ± {list_time_std:10.2f} ms")
    print(f"  HashTable speedup: {list_time_mean / ht_time_mean:.2f}x\n")

def main(sizes, repeats):
    print("=" * 40)
    print("  HASHTABLE vs LIST BENCHMARK")
    print("=" * 40)
    print()
    
    for n in sizes:
        print("=" * 40)
        print(f"  Dataset Size: {n} elements")
        print("=" * 40)
        print()
        
        benchmark_insertion(n, repeats)
        benchmark_search(n, repeats)
        benchmark_delete(n, repeats)
        benchmark_mixed_operations(n, repeats)
        benchmark_collision_analysis(n)
    
    print("=" * 40)
    print("      BENCHMARK COMPLETED")
    print("=" * 40)
    print()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark HashTable vs List')
    parser.add_argument('--sizes', type=str, default='100,1000,5000,10000',
                        help='Comma-separated list of dataset sizes (default: 100,1000,5000,10000)')
    parser.add_argument('--repeats', type=int, default=1,
                        help='Number of times to repeat each benchmark (default: 1)')
    
    args = parser.parse_args()
    sizes = [int(x) for x in args.sizes.split(',')]
    repeats = args.repeats
    main(sizes, repeats)