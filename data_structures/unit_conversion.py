from collections import defaultdict, deque
import math, time, argparse

class UnitConverter:
    """
    Parameters:
    - facts: list of tuples (u, rate, v)
    - do_cache: whether to cache results

    Variables:
    - graph: adjacency list where graph[u] = [(v1, rate1), (v2, rate2), ...]
    - queue: deque of tuples (node, amount)
    - visited: set of nodes
    - cache: memoization table where cache[(u, v)] = rate

    Notes:
    - If V is the number of units and E is the number of facts, then
    - graph build time: O(E)
    - graph query time: O(V + E) (worst case)
    - graph space complexity: O(V + E)
    """

    def __init__(self, facts:list[tuple], do_cache:bool = True):
        self.graph = defaultdict(list[tuple])
        self.cache = {} if do_cache else None
        for u, rate, v in facts:
            self.add_fact(u, rate, v)
    
    def add_fact(self, u, rate, v):
        if rate == 0: raise ValueError("Rate cannot be zero")
        self.graph[u].append((v, rate))
        self.graph[v].append((u, 1/rate))

    def convert(self, amount, u, v):
        if self.cache is not None:
            if (u, v) in self.cache: return amount * self.cache[(u, v)]
            if (v, u) in self.cache: return amount / self.cache[(v, u)]
        
        if u not in self.graph or v not in self.graph: return None
        if u == v: return amount

        # BFS initialization
        queue = deque([(u, amount)])
        visited = set([u])

        while queue:
            curr_node, curr_amount = queue.popleft()
            if curr_node == v: return curr_amount

            for neighbor, rate in self.graph[curr_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, curr_amount * rate))
                    if self.cache is not None:
                        self.cache[(u, neighbor)] = rate
        
        return None
    

class WeightedUnionFind:
    """
    Weighted Union-Find structure for maintaining unit relationships.

    Variables:
        - parents: maps each unit to its parent representative.
        - weights: stores cumulative conversion factors relative to the parent.

    Notes:
        - find: amortized inverse Ackermann, effectively O(1).
        - union: O(1) to merge two sets while preserving conversion rates.
        - convert: O(α(n)) using the stored weights after path compression.
    """
    def __init__(self):
        self.parents = {}
        self.weights = {}

    def find(self, x):
        
        # BASE CASE: IF UNIT IS NEW
        if x not in self.parents:
            self.parents[x] = x
            self.weights[x] = 1
            return x, 1

        # PATH COMPRESSION
        if self.parents[x] != x:
            root, root_weight = self.find(self.parents[x])
            self.weights[x] *= root_weight
            self.parents[x] = root
            
        return self.parents[x], self.weights[x]
    
    def union(self, x, y, value):
        # We are given: unit(x) = value * unit(y)
        root_x, weight_x = self.find(x)
        root_y, weight_y = self.find(y)

        if root_x != root_y:
            # merge root_x into root_y
            # Find W such that: unit(root_x) = W * unit(root_y)
            # Therefore W = value * weight_y / weight_x

            self.parents[root_x] = root_y
            W = value * weight_y / weight_x
            self.weights[root_x] = W
    
    def convert(self, amount, u, v):
        if u not in self.parents or v not in self.parents: return None
        if u == v: return amount

        root_u, weight_u = self.find(u)
        root_v, weight_v = self.find(v)
        if root_u != root_v: return None

        conversion_rate = weight_u / weight_v
        return amount * conversion_rate



BASE_CONVERSION_FACTS = [
    ('km', 1000, 'm'), ('m', 100, 'cm'), ('cm', 10, 'mm'), ('m', 3.28084, 'ft'), ('ft', 12, 'in'), ('in', 2.54, 'cm'),
    ('kg', 1000, 'g'), ('g', 1000, 'mg'), ('kg', 2.20462, 'lb'), ('lb', 16, 'oz'), ('ton', 2000, 'lb'),
    ('L', 1000, 'mL'), ('L', 0.264172, 'gal'), ('gal', 4, 'qt'), ('qt', 2, 'pt'), ('pt', 2, 'cup'), ('cup', 16, 'tbsp'), ('tbsp', 3, 'tsp'),
    ('hour', 60, 'min'), ('min', 60, 'sec'), ('day', 24, 'hour'), ('week', 7, 'day'),
    ('KB', 1024, 'B'), ('B', 8, 'bit'), ('MB', 1024, 'KB'), ('GB', 1024, 'MB')
]


def generate_facts(repeat_blocks:int = 1, connect_blocks:bool = True) -> list[tuple]:
    """Return a richer, realistic set of unit conversion facts.

    Args:
        repeat_blocks: number of times to replicate the base conversion
            graph. Blocks beyond the first receive a suffix ("_1", "_2", ...)
            to keep unit names unique, which is helpful when benchmarking
            large graphs.
        connect_blocks: when True, each replicated block is connected back
            to the canonical SI units (meter, kilogram, liter) so the entire
            graph remains traversable.

    Returns:
        list[tuple]: A list of (unit_a, rate, unit_b) triples describing the
        conversion rate between two units where
        ``1 unit_a = rate * unit_b``.
    """

    repeat_blocks = int(repeat_blocks)
    if repeat_blocks < 1:
        raise ValueError("repeat_blocks must be >= 1")

    facts = list(BASE_CONVERSION_FACTS)

    for block in range(1, repeat_blocks):
        suffix = f"_{block}"
        for u, rate, v in BASE_CONVERSION_FACTS:
            facts.append((f"{u}{suffix}", rate, f"{v}{suffix}"))

        if connect_blocks:
            facts.extend([
                (f"m{suffix}", 1.0, 'm'),
                (f"kg{suffix}", 1.0, 'kg'),
                (f"L{suffix}", 1.0, 'L')
            ])

    return facts


def example_BFS(facts:list[tuple]):
    converter = UnitConverter(facts, do_cache=False)
    
    print('--- EXAMPLE BFS ---')
    result_1 = converter.convert(2, 'm', 'cm')
    print(f"2 m = {result_1} cm")    
    result_2 = converter.convert(10, 'kg', 'lb')
    print(f"10 kg = {result_2} lb")    
    result_3 = converter.convert(40, 'L', 'gal')
    print(f"40 L = {result_3} gal")
    result_4 = converter.convert(1, 'm', 'sec')
    print(f"1 m = {result_4} sec")
    
    print('cache', converter.cache)
    
    print('--- BENCHMARK BFS ---')
    start = time.time()
    for _ in range(100000):
        converter.convert(1, 'm', 'cm')
    end = time.time()
    print(f"Time taken: {end - start} seconds")


def example_UnionFind(facts:list[tuple]):
    converter = WeightedUnionFind()
    for u, rate, v in facts: converter.union(u, v, rate)
    
    print('--- EXAMPLE UNION FIND ---')
    result_1 = converter.convert(2, 'm', 'cm')
    print(f"2 m = {result_1} cm")    
    result_2 = converter.convert(10, 'kg', 'lb')
    print(f"10 kg = {result_2} lb")    
    result_3 = converter.convert(40, 'L', 'gal')
    print(f"40 L = {result_3} gal")
    result_4 = converter.convert(1, 'm', 'sec')
    print(f"1 m = {result_4} sec")
    
    print('--- BENCHMARK UNION FIND ---')
    start = time.time()
    for _ in range(100000):
        converter.convert(1, 'm', 'cm')
    end = time.time()
    print(f"Time taken: {end - start} seconds")


def main():
    parser = argparse.ArgumentParser(description="Unit conversion benchmarking")
    parser.add_argument('--facts-length', type=int, default=len(BASE_CONVERSION_FACTS),
                        help='Approximate number of facts to generate (default: base fact count)')
    parser.add_argument('--connect-blocks', action='store_true', default=True,
                        help='Connect replicated blocks back to canonical SI units (default: True)')
    parser.add_argument('--no-connect-blocks', dest='connect_blocks', action='store_false',
                        help='Disable connecting replicated blocks to canonical units')
    parser.add_argument('--repeat-blocks', type=int, default=None,
                        help='Explicit number of times to replicate the base fact block. Overrides facts-length when provided.')
    args = parser.parse_args()

    base_count = len(BASE_CONVERSION_FACTS)

    if args.repeat_blocks is not None:
        repeat_blocks = max(1, args.repeat_blocks)
    else:
        repeat_blocks = max(1, math.ceil(args.facts_length / base_count))

    facts = generate_facts(repeat_blocks=repeat_blocks, connect_blocks=args.connect_blocks)
    
    if args.repeat_blocks is None and args.facts_length is not None:
        facts = facts[:args.facts_length]

    example_BFS(facts)
    example_UnionFind(facts)


if __name__ == "__main__":
    main()