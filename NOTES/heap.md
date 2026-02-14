# Heap Data Structure - Algorithm Documentation

## Overview

A **heap** is a specialized tree-based data structure that satisfies the heap property. In a **max heap**, for any given node, the value of that node is greater than or equal to the values of its children. This implementation uses an array-based representation where for any element at index `i`:
- Parent is at index `(i-1)/2`
- Left child is at index `2*i + 1`
- Right child is at index `2*i + 2`

## Time Complexity Summary

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Insert | O(log n) | O(1) |
| Extract Max | O(log n) | O(1) |
| Get Max | O(1) | O(1) |
| Build Heap | O(n) | O(1) |
| Heap Sort | O(n log n) | O(n) |

---

## Core Algorithms

### 1. Heapify Up (Bubble Up)

**Purpose:** Maintains heap property after inserting a new element at the end.

**Algorithm:**
```
heapifyUp(index):
    while index > 0:
        parent_index = (index - 1) / 2
        if arr[parent_index] < arr[index]:
            swap(arr[parent_index], arr[index])
            index = parent_index
        else:
            break
```

**How it works:**
1. Start at the newly inserted element (at the end of array)
2. Compare with its parent
3. If child > parent, swap them
4. Move up to parent's position
5. Repeat until heap property is satisfied or reach root

**Time Complexity:** O(log n) - In worst case, element bubbles up from leaf to root (height of tree)

**Example:**
```
Initial: [50, 30, 20, 10]
Insert 60: [50, 30, 20, 10, 60]

Step 1: Compare 60 with parent 30
        [50, 60, 20, 10, 30]  (swap)
        
Step 2: Compare 60 with parent 50
        [60, 50, 20, 10, 30]  (swap)
        
Done! Heap property restored.
```

---

### 2. Heapify Down (Trickle Down)

**Purpose:** Maintains heap property after removing the root or during heap construction.

**Algorithm:**
```
heapifyDown(index):
    loop:
        max_index = index
        left = 2 * index + 1
        right = 2 * index + 2
        
        if left < size AND arr[left] > arr[max_index]:
            max_index = left
        
        if right < size AND arr[right] > arr[max_index]:
            max_index = right
        
        if max_index != index:
            swap(arr[index], arr[max_index])
            index = max_index
        else:
            break
```

**How it works:**
1. Start at given index (usually root after deletion)
2. Find the maximum among: current node, left child, right child
3. If maximum is not the current node, swap with the maximum child
4. Move down to that child's position
5. Repeat until heap property is satisfied or reach leaf

**Time Complexity:** O(log n) - In worst case, element trickles down from root to leaf

**Example:**
```
After extracting max from [60, 50, 20, 10, 30]:
Replace root with last: [30, 50, 20, 10]

Step 1: Compare 30 with children 50 and 20
        Max is 50, swap: [50, 30, 20, 10]
        
Step 2: Compare 30 with children 10 and (none)
        30 > 10, done!
        
Final: [50, 30, 20, 10]
```

---

### 3. Insert

**Purpose:** Add a new element to the heap while maintaining heap property.

**Algorithm:**
```
insert(value):
    arr.push_back(value)
    heapifyUp(arr.size() - 1)
```

**How it works:**
1. Add the new element at the end of the array (as a leaf)
2. Call heapifyUp to restore heap property
3. The new element "bubbles up" to its correct position

**Time Complexity:** O(log n)

**Why it's efficient:**
- Adding to end of array: O(1)
- Heapify up: O(log n)
- Total: O(log n)

---

### 4. Extract Maximum

**Purpose:** Remove and return the maximum element (root) from the heap.

**Algorithm:**
```
extractMax():
    if empty:
        throw error
    
    max_value = arr[0]
    arr[0] = arr[arr.size() - 1]
    arr.pop_back()
    
    if not empty:
        heapifyDown(0)
    
    return max_value
```

**How it works:**
1. Save the root value (maximum)
2. Move the last element to the root
3. Remove the last element
4. Call heapifyDown from root to restore heap property
5. Return the saved maximum value

**Time Complexity:** O(log n)

**Why replace with last element:**
- Avoids shifting all elements
- Maintains complete binary tree structure
- O(1) array operation

---

### 5. Get Maximum

**Purpose:** Return the maximum element without removing it.

**Algorithm:**
```
getMax():
    if empty:
        throw error
    return arr[0]
```

**Time Complexity:** O(1)

**Why it's O(1):** The maximum element is always at the root (index 0) in a max heap.

---

### 6. Build Heap (Heapify)

**Purpose:** Convert an arbitrary array into a valid heap efficiently.

**Algorithm:**
```
buildHeap(elements[]):
    arr = elements
    
    // Start from last non-leaf node
    for i = (arr.size() / 2 - 1) down to 0:
        heapifyDown(i)
```

**How it works:**
1. Copy the array as-is
2. Start from the last non-leaf node (parent of last element)
3. Call heapifyDown on each node going backwards to root
4. Skip leaf nodes (they're already valid heaps of size 1)

**Time Complexity:** O(n) - NOT O(n log n)!

**Why O(n) and not O(n log n):**
- Most nodes are near the bottom (leaves)
- Leaves: 0 swaps, n/2 nodes
- Second-last level: 1 swap max, n/4 nodes
- Third-last level: 2 swaps max, n/8 nodes
- Summing up: n/2×0 + n/4×1 + n/8×2 + ... = O(n)

**Example:**
```
Input: [15, 10, 20, 8, 12, 25, 5]

Last non-leaf: index 2 (value 20)

Step 1: heapifyDown(2) - Compare 20 with 25, swap
        [15, 10, 25, 8, 12, 20, 5]

Step 2: heapifyDown(1) - Compare 10 with 8 and 12, swap with 12
        [15, 12, 25, 8, 10, 20, 5]

Step 3: heapifyDown(0) - Compare 15 with 12 and 25, swap with 25
        [25, 12, 15, 8, 10, 20, 5]
        Then compare 25 with 15 and 20, swap with 20
        [25, 12, 20, 8, 10, 15, 5]

Final heap: [25, 12, 20, 8, 10, 15, 5]
```

---

### 7. Heap Sort

**Purpose:** Sort an array using the heap data structure.

**Algorithm:**
```
heapSort():
    sorted = []
    while not empty:
        sorted.push_back(extractMax())
    return sorted
```

**Time Complexity:** O(n log n)
**Space Complexity:** O(n) for the sorted array

**How it works:**
1. Build a heap from the array: O(n)
2. Extract max n times: n × O(log n) = O(n log n)
3. Each extracted element is the next largest

**In-place version:**
```
heapSort(arr):
    buildHeap(arr)
    for i = arr.size()-1 down to 1:
        swap(arr[0], arr[i])
        size--
        heapifyDown(0)
```

---

## Applications

### 1. Priority Queue
- Insert with priority: O(log n)
- Get highest priority: O(1)
- Remove highest priority: O(log n)

### 2. Top K Elements
- Build heap from n elements: O(n)
- Extract k elements: O(k log n)
- Total: O(n + k log n)

### 3. Heap Sort
- In-place sorting: O(n log n) time, O(1) space
- Not stable, but guaranteed O(n log n) worst case

### 4. Median Maintenance
- Use two heaps (max heap for lower half, min heap for upper half)
- Insert: O(log n)
- Get median: O(1)

---

## Heap vs Other Data Structures

| Operation | Heap | Sorted Array | Unsorted List |
|-----------|------|--------------|---------------|
| Insert | O(log n) | O(n) | O(1) |
| Get Max | O(1) | O(1) | O(n) |
| Extract Max | O(log n) | O(n) | O(n) |
| Build from n elements | O(n) | O(n log n) | O(n) |
| Space | O(n) | O(n) | O(n) |

**When to use Heap:**
- Need frequent access to maximum/minimum
- Implementing priority queues
- When you need both insertion and extraction to be fast
- Finding top K elements efficiently

**When to avoid Heap:**
- Need to search for arbitrary elements: O(n)
- Need sorted order of all elements: use sorting
- Simple FIFO/LIFO: use queue/stack

---

## Implementation Details

### Array-based vs Pointer-based

**Array-based (our implementation):**
- **Pros:** Cache-friendly, less memory overhead, simple indexing
- **Cons:** Fixed capacity (can use vector for dynamic)

**Pointer-based:**
- **Pros:** Dynamic size, can store additional data per node
- **Cons:** More memory overhead, pointer chasing is slower

### Min Heap vs Max Heap

Our implementation is a **max heap** (parent ≥ children).

To convert to **min heap:**
- In heapifyUp: change `arr[parent] < arr[i]` to `arr[parent] > arr[i]`
- In heapifyDown: find minimum child instead of maximum

---

## Common Pitfalls

1. **Off-by-one errors** in parent/child calculations
   - Parent: `(i-1)/2` not `i/2`
   - Children: `2*i+1` and `2*i+2` not `2*i` and `2*i+1`

2. **Forgetting to check bounds** when accessing children
   - Always check `left < size` and `right < size`

3. **Building heap with repeated insertions**
   - O(n log n) time
   - Use buildHeap() for O(n) time

4. **Not maintaining complete binary tree structure**
   - Always insert at end and extract from root
   - Never insert/delete from middle

---

## Complexity Analysis Summary

### Why Insert is O(log n):
- Height of heap = log₂(n)
- In worst case, element travels from leaf to root
- Number of comparisons/swaps ≤ height = O(log n)

### Why BuildHeap is O(n):
- Sum of work at each level: Σ(nodes at level × swaps per node)
- Most nodes are at bottom with minimal work
- Mathematical proof: Σ(n/2^i × i) for i=1 to log n = O(n)

### Why GetMax is O(1):
- Max heap property guarantees root is maximum
- Array access at index 0 is constant time

---

## Conclusion

Heaps are powerful for priority-based operations with O(log n) insertion and extraction, O(1) max access, and O(n) construction. They're the foundation of efficient priority queues and have applications in sorting, graph algorithms (Dijkstra's, Prim's), and data stream processing.
