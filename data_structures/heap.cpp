#include <iostream>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <random>
#include <chrono>
#include <iomanip>
using namespace std;
using namespace std::chrono;

class Heap {
    private:
        vector<int> arr;
        int capacity;
        int parent(int i) {return (i - 1) / 2;}
        int leftChild(int i) {return 2 * i + 1;}
        int rightChild(int i) {return 2 * i + 2;}
        
        // Heapify up (for insertion)
        void heapifyUp(int i) {
            while (i > 0 && arr[parent(i)] < arr[i]) {
                swap(arr[i], arr[parent(i)]);
                i = parent(i);
            }
        }
        
        // Heapify down (for deletion)
        void heapifyDown(int i) {
            int maxIndex = i;
            int left = leftChild(i);
            int right = rightChild(i);
            
            if (left < arr.size() && arr[left] > arr[maxIndex]) {
                maxIndex = left;
            }
            
            if (right < arr.size() && arr[right] > arr[maxIndex]) {
                maxIndex = right;
            }
            
            if (i != maxIndex) {
                swap(arr[i], arr[maxIndex]);
                heapifyDown(maxIndex);
            }
        }
        
    public:
        // Constructor
        Heap(int cap = 100) : capacity(cap) {
            arr.reserve(capacity);
        }
        
        // Insert an element into the heap
        void insert(int value) {
            if (arr.size() >= capacity) {
                throw overflow_error("Heap is full");
            }
            
            arr.push_back(value);
            heapifyUp(arr.size() - 1);
        }
        
        // Extract the maximum element (root)
        int extractMax() {
            if (arr.empty()) {
                throw underflow_error("Heap is empty");
            }
            
            int maxValue = arr[0];
            arr[0] = arr.back();
            arr.pop_back();
            
            if (!arr.empty()) {
                heapifyDown(0);
            }
            
            return maxValue;
        }
        
        // Get the maximum element without removing it
        int getMax() const {
            if (arr.empty()) {
                throw underflow_error("Heap is empty");
            }
            return arr[0];
        }
        
        // Get the current size of the heap
        int size() const {
            return arr.size();
        }
        
        // Check if heap is empty
        bool isEmpty() const {return arr.empty();}
        
        // Build heap from an array (heapify)
        void buildHeap(const vector<int>& elements) {
            arr = elements;
            
            // Start from the last non-leaf node and heapify down
            for (int i = (arr.size() / 2) - 1; i >= 0; i--) {
                heapifyDown(i);
            }
        }
        
        // Display the heap
        void display() const {
            if (arr.empty()) {
                cout << "Heap is empty" << endl;
                return;
            }
            
            cout << "Heap elements: ";
            for (int val : arr) {
                cout << val << " ";
            }
            cout << endl;
        }
        
        // Heap sort (returns sorted array in descending order)
        vector<int> heapSort() {
            vector<int> sorted;
            Heap tempHeap = *this;
            
            while (!tempHeap.isEmpty()) {
                sorted.push_back(tempHeap.extractMax());
            }
            
            return sorted;
        }
};

// Generate random data
vector<int> generateRandomData(int n, int seed = 42) {
    vector<int> data;
    mt19937 gen(seed);
    uniform_int_distribution<> dis(1, 100000);
    
    for (int i = 0; i < n; i++) {
        data.push_back(dis(gen));
    }
    
    return data;
}

// Benchmark insertion
void benchmarkInsertion(int n) {
    vector<int> data = generateRandomData(n);
    
    // Heap insertion
    auto start = high_resolution_clock::now();
    Heap heap;
    for (int val : data) {
        heap.insert(val);
    }
    auto end = high_resolution_clock::now();
    auto heapTime = duration_cast<microseconds>(end - start).count();
    
    // List insertion (unsorted)
    start = high_resolution_clock::now();
    vector<int> list;
    for (int val : data) {
        list.push_back(val);
    }
    end = high_resolution_clock::now();
    auto listTime = duration_cast<microseconds>(end - start).count();
    
    // Sorted list insertion (maintaining sorted order)
    start = high_resolution_clock::now();
    vector<int> sortedList;
    for (int val : data) {
        auto pos = lower_bound(sortedList.begin(), sortedList.end(), val, greater<int>());
        sortedList.insert(pos, val);
    }
    end = high_resolution_clock::now();
    auto sortedListTime = duration_cast<microseconds>(end - start).count();
    
    cout << "Insertion (" << n << " elements):" << endl;
    cout << "  Heap:             " << setw(10) << heapTime << " μs" << endl;
    cout << "  List (unsorted):  " << setw(10) << listTime << " μs" << endl;
    cout << "  List (sorted):    " << setw(10) << sortedListTime << " μs" << endl;
    cout << "  Heap speedup vs sorted list: " << fixed << setprecision(2) 
         << (double)sortedListTime / heapTime << "x" << endl << endl;
}

// Benchmark finding maximum
void benchmarkFindMax(int n) {
    vector<int> data = generateRandomData(n);
    
    // Build heap
    Heap heap;
    for (int val : data) {
        heap.insert(val);
    }
    
    // Build list
    vector<int> list = data;
    
    // Heap getMax
    auto start = high_resolution_clock::now();
    for (int i = 0; i < 1000; i++) {
        int max = heap.getMax();
    }
    auto end = high_resolution_clock::now();
    auto heapTime = duration_cast<nanoseconds>(end - start).count() / 1000.0;
    
    // List max_element
    start = high_resolution_clock::now();
    for (int i = 0; i < 1000; i++) {
        auto maxIt = max_element(list.begin(), list.end());
    }
    end = high_resolution_clock::now();
    auto listTime = duration_cast<nanoseconds>(end - start).count() / 1000.0;
    
    cout << "Find Maximum (" << n << " elements, averaged over 1000 calls):" << endl;
    cout << "  Heap:  " << setw(10) << fixed << setprecision(2) << heapTime << " ns" << endl;
    cout << "  List:  " << setw(10) << fixed << setprecision(2) << listTime << " ns" << endl;
    cout << "  Heap speedup: " << fixed << setprecision(2) 
         << listTime / heapTime << "x" << endl << endl;
}

// Benchmark extraction of maximum
void benchmarkExtractMax(int n) {
    vector<int> data = generateRandomData(n);
    
    // Build heap
    Heap heap;
    for (int val : data) {
        heap.insert(val);
    }
    
    // Build list
    vector<int> list = data;
    
    // Heap extractMax
    auto start = high_resolution_clock::now();
    while (!heap.isEmpty()) {
        heap.extractMax();
    }
    auto end = high_resolution_clock::now();
    auto heapTime = duration_cast<microseconds>(end - start).count();
    
    // List extract max (find and erase)
    start = high_resolution_clock::now();
    while (!list.empty()) {
        auto maxIt = max_element(list.begin(), list.end());
        list.erase(maxIt);
    }
    end = high_resolution_clock::now();
    auto listTime = duration_cast<microseconds>(end - start).count();
    
    cout << "Extract All Maximum Elements (" << n << " elements):" << endl;
    cout << "  Heap:  " << setw(10) << heapTime << " μs" << endl;
    cout << "  List:  " << setw(10) << listTime << " μs" << endl;
    cout << "  Heap speedup: " << fixed << setprecision(2) 
         << (double)listTime / heapTime << "x" << endl << endl;
}

// Benchmark building from existing data
void benchmarkBuildHeap(int n) {
    vector<int> data = generateRandomData(n);
    
    // Build heap from array
    auto start = high_resolution_clock::now();
    Heap heap;
    heap.buildHeap(data);
    auto end = high_resolution_clock::now();
    auto heapTime = duration_cast<microseconds>(end - start).count();
    
    // Sort list
    vector<int> list = data;
    start = high_resolution_clock::now();
    sort(list.begin(), list.end(), greater<int>());
    end = high_resolution_clock::now();
    auto listTime = duration_cast<microseconds>(end - start).count();
    
    cout << "Build/Sort from " << n << " elements:" << endl;
    cout << "  Heap (buildHeap):  " << setw(10) << heapTime << " μs" << endl;
    cout << "  List (sort):       " << setw(10) << listTime << " μs" << endl;
    cout << "  Heap speedup: " << fixed << setprecision(2) 
         << (double)listTime / heapTime << "x" << endl << endl;
}

// Benchmark priority queue operations
void benchmarkPriorityQueue(int n) {
    vector<int> data = generateRandomData(n);
    
    // Heap: insert all, then extract top k elements
    int k = min(100, n);
    auto start = high_resolution_clock::now();
    Heap heap;
    for (int val : data) {
        heap.insert(val);
    }
    vector<int> topK;
    for (int i = 0; i < k; i++) {
        topK.push_back(heap.extractMax());
    }
    auto end = high_resolution_clock::now();
    auto heapTime = duration_cast<microseconds>(end - start).count();
    
    // List: sort and get top k
    start = high_resolution_clock::now();
    vector<int> list = data;
    sort(list.begin(), list.end(), greater<int>());
    vector<int> topKList(list.begin(), list.begin() + k);
    end = high_resolution_clock::now();
    auto listTime = duration_cast<microseconds>(end - start).count();
    
    cout << "Priority Queue: Get top " << k << " from " << n << " elements:" << endl;
    cout << "  Heap:  " << setw(10) << heapTime << " μs" << endl;
    cout << "  List:  " << setw(10) << listTime << " μs" << endl;
    cout << "  Heap speedup: " << fixed << setprecision(2) 
         << (double)listTime / heapTime << "x" << endl << endl;
}

int main() {
    cout << "=====================================" << endl;
    cout << "    HEAP vs LIST BENCHMARK" << endl;
    cout << "=====================================" << endl << endl;
    
    vector<int> sizes = {100, 1000, 5000, 10000};
    
    for (int n : sizes) {
        cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << endl;
        cout << "  Dataset Size: " << n << " elements" << endl;
        cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << endl << endl;
        
        benchmarkInsertion(n);
        benchmarkFindMax(n);
        benchmarkExtractMax(n);
        benchmarkBuildHeap(n);
        benchmarkPriorityQueue(n);
    }
    
    cout << "=====================================" << endl;
    cout << "       BENCHMARK COMPLETED" << endl;
    cout << "=====================================" << endl << endl;
    
    cout << "Key Observations:" << endl;
    cout << "1. Heap insertion: O(log n) - Good for dynamic data" << endl;
    cout << "2. Heap getMax: O(1) - Instant access to maximum" << endl;
    cout << "3. Heap extractMax: O(log n) - Efficient priority queue" << endl;
    cout << "4. Heap buildHeap: O(n) - Faster than sorting" << endl;
    cout << "5. List is better for simple append operations" << endl;
    cout << "6. Heap is significantly better for priority queue operations" << endl;
    
    return 0;
}
