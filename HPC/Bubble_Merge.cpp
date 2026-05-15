#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

// ---------- Bubble Sort ----------

// Sequential Bubble Sort
void sequentialBubbleSort(vector<int>& arr) {
    int n = arr.size();

    for(int i = 0; i < n - 1; i++) {
        for(int j = 0; j < n - i - 1; j++) {
            if(arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

// Parallel Bubble Sort (Odd-Even Sort)
void parallelBubbleSort(vector<int>& arr) {
    int n = arr.size();
    /*If current pass number is even:
        perform EVEN phase comparisons
        Else:
        perform ODD phase comparisons
    */

    for(int i = 0; i < n; i++) {

        // Even phase
        if(i % 2 == 0) {
//OpenMP distributes pair comparisons among threads.
            #pragma omp parallel for
            for(int j = 0; j < n - 1; j += 2) {
                if(arr[j] > arr[j + 1]) {
                    swap(arr[j], arr[j + 1]);
                }
            }
        }

        // Odd phase
        else {

            #pragma omp parallel for
            for(int j = 1; j < n - 1; j += 2) {
                if(arr[j] > arr[j + 1]) {
                    swap(arr[j], arr[j + 1]);
                }
            }
        }
    }
}

// ---------- Merge Sort ----------

void merge(vector<int>& arr, int low, int mid, int high) {

    vector<int> temp;

    int i = low;
    int j = mid + 1;

    while(i <= mid && j <= high) {
        if(arr[i] <= arr[j]) {
            temp.push_back(arr[i++]);
        }
        else {
            temp.push_back(arr[j++]);
        }
    }

    while(i <= mid)
        temp.push_back(arr[i++]);

    while(j <= high)
        temp.push_back(arr[j++]);

    for(int k = low; k <= high; k++) {
        arr[k] = temp[k - low];
    }
}

// Sequential Merge Sort
void sequentialMergeSort(vector<int>& arr, int low, int high) {

    if(low < high) {

        int mid = (low + high) / 2;

        sequentialMergeSort(arr, low, mid);
        sequentialMergeSort(arr, mid + 1, high);

        merge(arr, low, mid, high);
    }
}

// Parallel Merge Sort
void parallelMergeSort(vector<int>& arr, int low, int high) {

    if(low < high) {

        int mid = (low + high) / 2;

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                parallelMergeSort(arr, low, mid);
            }

            #pragma omp section
            {
                parallelMergeSort(arr, mid + 1, high);
            }
        }

        merge(arr, low, mid, high);
    }
}

// ---------- Utility Function ----------

void printArray(vector<int>& arr) {
    for(int x : arr)
        cout << x << " ";
    cout << endl;
}

// ---------- Main ----------

int main() {

    int n;

    cout << "Enter number of elements: ";
    cin >> n;

    vector<int> arr(n);

    cout << "Enter elements:\n";

    for(int i = 0; i < n; i++) {
        cin >> arr[i];
    }

    vector<int> arr1 = arr;
    vector<int> arr2 = arr;
    vector<int> arr3 = arr;
    vector<int> arr4 = arr;

    double start, end;

    // Sequential Bubble Sort
    start = omp_get_wtime();
    sequentialBubbleSort(arr1);
    end = omp_get_wtime();

    cout << "\nSequential Bubble Sort:\n";
    printArray(arr1);

    cout << "Time: " << (end - start) << " seconds\n";

    // Parallel Bubble Sort
    start = omp_get_wtime();
    parallelBubbleSort(arr2);
    end = omp_get_wtime();

    cout << "\nParallel Bubble Sort:\n";
    printArray(arr2);

    cout << "Time: " << (end - start) << " seconds\n";

    // Sequential Merge Sort
    start = omp_get_wtime();
    sequentialMergeSort(arr3, 0, n - 1);
    end = omp_get_wtime();

    cout << "\nSequential Merge Sort:\n";
    printArray(arr3);

    cout << "Time: " << (end - start) << " seconds\n";

    // Parallel Merge Sort
    start = omp_get_wtime();
    parallelMergeSort(arr4, 0, n - 1);
    end = omp_get_wtime();

    cout << "\nParallel Merge Sort:\n";
    printArray(arr4);

    cout << "Time: " << (end - start) << " seconds\n";

    return 0;
}