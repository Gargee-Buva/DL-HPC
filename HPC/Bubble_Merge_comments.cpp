#include <iostream>     // For input and output
#include <vector>       // For dynamic arrays (vector)
#include <omp.h>        // OpenMP library for parallel programming

using namespace std;

// ======================================================
//                 BUBBLE SORT SECTION
// ======================================================

// Sequential Bubble Sort Function
void sequentialBubbleSort(vector<int>& arr) {

    // Get size of array
    int n = arr.size();

    // Outer loop controls number of passes
    // In worst case we need n-1 passes
    for(int i = 0; i < n - 1; i++) {

        // Inner loop compares adjacent elements
        // n-i-1 because last i elements are already sorted
        for(int j = 0; j < n - i - 1; j++) {

            // If current element is greater than next element
            // swap them
            if(arr[j] > arr[j + 1]) {

                // Swap adjacent elements
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

// ======================================================
//          PARALLEL BUBBLE SORT (ODD-EVEN SORT)
// ======================================================

void parallelBubbleSort(vector<int>& arr) {

    // Get array size
    int n = arr.size();

    /*
        Odd-Even Sort Logic:

        Even Phase:
        Compare:
        (0,1), (2,3), (4,5)

        Odd Phase:
        Compare:
        (1,2), (3,4), (5,6)

        Independent pairs can be processed in parallel
    */

    // Run n phases
    for(int i = 0; i < n; i++) {

        // ==================================================
        // EVEN PHASE
        // ==================================================

        // If current pass number is even
        if(i % 2 == 0) {

            // Parallelize loop using OpenMP
            // Multiple threads compare pairs simultaneously
            #pragma omp parallel for

            // j+=2 because we compare alternate pairs
            // (0,1), (2,3), (4,5)
            for(int j = 0; j < n - 1; j += 2) {

                // Compare adjacent elements
                if(arr[j] > arr[j + 1]) {

                    // Swap if wrong order
                    swap(arr[j], arr[j + 1]);
                }
            }
        }

        // ==================================================
        // ODD PHASE
        // ==================================================

        else {

            // Parallelize loop
            #pragma omp parallel for

            // Start from index 1
            // Compare:
            // (1,2), (3,4), (5,6)
            for(int j = 1; j < n - 1; j += 2) {

                // Compare adjacent elements
                if(arr[j] > arr[j + 1]) {

                    // Swap if wrong order
                    swap(arr[j], arr[j + 1]);
                }
            }
        }
    }
}

// ======================================================
//                 MERGE SORT SECTION
// ======================================================

// Merge function combines two sorted halves
void merge(vector<int>& arr, int low, int mid, int high) {

    // Temporary array to store merged result
    vector<int> temp;

    // Pointer for left half
    int i = low;

    // Pointer for right half
    int j = mid + 1;

    // ==================================================
    // Compare elements from both halves
    // ==================================================

    while(i <= mid && j <= high) {

        // If left element is smaller
        if(arr[i] <= arr[j]) {

            // Put left element into temp array
            // i++ moves pointer forward
            temp.push_back(arr[i++]);
        }

        else {

            // Otherwise put right element
            temp.push_back(arr[j++]);
        }
    }

    // ==================================================
    // Copy remaining elements from left half
    // ==================================================

    while(i <= mid)

        temp.push_back(arr[i++]);

    // ==================================================
    // Copy remaining elements from right half
    // ==================================================

    while(j <= high)

        temp.push_back(arr[j++]);

    // ==================================================
    // Copy merged elements back into original array
    // ==================================================

    for(int k = low; k <= high; k++) {

        arr[k] = temp[k - low];
    }
}

// ======================================================
//             SEQUENTIAL MERGE SORT
// ======================================================

void sequentialMergeSort(vector<int>& arr, int low, int high) {

    // Continue dividing until one element remains
    if(low < high) {

        // Find middle index
        int mid = (low + high) / 2;

        // Recursively sort left half
        sequentialMergeSort(arr, low, mid);

        // Recursively sort right half
        sequentialMergeSort(arr, mid + 1, high);

        // Merge sorted halves
        merge(arr, low, mid, high);
    }
}

// ======================================================
//              PARALLEL MERGE SORT
// ======================================================

void parallelMergeSort(vector<int>& arr, int low, int high) {

    // Continue until subarray size becomes 1
    if(low < high) {

        // Find middle index
        int mid = (low + high) / 2;

        // ==================================================
        // Parallel Sections
        // Different code blocks run simultaneously
        // ==================================================

        #pragma omp parallel sections
        {

            // ==============================================
            // Section 1 -> Sort left half
            // ==============================================

            #pragma omp section
            {

                parallelMergeSort(arr, low, mid);
            }

            // ==============================================
            // Section 2 -> Sort right half
            // ==============================================

            #pragma omp section
            {

                parallelMergeSort(arr, mid + 1, high);
            }
        }

        // Merge sorted halves
        merge(arr, low, mid, high);
    }
}

// ======================================================
//                PRINT ARRAY FUNCTION
// ======================================================

void printArray(vector<int>& arr) {

    // Traverse all elements
    for(int x : arr)

        // Print each element
        cout << x << " ";

    cout << endl;
}

// ======================================================
//                     MAIN FUNCTION
// ======================================================

int main() {

    // Variable for number of elements
    int n;

    // Input size
    cout << "Enter number of elements: ";
    cin >> n;

    // Create array of size n
    vector<int> arr(n);

    // Input elements
    cout << "Enter elements:\n";

    for(int i = 0; i < n; i++) {

        cin >> arr[i];
    }

    // ==================================================
    // Create copies of original array
    // So all algorithms work on same input
    // ==================================================

    vector<int> arr1 = arr; // Sequential Bubble
    vector<int> arr2 = arr; // Parallel Bubble
    vector<int> arr3 = arr; // Sequential Merge
    vector<int> arr4 = arr; // Parallel Merge

    // Variables for timing
    double start, end;

    // ==================================================
    // SEQUENTIAL BUBBLE SORT
    // ==================================================

    // Start timer
    start = omp_get_wtime();

    // Run sequential bubble sort
    sequentialBubbleSort(arr1);

    // Stop timer
    end = omp_get_wtime();

    // Print sorted array
    cout << "\nSequential Bubble Sort:\n";
    printArray(arr1);

    // Print execution time
    cout << "Time: " << (end - start) << " seconds\n";

    // ==================================================
    // PARALLEL BUBBLE SORT
    // ==================================================

    start = omp_get_wtime();

    parallelBubbleSort(arr2);

    end = omp_get_wtime();

    cout << "\nParallel Bubble Sort:\n";
    printArray(arr2);

    cout << "Time: " << (end - start) << " seconds\n";

    // ==================================================
    // SEQUENTIAL MERGE SORT
    // ==================================================

    start = omp_get_wtime();

    sequentialMergeSort(arr3, 0, n - 1);

    end = omp_get_wtime();

    cout << "\nSequential Merge Sort:\n";
    printArray(arr3);

    cout << "Time: " << (end - start) << " seconds\n";

    // ==================================================
    // PARALLEL MERGE SORT
    // ==================================================

    start = omp_get_wtime();

    parallelMergeSort(arr4, 0, n - 1);

    end = omp_get_wtime();

    cout << "\nParallel Merge Sort:\n";
    printArray(arr4);

    cout << "Time: " << (end - start) << " seconds\n";

    // Program finished successfully
    return 0;
}