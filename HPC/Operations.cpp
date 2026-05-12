#include <iostream>
#include <vector>
#include <climits>
#include <omp.h>

using namespace std;

int main() {
    int n;

    cout << "Enter number of elements: ";
    cin >> n;

    vector<int> arr(n);

    cout << "Enter elements:\n";
    for (int i = 0; i < n; i++) {
        cin >> arr[i];
    }

    int sum = 0;
    int minVal = INT_MAX;
    int maxVal = INT_MIN;

    #pragma omp parallel for reduction(+:sum) reduction(min:minVal) reduction(max:maxVal)
    for (int i = 0; i < n; i++) {
        sum += arr[i];
        if (arr[i] < minVal) minVal = arr[i];
        if (arr[i] > maxVal) maxVal = arr[i];
    }

    double avg = (double)sum / n;

    cout << "\nResults (Parallel Reduction):\n";
    cout << "Sum     = " << sum << endl;
    cout << "Minimum = " << minVal << endl;
    cout << "Maximum = " << maxVal << endl;
    cout << "Average = " << avg << endl;

    return 0;
}