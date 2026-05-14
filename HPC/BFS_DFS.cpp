//OpenMP is an API for parallel programming in C/C++ and Fortran.
#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <omp.h>

using namespace std;

class Graph {
    int V;
    vector<vector<int>> adj;

public:
    Graph(int v) {
        V = v;
        adj.resize(V);
    }

    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u); // Undirected graph
    }

    // Parallel BFS
    void parallelBFS(int start) {
        vector<bool> visited(V, false);
        queue<int> q;

        visited[start] = true;
        q.push(start);

        cout << "\nParallel BFS Traversal: ";

        while (!q.empty()) {
            int node;
//#pragma omp critical is used here to avoid race conditions while accessing and modifying the shared queue
            #pragma omp critical
            {
                node = q.front();
                q.pop();
                cout << node << " ";
            }
            /*Parallelizes neighbor processing. 
            Instead of:
            1. One thread checks neighbors one by one
            2. Multiple threads check neighbors simultaneously.*/
            #pragma omp parallel for
            for (int i = 0; i < adj[node].size(); i++) {
                int neighbor = adj[node][i];

                if (!visited[neighbor]) {
                    /*Only one thread can execute this block at a time.Needed because:
                    Without critical:
                    1.Two threads may simultaneously check visited[neighbor]
                    2.Both may find it false
                    3.Both may push same node into queue
                    This causes:
                    1. Queue is shared
                    2. Multiple threads accessing queue causes race condition*/
                    #pragma omp critical
                    {
                        if (!visited[neighbor]) {
                            visited[neighbor] = true;
                            q.push(neighbor);
                        }
                    }
                }
            }
        }
    }

    // Parallel DFS Utility
    void parallelDFSUtil(int node, vector<bool>& visited) {

        #pragma omp critical
        {
            visited[node] = true;
            cout << node << " ";
        }

        #pragma omp parallel for
        for (int i = 0; i < adj[node].size(); i++) {
            int neighbor = adj[node][i];

            if (!visited[neighbor]) {
                parallelDFSUtil(neighbor, visited);
            }
        }
    }

    // Parallel DFS
    void parallelDFS(int start) {
        vector<bool> visited(V, false);

        cout << "\nParallel DFS Traversal: ";
        parallelDFSUtil(start, visited);
    }
};

int main() {

    int V, E;
    cout << "Enter number of vertices: ";
    cin >> V;

    cout << "Enter number of edges: ";
    cin >> E;

    Graph g(V);

    cout << "Enter edges (u v):\n";
    for (int i = 0; i < E; i++) {
        int u, v;
        cin >> u >> v;
        g.addEdge(u, v);
    }

    int start;
    cout << "Enter starting vertex: ";
    cin >> start;

    g.parallelBFS(start);
    g.parallelDFS(start);

    return 0;
}