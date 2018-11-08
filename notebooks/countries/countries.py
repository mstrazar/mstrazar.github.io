import matplotlib.pyplot as plt
import numpy as np
import itertools as it

# TODO visualize visited nodes

def random_color(i, a=1611622097, n=2**24-1):
    # TODO: try different values of a
    # Get i-th random color for this run
    # Use a simple random generator that is modulo to some large number
    # although a graph can be colored by 4 colors
    return (a * i) % n

def color2rbg(x):
    # Integer to rgb
    return np.array([((x // 256**2) % 256, (x // 256) % 256, x % 256)])


def solution(A):
    # Solution based on search and iterative-recursion with stacks.
    # Add nodes to stack as you go along.
    m, n = len(A), len(A[0])
    total = 0
    time = 0
    done = set()

    # Main stack
    todo = {(0, 0)}
    todo_empty = len(todo) == 0

    # Color and time array
    C = np.zeros((m, n))
    T = np.zeros((m, n))

    while not todo_empty:
        i, j = todo.pop()
        if (i, j) in done:
            todo_empty = len(todo) == 0
            continue
        stack = {(i, j)}
        stack_empty = len(stack) == 0
        total += 1

        # Time increases each time a new tile is connected to current search
        time += 1
        T[i, j] = time
        C[i, j] = total

        while not stack_empty:
            a, b = stack.pop()
            done.add((a, b))
            current_color = A[a][b]
            for di, dj in (-1, 0), (1, 0), (0, -1), (0, 1):
                x = a + di
                y = b + dj
                if x < 0 or y < 0 or x >= m or y >= n:
                    continue
                if A[x][y] == current_color and (x, y) not in done:
                    stack.add((x, y))
                    time += 1
                    T[x, y] = time
                    C[x, y] = total
                elif (x, y) not in done:
                    todo.add((x, y))
            stack_empty = len(stack) == 0
        todo_empty = len(todo) == 0
    return total, C, T

if __name__ == "__main__":

    # Initial map
    A = np.array([[5., 4., 4.],
                   [4., 3., 4.],
                   [3., 2., 4.],
                   [2., 2., 2.],
                   [3., 3., 4.],
                   [1., 4., 4.],
                   [4., 1., 1.]])
    m, n = A.shape[0], A.shape[1]

    # Compute solution
    total, C, T = solution(A)

    # Plot a color array
    T = T.reshape((m, n, 1))
    D = np.zeros((m, n, 3))
    for i, j in it.product(range(m), range(n)):
        D[i, j] = color2rbg(random_color(C[i, j]))

    # Plot a figure at time t
    for t in range(1, m*n):
        fname = "/Users/martins/Dev/www/img/posts/countries/frame_%03d.png" % t
        plt.figure()
        plt.imshow(D * (T < t))
        plt.savefig(fname)
        plt.close()
        print("Written %s" % fname)

