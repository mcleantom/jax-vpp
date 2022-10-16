import jax.numpy as np
import jax


@jax.jit
def rosenbrock(x):
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)


X = np.array([1.5, -0.5])

LEARNING_RATE = 0.001

for _ in range(10000):
    grad = jax.grad(rosenbrock)(X)
    X = X - LEARNING_RATE * grad

f, grad = jax.value_and_grad(rosenbrock)(X)
