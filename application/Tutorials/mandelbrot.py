import numpy as np
import matplotlib.pyplot as plt

# Define the function to check if a point belongs to the Mandelbrot set
def mandelbrot(h, w, max_iter):
    y, x = np.ogrid[-1.4:1.4:h*1j, -2:0.8:w*1j]
    c = x + y*1j
    z = c
    divtime = max_iter + np.zeros(z.shape, dtype=int)

    for i in range(max_iter):
        z = z**2 + c
        diverge = z*np.conj(z) > 2**2            # who is diverging
        div_now = diverge & (divtime == max_iter)  # who is diverging now
        divtime[div_now] = i                      # note when
        z[diverge] = 2                             # avoid diverging too much

    return divtime

# Parameters for the plot
height, width = 2000,3000
max_iterations = 100

# Compute the Mandelbrot set
mandelbrot_set = mandelbrot(height, width, max_iterations)

# Create the plot
plt.figure(figsize=(10, 7))
plt.imshow(mandelbrot_set, cmap='hot', extent=[-2, 0.8, -1.4, 1.4])
plt.title('Mandelbrot Set')
plt.xlabel('Re(c)')
plt.ylabel('Im(c)')

# Remove axis ticks
plt.xticks([])
plt.yticks([])

# Show the plot
plt.show()