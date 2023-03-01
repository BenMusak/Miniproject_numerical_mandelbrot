


def mandel_cal_multi(x, y, num_iterations, threshold = 2):
    c = complex(x, y)
    z = 0.0j
    for i in range(num_iterations):
        z = z*z + c
        if abs(z) >= threshold:
            return i
    return num_iterations


def mandelbrot_multi():
    # Variables
    cpu_n = 3
    threshold = 2
    xmin, xmax, ymin, ymax = -2, 0.5, -1.5, 1.5
    pixel_density = 500
    num_iterations = 60

    # Generate 2D grid of complex numbers, and 2D array of output. Im not using the complex matrix function here, since we want the imaginary part later.
    print("Generating 2D grid of complex numbers...")
    length_density = int((xmax - xmin) * pixel_density)
    xs = np.linspace(xmin, xmax, length_density)
    ys = np.linspace(ymin, ymax, length_density)
    zs = np.zeros((length_density, length_density))


    # Calculate output in parallel
    pool = mp.Pool(cpu_n)
    print("Calculating Mandelbrot set with", cpu_n, "processes...")
    results = [pool.apply_async(mandel_cal_multi, args=(x, y, num_iterations, threshold)) for x in xs for y in ys]
    pool.close()
    pool.join()
    output = [p.get() for p in results]

    print(output)
    
    # Generate 2D output array for plotting
    print("Generating 2D array...")
    for i in range(length_density):
        for j in range(length_density):
                zs[i, j] = output[i*length_density + j]
    
    # Plot the results
    print("Plotting...")
    plt.imshow(zs.T, cmap='hot', interpolation='bilinear')
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.xlabel("Re(c)")
    plt.ylabel("Im(c)")
    plt.show()