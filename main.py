import time
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numba import jit
from matplotlib import cm
from matplotlib.ticker import LinearLocator


# I have to change the matplotlib backend to "TkAgg" to get the plots to show, since PyQt5 decided not to work anymore.
plt.switch_backend("TkAgg")


# Find divisible number of chunks
def find_divisible_numbers(xmax, xmin, pixel_density, range_chunks):
    length_density = int((xmax - xmin) * pixel_density)
    numbers = []
    for i in range(1, range_chunks):
        if length_density % i == 0:
            numbers.append(i)
    return numbers


# A function which is used for the mapping of the values for the plot
def complex_matrix(xmin, xmax, ymin, ymax, pixel_density):
    length_density = int((xmax - xmin) * pixel_density)
    re = np.linspace(xmin, xmax, length_density)
    im = np.linspace(ymin, ymax, length_density)
    return re[np.newaxis, :] + im[:, np.newaxis] * 1j, length_density


def mandel_cal_naive(c, num_iterations, threshold, M):
    # Calculate the mandelbrot set
    for i in range(len(c[0])):
        for j in range(len(c[1])):
            c_i = c[i, j]
            z = 0 + 0j
            for k in range(num_iterations):
                z = z ** 2 + c_i
                if abs(z) <= threshold:
                    M[i, j] = num_iterations
                else:
                    M[i, j] = k
                    break


def mandelbrot_naive(xmin, xmax, ymin, ymax, pixel_density, num_iterations, threshold):

    # Create the complex matrix
    c, __ = complex_matrix(xmin, xmax, ymin, ymax, pixel_density=pixel_density)

    # Create the matrix for the plot
    M = np.zeros((len(c[0]), len(c[1])))

    start = time.time()

    # Calculate the mandelbrot set
    mandel_cal_naive(c, num_iterations, threshold, M)

    end = time.time()
    print("Time naive, no numba: ", end - start)

    # Plot the mandelbrot set
    plt.imshow(M, cmap="hot",  interpolation='bilinear', extent=[xmin, xmax, ymin, ymax])
    plt.gca().set_aspect("equal")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def mandle_sequence(c, num_iterations, threshold):
    z = np.zeros((len(c[0]), len(c[1])))
    M = np.zeros((len(c[0]), len(c[1])))

    for k in range(num_iterations):
        z = z ** 2 + c

        M[abs(z) > threshold] = k

    M[abs(z) <= threshold] = num_iterations

    return M


def mandbrot_numpy(xmin, xmax, ymin, ymax, pixel_density, num_iterations, threshold):

    # Create the complex matrix
    c, __ = complex_matrix(xmin, xmax, ymin, ymax, pixel_density=pixel_density)

    start = time.time()
    M = mandle_sequence(c, num_iterations, threshold)
    end = time.time()
    print("Time numpy, no numba: ", end - start)

    plt.imshow(M, cmap="hot", interpolation="bilinear", extent=[xmin, xmax, ymin, ymax])
    plt.gca().set_aspect("equal")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


@jit(nopython=True)
def mandel_cal_numba(c, num_iterations, threshold, M):
    # Calculate the mandelbrot set
    for i in range(len(c[0])):
        for j in range(len(c[1])):
            c_i = c[i, j]
            z = 0 + 0j
            for k in range(num_iterations):
                z = z ** 2 + c_i
                if abs(z) <= threshold:
                    M[i, j] = num_iterations
                else:
                    M[i, j] = k
                    break


def mandelbrot_numba(xmin, xmax, ymin, ymax, pixel_density, num_iterations, threshold):

    # Create the complex matrix
    c, __ = complex_matrix(xmin,xmax, ymin, ymax, pixel_density=pixel_density)

    # Create the matrix for the plot
    M = np.zeros((len(c[0]), len(c[1])))

    start = time.time()

    mandel_cal_numba(c, num_iterations, threshold, M)

    end = time.time()
    print("Time naive, with numba: ", end - start)

    # Plot the mandelbrot set
    plt.imshow(M, cmap="hot", interpolation='bilinear', extent=[xmin, xmax, ymin, ymax])
    plt.gca().set_aspect("equal")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def split(array, nrows, ncols):
    #Split a matrix into sub-matrices
    r, h = array.shape
    return (array.reshape(h//nrows, nrows, -1, ncols)
                 .swapaxes(1, 2)
                 .reshape(-1, nrows, ncols))


def mandel_cal_multi(chunk, num_iterations, threshold = 2):
    # Create the matrix for the chunk
    M = np.zeros((len(chunk[0]), len(chunk[1])))

    # Calculate the mandelbrot set for the chunk.
    for i in range(len(chunk[0])):
        for j in range(len(chunk[1])):
            c_i = chunk[i, j]
            z = 0 + 0j
            for k in range(num_iterations):
                z = z ** 2 + c_i
                if abs(z) <= threshold:
                    M[i, j] = num_iterations
                else:
                    M[i, j] = k
                    break
    
    return M


def mandelbrot_multi(xmin, xmax, ymin, ymax, pixel_density, num_iterations, threshold, cpu_n, chunks_N):

    # Create the complex matrix
    c, length_density = complex_matrix(xmin, xmax, ymin, ymax, pixel_density=pixel_density)

    # Split the matrix into equal matrix chunks
    print("Shape X: {}".format(int(c.shape[0]/chunks_N)))
    print("Shape Y: {}".format(int(c.shape[1]/chunks_N)))
    chunks = split(c, int(c.shape[0]/chunks_N), int(c.shape[1]/chunks_N))

    # Create the matrix for the plot
    M = np.zeros((length_density, length_density))

    start = time.time()

    # Calculate the chunks in parallel
    pool = mp.Pool(cpu_n)
    print("Calculating Mandelbrot set with", cpu_n, "processes...")
    results = [pool.apply_async(mandel_cal_multi, args=(chunk, num_iterations, threshold)) for chunk in chunks]
    pool.close()
    pool.join()
    output = [p.get() for p in results]

    end = time.time()
    print("Time parallel: ", end - start)

    # Convert list of arrays to a tuple of arrays
    tuple_arrays = tuple(output)

    # Combine the chunks into one matrix again using the amount of chunks
    for i in range(chunks_N):
        for j in range(chunks_N):
            M[i*int(length_density/chunks_N):(i+1)*int(length_density/chunks_N),
              j*int(length_density/chunks_N):(j+1)*int(length_density/chunks_N)] = tuple_arrays[i*chunks_N + j]

    # Plot the mandelbrot set
    plt.imshow(M, cmap="hot",  interpolation='bilinear', extent=[xmin, xmax, ymin, ymax])
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.xlabel("Re(c)")
    plt.ylabel("Im(c)")
    plt.show()


def mandelbrot_multi_optimal(xmin, xmax, ymin, ymax, pixel_density, num_iterations, threshold, cpu_n_list, chunks_N_list):

    # Create the complex matrix
    c, __ = complex_matrix(xmin, xmax, ymin, ymax, pixel_density=pixel_density)

    # Times for the different combinations of processes and chunk sizes
    times_ = np.zeros((len(cpu_n_list), len(chunks_N_list)), dtype=float)

    for processor_idx, num_processor in enumerate(cpu_n_list):
        for chunk_idx, chunk_size in enumerate(chunks_N_list):
            # Split the matrix into equal matrix chunks
            print("Shape X: {}".format(int(c.shape[0]/chunk_size)))
            print("Shape Y: {}".format(int(c.shape[1]/chunk_size)))
            chunks = split(c, int(c.shape[0]/chunk_size), int(c.shape[1]/chunk_size))

            # create a pool of processes
            print("Creating Pool with", num_processor, "processes...")
            pool = mp.Pool(num_processor)

            print("Calculating Mandelbrot set with", chunk_size, "chunk size...")
            
            start = time.time()
            
            results = [pool.apply_async(mandel_cal_multi, args=(chunk, num_iterations, threshold)) for chunk in chunks]

            print("Closing pool for number of processes {}, and number of chunks {}...".format(num_processor, chunk_size))
            pool.close()
            pool.join()
            
            print("Getting results...")
            end = time.time()
            time_result = end - start

            # Add the time to the matrix
            times_[processor_idx, chunk_idx] = time_result

            print("Time parallel with number of process {} and chunk size {}: {}".format(num_processor, chunk_size, time_result))
        

        
        print("Done calculating Mandelbrot set with", num_processor, "processes...")

        
        
    # Plot the times for the different combinations of processes and chunk sizes as a heatmap using seaborn and matplotlib
    # Create the labels for the x and y axis
    x_labels = ["{}".format(i + 1) for i in range(len(chunks_N_list))]
    y_labels = ["{}".format(i + 1) for i in range(len(cpu_n_list))]
    # Plot the heatmap
    print("Plotting the heatmap...")
    sns.heatmap(times_, annot=True, xticklabels=x_labels, yticklabels=y_labels)
    plt.xlabel("Chunk size")
    plt.ylabel("Number of processes")
    plt.show()
    print("Done plotting the heatmap")

    # Plot the times for the different combinations of processes and chunk sizes as a 3D plot using matplotlib
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(chunks_N_list, cpu_n_list)
    ax.plot_surface(X, Y, times_, cmap="hot")
    ax.set_xlabel("Chunk size")
    ax.set_ylabel("Number of processes")
    ax.set_zlabel("Time")
    plt.show()


def main():
    # Variables
    xmin, xmax, ymin, ymax = -2, 0.5, -1.5, 1.5     # - The area of the complex plane to plot, dont need to touch these
    pixel_density = 1000                            # - This value needs to be a multiple of chunks_N, or the other way around. The higher the number, the more detailed the plot (determines the size of the real and imaginary matrix)
    num_iterations = 60                             # - Number of the iterations for the mandelbrot, the higher value the prettier result
    threshold = 2                                   # - Threshold for the mandelbrot set
    cpu_n = 4                                       # - Number of CPUs to use
    chunks_N = 4                                    # - Is used to split the matrix into chunks, needs to be a multiple of the pixel_density, or the other way around

    # Functions to calculate the mandelbrot set with visualization
    mandelbrot_naive(xmin, xmax, ymin, ymax, pixel_density, num_iterations, threshold)
    mandbrot_numpy(xmin, xmax, ymin, ymax, pixel_density, num_iterations, threshold)
    mandelbrot_numba(xmin, xmax, ymin, ymax, pixel_density, num_iterations, threshold)
    mandelbrot_multi(xmin, xmax, ymin, ymax, pixel_density, num_iterations, threshold, cpu_n, chunks_N) 

    #### YOU NEED SEABORN TO PLOT THE HEATMAP ---> "pip install seaborn" or "conda install -c anaconda seaborn" #####
    # Find the better combination of processors and chunk sizes without visualization
    #numbers = find_divisible_numbers(xmax, xmin, pixel_density, 100)    # - Returns a list of numbers divisible by the pixel_density and chunks_numbers variables withing the range specified
    #print("Numbers divisible by {} and {}: {}".format(pixel_density, 100, numbers))
    #chunk_numbers = numbers                                             # - Number of chunks to split the matrix into
    #num_processors = [1, 2, 3, 4, 5]                                    # - Number of CPU cores to use

    #mandelbrot_multi_optimal(xmin, xmax, ymin, ymax, pixel_density, num_iterations, threshold, num_processors, chunk_numbers)


if __name__ == '__main__':
    main()
