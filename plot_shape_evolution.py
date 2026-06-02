from numba import njit, prange
import numpy as np
import matplotlib.pyplot as plt



#Parameters

alpha = np.pi/4
resolution = 1000
n_steps = 10 #how many steps to iterate
R = np.sqrt(n_steps) # initial radius



#Helper functions

def create_grid(Lx, Ly, Nx, Ny):
    '''
    Creates uniform grid in [-Lx, Lx] x [-Ly, Ly] with 
    Nx steps in x direction and Ny steps in y direction.

    Output:
    x, y: arrays of x coordinates, y coordinates
    X, Y: coordinate pairs taken from x, y
    dx, dy: respective step sizes in x, y
    '''

    x = np.linspace(-Lx, Lx, Nx)
    y = np.linspace(-Ly, Ly, Ny)
    X, Y = np.meshgrid(x,y)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    return x, y, X, Y, dx, dy



def initialize_circle(X, Y, R):
    '''
    Creates Boolean array defining a circle of radius R
    in grid X x Y.
    '''
    return X**2 + Y**2 <= R**2





def compute_unit_offsets(dx, dy, alpha=alpha, num_angles=512):
    '''
    Creates a discretized version of the unit 
    
    
    '''
    angles = np.concatenate([
        np.linspace(-alpha, alpha, num_angles//2),
        np.linspace(np.pi-alpha, np.pi+alpha, num_angles//2)
    ])
    offsets = set()
    for theta in angles:
        di = int(round(np.sin(theta)/dy))
        dj = int(round(np.cos(theta)/dx))
        offsets.add((di, dj))
    return list(offsets)

units = compute_unit_offsets(dx = 0.00001, dy = 0.00001)

for vec in units:
    print(vec[0]**2+vec[1]**2)

@njit(parallel=True)
def update_step(A, offsets):
    Ny, Nx = A.shape
    B = np.ones((Ny, Nx), dtype=np.bool_)

    for i in prange(Ny):
        for j in range(Nx):
            keep = True
            for k in range(len(offsets)):
                di, dj = offsets[k]  # integers now

                # plus
                ip = i + di
                jp = j + dj
                inside_plus = False
                if 0 <= ip < Ny and 0 <= jp < Nx:
                    inside_plus = A[ip, jp]

                # minus
                im = i - di
                jm = j - dj
                inside_minus = False
                if 0 <= im < Ny and 0 <= jm < Nx:
                    inside_minus = A[im, jm]

                if not (inside_plus or inside_minus):
                    keep = False
                    break

            B[i, j] = keep

    return B

# --- Bounding box for cropping plots ---
def bounding_box(A, x, y, margin=0):
    rows = np.any(A, axis=1)
    cols = np.any(A, axis=0)
    if not rows.any() or not cols.any():
        return A, x, y
    i_min, i_max = np.where(rows)[0][[0, -1]]
    j_min, j_max = np.where(cols)[0][[0, -1]]

    dx = x[1]-x[0]
    dy = y[1]-y[0]
    i_min = max(0, i_min - int(np.ceil(margin/dy)))
    i_max = min(A.shape[0]-1, i_max + int(np.ceil(margin/dy)))
    j_min = max(0, j_min - int(np.ceil(margin/dx)))
    j_max = min(A.shape[1]-1, j_max + int(np.ceil(margin/dx)))

    A_crop = A[i_min:i_max+1, j_min:j_max+1]
    x_crop = x[j_min:j_max+1]
    y_crop = y[i_min:i_max+1]
    return A_crop, x_crop, y_crop


def left_corner_crop(A, x, y, margin=0.1, width=None):
    """
    Crop the left part of the shape.

    Parameters
    ----------
    A : 2D bool array
        Shape array
    x, y : 1D arrays
        Grid coordinates
    margin : float
        Extra padding around the shape
    width : float
        Width of left corner to crop (in physical units). 
        If None, crops from leftmost pixel to the first width fraction of the x-range.
    """
    rows = np.any(A, axis=1)
    cols = np.any(A, axis=0)
    
    if not rows.any() or not cols.any():
        return A, x, y

    i_min, i_max = np.where(rows)[0][[0, -1]]
    j_min, j_max = np.where(cols)[0][[0, -1]]

    # Determine the left boundary
    if width is None:
        j_max_left = j_min + (j_max - j_min) // 3  # crop left third by default
    else:
        j_max_left = np.searchsorted(x, x[j_min] + width)
        j_max_left = min(j_max_left, j_max)

    # Add margin in indices
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    i_min_crop = max(0, i_min - int(np.ceil(margin / dy)))
    i_max_crop = min(A.shape[0]-1, i_max + int(np.ceil(margin / dy)))
    j_min_crop = max(0, j_min - int(np.ceil(margin / dx)))
    j_max_crop = min(A.shape[1]-1, j_max_left + int(np.ceil(margin / dx)))

    A_crop = A[i_min_crop:i_max_crop+1, j_min_crop:j_max_crop+1]
    x_crop = x[j_min_crop:j_max_crop+1]
    y_crop = y[i_min_crop:i_max_crop+1]

    return A_crop, x_crop, y_crop



import numpy as np
import matplotlib.pyplot as plt

def initialize_eye_piecewise_grid(X, Y, R, alpha):
    """
    Initialize the eye shape (circle + diamond wings) on an existing X,Y grid.
    
    Parameters:
        X, Y : 2D arrays
        R : float - circle radius
        alpha : float - diamond angle in radians
    
    Returns:
        eye_mask : 2D Boolean array
        area : float - approximate area on the grid
    """
    # Circle in the center
    circle_mask = X**2 + Y**2 <= R**2
    center_mask = np.abs(X) <= R * np.sin(alpha)
    center = center_mask & circle_mask

    # Right triangle (X > R*sin(alpha))
    mask_right = X > R * np.sin(alpha)
    y_top_right = -np.tan(alpha) * X + R*(1/np.cos(alpha))
    y_bottom_right = np.tan(alpha) * X - R*(1/np.cos(alpha))
    right_triangle = mask_right & (Y >= y_bottom_right) & (Y <= y_top_right)

    # Left triangle (X < -R*sin(alpha))
    mask_left = X < -R * np.sin(alpha)
    y_top_left = np.tan(alpha) * X + R*(1/np.cos(alpha))
    y_bottom_left = -np.tan(alpha) * X - R*(1/np.cos(alpha))
    left_triangle = mask_left & (Y >= y_bottom_left) & (Y <= y_top_left)

    # Combine center and wings
    eye_mask = center | right_triangle | left_triangle

    # Approximate area
    dx = X[0,1] - X[0,0]
    dy = Y[1,0] - Y[0,0]
    area = np.sum(eye_mask) * dx * dy

    print(f'Area of eye: {area}')
    return eye_mask



def simple_implementation(n_steps, R=R, Nx = resolution, Ny = resolution, alpha = alpha, num_plots = 12):
    '''
    Evolves the shape for n_steps many steps with a grid of resolution Nx x Ny.
    
    '''
    
    # Initialize grid and circle
    Lx = R * 2
    Ly = R * 1.2
    x, y, X, Y, dx, dy = create_grid(Lx, Ly, Nx, Ny)
    A = initialize_circle(X, Y, R)
    plots=[]
    corner_plots=[]
    areas=[]
    #Compute offsets
    offsets = compute_unit_offsets(dx=dx, dy=dy, alpha=alpha, num_angles=512)
    plot_steps = np.unique(np.linspace(1, n_steps, num_plots, dtype=int))
    for step in range(1, n_steps +1):
        #Expand x-axis by 2 units because we expect the shape to grow outwards.
        extra_pixels = int(np.ceil(2 / dx))
        A_exp = np.zeros((A.shape[0], A.shape[1]+2*extra_pixels), dtype = bool)
        A_exp[:, extra_pixels: extra_pixels + A.shape[1]] = A #Update interior to be equal to A
        A = A_exp

        #Update x coordinates to match
        x = np.linspace(x[0]-2, x[-1]+2, len(x)+2*extra_pixels)

        #Perform update
        A = update_step(A, offsets)

        #Area Computation
        area = np.sum(A) * (x[1]-x[0])*(y[1]-y[0])
        areas.append(area)
        print(f"Step {step}: Area = {area:.4f}")

        if step in plot_steps:
            # Crop to current shape + margin
            A_crop, x_crop, y_crop = bounding_box(A, x, y, margin=0.1)
            plots.append((A_crop.copy(), x_crop.copy(), y_crop.copy(), step, np.round(area, 2)))

            # left corner plot
            A_left, x_left, y_left = left_corner_crop(A, x, y, margin=0.1)
            corner_plots.append((A_left.copy(), x_left.copy(), y_left.copy(), step))
 

    area_array = np.array(areas)
    print(np.diff(area_array))
    # Display all plots together
    n_cols = 4
    n_rows = n_rows = int(np.ceil(num_plots/n_cols))
    plt.figure(figsize=(4*n_cols, 4*n_rows))
    for idx, (A_plot, x_plot, y_plot, step, area) in enumerate(plots):
        plt.subplot(n_rows, n_cols, idx+1)
        plt.imshow(A_plot, origin='lower', extent=[x_plot[0], x_plot[-1], y_plot[0], y_plot[-1]], cmap='Greys')
        plt.axis('equal')
        plt.title(f"Step {step}, Area: {area}")
    plt.tight_layout()
    plt.show()

    for idx, (A_plot, x_plot, y_plot, step) in enumerate(corner_plots):
        plt.subplot(n_rows, n_cols, idx+1)
        plt.imshow(A_plot, origin='lower', extent=[x_plot[0], x_plot[-1], y_plot[0], y_plot[-1]], cmap='Greys')
        plt.axis('equal')
        plt.title(f"Step {step}, Area: {area}")
    plt.tight_layout()
    plt.show()

    return A, areas, plots, corner_plots
    

def compute_area_difference(Nx = resolution, Ny = resolution, alpha = alpha, num_plots=10, n_steps = n_steps):
    '''
    Given as input the "eye"-shape, computes the area change in one update step. The base area is a fixed quantity, calculated at the beginning.
    
    '''
     # Initialize grid and circle
    Lx = R * 1.8
    Ly = R * 1.2
    x, y, X, Y, dx, dy = create_grid(Lx, Ly, Nx, Ny)
    A = initialize_eye_piecewise_grid(X, Y, R, alpha=alpha)
    plots=[]
    corner_plots=[]
    areas=[]
    #Compute offsets
    offsets = compute_unit_offsets(dx=dx, dy=dy, alpha=alpha, num_angles=512)
    plot_steps = np.unique(np.linspace(1, n_steps, num_plots, dtype=int))
    for step in range(1, n_steps +1):
        #Expand x-axis by 2 units because we expect the shape to grow outwards.
        extra_pixels = int(np.ceil(2 / dx))
        '''A_exp = np.zeros((A.shape[0], A.shape[1]+2*extra_pixels), dtype = bool)
        A_exp[:, extra_pixels: extra_pixels + A.shape[1]] = A #Update interior to be equal to A
        A = A_exp

        #Update x coordinates to match
        x = np.linspace(x[0]-2, x[-1]+2, len(x)+2*extra_pixels)'''

        dx = x[1] - x[0]

        x_left = x[0] - dx*np.arange(extra_pixels,0,-1)
        x_right = x[-1] + dx*np.arange(1,extra_pixels+1)

        x = np.concatenate([x_left, x, x_right])

        #Perform update
        A = update_step(A, offsets)

        #Area Computation
        area = np.sum(A) * (x[1]-x[0])*(y[1]-y[0])
        areas.append(area)
        print(f"Step {step}: Area = {area:.4f}")

        if step in plot_steps:
            # Crop to current shape + margin
            A_crop, x_crop, y_crop = bounding_box(A, x, y, margin=0.1)
            plots.append((A_crop.copy(), x_crop.copy(), y_crop.copy(), step, np.round(area, 2)))

            # left corner plot
            A_left, x_left, y_left = left_corner_crop(A, x, y, margin=0.1)
            corner_plots.append((A_left.copy(), x_left.copy(), y_left.copy(), step))
 

    area_array = np.array(areas)
    print(np.diff(area_array))
    # Display all plots together
    n_cols = 4
    n_rows = n_rows = int(np.ceil(num_plots/n_cols))
    plt.figure(figsize=(4*n_cols, 4*n_rows))
    for idx, (A_plot, x_plot, y_plot, step, area) in enumerate(plots):
        plt.subplot(n_rows, n_cols, idx+1)
        plt.imshow(A_plot, origin='lower', extent=[x_plot[0], x_plot[-1], y_plot[0], y_plot[-1]], cmap='Greys')
        plt.axis('equal')
        plt.title(f"Step {step}, Area: {area}")
    plt.tight_layout()
    plt.show()

compute_area_difference()



def until_zero_area(R=R, Nx = resolution, Ny = resolution, alpha = alpha, num_plots = 12):
    '''
    Evolves the shape for n_steps many steps with a grid of resolution Nx x Ny.
    
    '''
    # Initialize grid and circle
    Lx = R * 1.5
    Ly = R * 1.2
    x, y, X, Y, dx, dy = create_grid(Lx, Ly, Nx, Ny)
    A = initialize_circle(X, Y, R)
    plots=[]
    corner_plots=[]
    areas=[]
    #Compute offsets
    offsets = compute_unit_offsets(dx=dx, dy=dy, alpha=alpha, num_angles=512)
    plot_steps = np.unique(np.linspace(1, n_steps, num_plots, dtype=int))
    area=1
    step = 1
    while area>0:
        #Expand x-axis by 2 units because we expect the shape to grow outwards.
        extra_pixels = int(np.ceil(2 / dx))
        A_exp = np.zeros((A.shape[0], A.shape[1]+2*extra_pixels), dtype = bool)
        A_exp[:, extra_pixels: extra_pixels + A.shape[1]] = A #Update interior to be equal to A
        A = A_exp

        #Update x coordinates to match
        x = np.linspace(x[0]-2, x[-1]+2, len(x)+2*extra_pixels)

        #Perform update
        A = update_step(A, offsets)

        #Area Computation
        area = np.sum(A) * (x[1]-x[0])*(y[1]-y[0])
        areas.append(area)
        print(f"Step {step}: Area = {area:.4f}")

        if step in plot_steps:
            # Crop to current shape + margin
            A_crop, x_crop, y_crop = bounding_box(A, x, y, margin=0.1)
            plots.append((A_crop.copy(), x_crop.copy(), y_crop.copy(), step, np.round(area, 2)))

            # left corner plot
            A_left, x_left, y_left = left_corner_crop(A, x, y, margin=0.1)
            corner_plots.append((A_left.copy(), x_left.copy(), y_left.copy(), step))
        step+=1
 

    area_array = np.array(areas)
    print(np.diff(area_array))
    # Display all plots together
    n_cols = 4
    n_rows = n_rows = int(np.ceil(num_plots/n_cols))
    plt.figure(figsize=(10*n_cols, 4*n_rows))
    for idx, (A_plot, x_plot, y_plot, step, area) in enumerate(plots):
        plt.subplot(n_rows, n_cols, idx+1)
        plt.imshow(A_plot, origin='lower', extent=[x_plot[0], x_plot[-1], y_plot[0], y_plot[-1]], cmap='Greys')
        plt.axis('equal')
        plt.title(f"Step {step}, Area: {area}")
    plt.tight_layout()
    plt.show()

    for idx, (A_plot, x_plot, y_plot, step) in enumerate(corner_plots):
        plt.subplot(n_rows, n_cols, idx+1)
        plt.imshow(A_plot, origin='lower', extent=[x_plot[0], x_plot[-1], y_plot[0], y_plot[-1]], cmap='Greys')
        plt.axis('equal')
        plt.title(f"Step {step}, Area: {area}")
    plt.tight_layout()
    plt.show()

    return A, areas, plots, corner_plots





