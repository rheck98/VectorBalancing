from numba import njit, prange
import numpy as np
import matplotlib.pyplot as plt
import bisect
import pandas as pd

#Parameters

alpha = np.pi/4
n_steps = 10
R = 25 # initial radius
dx = 0.005


#Helper functions!

def find_jumps(arr, threshold):
    '''
    Returns array of indices where difference between adjacent values is at least some threshold.
    '''
    arr = np.asarray(arr)
    diffs = np.abs(np.diff(arr))   # differences between consecutive elements
    jump_indices = np.where(diffs > threshold)[0]
    return jump_indices

def nearest_index(arr, value):
    """
    Returns the index of the element in sorted `arr`
    closest to `value`.
    """
    #if not arr:
        #raise ValueError("Array must not be empty")

    pos = bisect.bisect_left(arr, value)

    if pos == 0:
        return 0
    if pos == len(arr):
        return len(arr) - 1

    before = arr[pos - 1]
    after = arr[pos]

    if abs(value - before) <= abs(after - value):
        return pos - 1
    else:
        return pos
    

def interpolate_inner_zeros(arr):

    idx = np.nonzero(arr)[0]
    if len(idx) == 0:
        raise ValueError('Array only contains 0s')

    first_idx = idx[0]
    last_idx = idx[-1]

    s = pd.Series(arr, dtype=float)

    # interpolate only between first and last non-zero
    inner = s[first_idx:last_idx + 1].replace(0, np.nan).interpolate()

    result = s.copy()
    result[first_idx:last_idx + 1] = inner

    return result.to_numpy()




#Definition of curves--line, circle, and eye

def eye_curve(x):
    '''
    Gives the value of the eye function on the upper right side. Here x is the EUCLIDEAN input!
    '''
    if x > R*np.sin(alpha):
        return -np.tan(alpha) * x + R*(1/np.cos(alpha))
    else:
        return np.sqrt(R**2-x**2)

def generate_boundary_curve(R = R, alpha = alpha, plot = False, dx=dx):
    '''
    Given parameters R and alpha, generates the boundary curve for the "eye shape" as a list of values
    for evenly spaced x.
    '''
    Lx = R*1/np.sin(alpha)
    n_left = int(np.round(2 / dx))
    n_right = int(np.round(Lx / dx))

    x = np.arange(-n_left, n_right + 1) * dx
    f_x = np.vectorize(eye_curve)(x) #applies the function eye_curve to each element of x.

    if plot == True:
        plt.figure()
        plt.scatter(x, f_x)
        plt.axis('equal')
        plt.show()

    return -x[::-1], f_x[::-1], dx

#Same for circle

def circle(x):
    inside = R**2 - x**2
    return np.sqrt(np.maximum(inside, 0))

def generate_circle(R = R, dx=dx):
    '''
    Given parameters R and dx, generates circle of radius R with Euclidean spacing dx.
    '''

    Lx = R
    n_left = int(np.round(2 / dx))
    n_right = int(np.round(Lx / dx))

    x = np.arange(-n_left, n_right + 1) * dx
    f_x = np.vectorize(circle)(x) #applies the function circle to each element of x.


    '''plt.figure()
    plt.scatter(-x[::-1], f_x[::-1])
    plt.axis('equal')
    plt.show()'''

    return -x[::-1], f_x[::-1], dx

#Same for line

def line(x):
    '''
    Gives the value of the eye function on the upper right side. Here x is the EUCLIDEAN input!
    '''

    return -np.tan(alpha) * np.abs(x) + R / np.cos(alpha)

def generate_line(R=R, alpha=alpha, dx=dx):
    '''
    Given parameters R and alpha, generates a line (with reflection) at angle alpha.
    '''

    Lx = R*1/np.sin(alpha)
    n_left = int(np.round(2 / dx))
    n_right = int(np.round(Lx / dx))

    x = np.arange(-n_left, n_right + 1) * dx
    f_x = np.vectorize(line)(x) #applies the function line to each element of x.

    plt.figure()
    plt.scatter(-x[::-1], f_x[::-1])
    plt.axis('equal')
    plt.show()
    return -x[::-1], f_x[::-1], dx


x, f_x, dx = generate_boundary_curve()
#x, f_x, dx = generate_circle()
#x, f_x, dx = generate_line()


def get_inner_outer(x=x, f_x=f_x, dx=dx, plot_all = False):
    '''
    Computes both inner and outer approximations of the shape at each step through a double pass argument.
    '''
    #x, f_x, dx = generate_boundary_curve(R = R, alpha = alpha, res = resolution, plot = False)
    #Step 1: away from the corners, do the same approximation

    if plot_all:
        plt.figure()
        plt.title('Original Boundary')
        plt.scatter(x, f_x)
        plt.show()

    #this is the new boundary array that we will update as we step.
    new_boundary = np.zeros_like(f_x)
    if len(np.where(f_x!=0)[0])!=0: #if there are leading zeros, from the curve shortening until now
        start = np.where(f_x!=0)[0][0] #then start the computation from the first non-zero value
        print(f'Start:{start}') #Sanity check that this is moving to the right.
    else:
        start=0

    #Compute the update for the inner portion of the curve, away from the corners
    for i in range(start-1, len(x)):
        for j in range(i+1, len(x)):
            if f_x[j]<np.sqrt(4-(x[j]-x[i])**2)+f_x[i]:
                pass
            else:
                mid_point = (x[j]+x[i])/2
                idx_mp = np.round((mid_point-x[0])/dx).astype(int)
                idx_mp = np.clip(idx_mp, 0, len(x)-1)
                new_boundary[idx_mp] = (f_x[j]+f_x[i])/2
                break
    if plot_all:
        plt.figure()
        plt.title('First pass, with missing values')
        plt.scatter(x,new_boundary)
        plt.show()


    idx = np.nonzero(new_boundary)[0]
    if len(idx !=0) and (len(x) - idx[0]) != 0:
        new_boundary = interpolate_inner_zeros(new_boundary)
    
    #Update end of boundary using symmetry, due to missing values at the end
    zero_idx = np.argmin(np.abs(x))
    right_len = len(new_boundary) - zero_idx
    left_block = new_boundary[:zero_idx]
    mirror_len = min(len(left_block), right_len)

    reversed_segment = left_block[-mirror_len:][::-1]

    new_boundary = new_boundary.copy()
    new_boundary[zero_idx:zero_idx + mirror_len] = reversed_segment

    #Everything above gives the inner part away from the corners, replacing the end as well

    if plot_all:
        plt.figure()
        plt.scatter(x,new_boundary)
        plt.title('Away from Corners')
        plt.show()

    #This is the last index where we have a zero from the middle approximation, which is the upper approximation
    top_end = np.where(new_boundary == 0)[0][-1]
    print(f'Upper end point: {top_end}')

    #Now we need to calculate the point UNTIL which the bottom curve plays a role, and check whether it is before or after top_end

    bottom_end=0 #safeguard for when things run out
    for i in range(len(x)):
        #find the value of x that is at the desired shift.
        next_x = nearest_index(x, x[i]+2*np.cos(alpha))
        #check if we have crossed the upper boundary yet:
        if -f_x[i]+2*np.sin(alpha) >= f_x[next_x]:
            pass
        else:
            bottom_end = nearest_index(x, (x[i]+np.cos(alpha))) #this can cause problems later--be careful!
            break
    print(f'Lower end point: {bottom_end}')

    if top_end<=bottom_end:
        print('Good!')
    else:
        print('Bad!')

    #We need to go up to next_x in order to fill in the missing zeros at the beginning

    #First, find the nearest index to this shifted value to start filling values.
    shifted_top_end = nearest_index(x, x[top_end]+np.cos(alpha))

    #We will fill this array with the estimates near the corner
    near_corner = np.zeros_like(f_x)

    current_x = nearest_index(x, x[start]+np.cos(alpha))
    while current_x <= shifted_top_end:
        idx = nearest_index(x, x[current_x]-np.cos(alpha))
        near_corner[idx] = f_x[current_x]-np.sin(alpha)
        current_x+=1
        if near_corner[idx]<0:
            near_corner[idx]=0 #!!!

        #Now we have the two estimates, away from the corner and near the corner. There are two cases to consider: they overlap, or they don't overlap. 

    #Case 1: Overlap
    if top_end == len(new_boundary)-1: #deal with edge case
        top_end = top_end-1

    if top_end <= bottom_end:
    #We start by over-estimating--that is, taking the center estimate for the overlap. This means that from top_end, we replace with the upper estimate
        over = near_corner.copy()

        over[top_end+1:] = new_boundary[top_end+1:]
        
        under = near_corner.copy()
        under[bottom_end:] = new_boundary[bottom_end:]
        under = interpolate_inner_zeros(under) #necessary because of rounding--some internal 0s may show up.

    else: #this is the case where they do NOT overlap, in which case we linearly interpolate between the two 
        over = near_corner.copy()
        over[bottom_end:top_end+1]=np.linspace(near_corner[bottom_end], new_boundary[top_end+1], top_end-bottom_end+1)
        over[bottom_end:]=new_boundary[bottom_end:]

        under = near_corner.copy()
        under[bottom_end: top_end+1]=np.linspace(near_corner[bottom_end], new_boundary[top_end+1], top_end-bottom_end+1)
        under[top_end+1:]=new_boundary[top_end+1]

    if plot_all:
        plt.figure()
        #plt.scatter(x, over)
        plt.scatter(x, under)
        plt.title('Over, Under')
        plt.show()

    return x, over, under

#get_inner_outer(plot_all=True)

def iterate_inner_outer(x=x, f_x=f_x, dx=dx, n_steps=30, plot_all=False):
    zero_idx = np.argmin(np.abs(x))
    
    upper = f_x.copy()
    lower = f_x.copy()

    upper_history = [upper.copy()]
    upper_diffs = []
    lower_history = [lower.copy()]
    lower_diffs = []

    for _ in range(n_steps):
        # iterate upper approximation
        original_upper = upper.copy()
        x, upper, _ = get_inner_outer(x=x, f_x=upper, dx=dx, plot_all=False)
        if plot_all:
            plt.figure()
            plt.title(f'Upper {i}')
            plt.scatter(x, upper)
            plt.show()

        upper_diffs.append(np.sum(original_upper[:zero_idx]-upper[:zero_idx])*4*dx)
        # iterate lower approximation
        original_lower = lower.copy()
        x, _, lower = get_inner_outer(x=x, f_x=lower, dx=dx)
        lower_diffs.append(np.sum(original_lower[:zero_idx]-lower[:zero_idx])*4*dx)
        if plot_all:
            plt.figure()
            plt.title(f'Lower {i}')
            plt.scatter(x, lower)
            plt.show()

        upper_history.append(upper.copy())
        lower_history.append(lower.copy())
    print(f'Lower approx. area changes: {lower_diffs}')
    print(f'Upper approx. area changes: {upper_diffs}')

    print(f'Lower approx. avg: {np.mean(lower_diffs)}, Lower approx. std dev: {np.std(lower_diffs)}')
    print(f'Upper approx. avg: {np.mean(upper_diffs)}, Lower approx. std dev: {np.std(upper_diffs)}')
    breakpoint()
    return x, upper_history, lower_history


# Run iterations
x, upper_history, lower_history = iterate_inner_outer(
    x, f_x, dx, n_steps=25
)

# Plot upper iterations
plt.figure(figsize=(10, 6))

for i, upper in enumerate(upper_history):
    #if i%10==0:
    plt.plot(x, upper, label=f'Upper {i}')

plt.title('Upper Approximations')
plt.legend()
plt.axis('equal')
plt.show()

# Plot lower iterations
plt.figure(figsize=(10, 6))

for i, lower in enumerate(lower_history):
    #if i%10==0:
    plt.plot(x, lower, label=f'Lower {i}')


plt.title('Lower Approximations')
plt.legend()
plt.axis('equal')
plt.show()







'''
In the following code I will debug the previous code--what we know for sure is that far from the corner, the two points should be the same!
'''
