import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar


# ============================================================
# Solve the homothetic differential equation
#
#     kappa = C r^2 / sqrt(r^2 + (r')^2)
#
# for a polar curve, using r = exp(f).
#
# This gives the ODE
#
#     f'' = (1 + (f')^2)(1 - C e^{2f})
#
# ============================================================


# ----------------------------
# Parameters
# ----------------------------

C = 1.0
L = np.pi / 2

angle = np.pi / 6
angle_name = "pi/6"

#This is the initial condition
target_slope = 1 / np.tan(angle)


# ----------------------------
# ODE for f
# ----------------------------

def ode(theta, y):
    f = y[0]
    fp = y[1]

    # Correct sign for:
    # kappa = C r^2 / sqrt(r^2 + r'^2)
    fpp = (1 + fp**2) * (1 - C * np.exp(2 * f))

    return [fp, fpp]


# ----------------------------
# Shooting condition -- this is necessary because we are given boundary conditions
# and want to solve an IVP
# ----------------------------
#
# Evenness gives f'(0) = 0.
# We choose f(0) = a so that
#
#     f'(pi/2) = 1 / tan(angle).

def shooting_error(a):
    sol = solve_ivp(
        ode,
        t_span=(0, L),
        y0=[a, 0.0],
        rtol=1e-10,
        atol=1e-12,
        max_step=0.001,
    )

    fp_at_L = sol.y[1, -1]

    return fp_at_L - target_slope


# ----------------------------
# Find f(0)
# ----------------------------
#
# For this corrected sign, the correct f(0)
# may lie below 0, so use a bracket with negative values.

result = root_scalar(
    shooting_error,
    bracket=[-3.0, 1.0],
    xtol=1e-12,
    rtol=1e-12,
)

if not result.converged:
    raise RuntimeError("Root search did not converge.")

a_star = result.root

print(f"Found f(0) = {a_star:.12f}")


# ----------------------------
# Solve final IVP
# ----------------------------

sol_pos = solve_ivp(
    ode,
    t_span=(0, L),
    y0=[a_star, 0.0],
    rtol=1e-10,
    atol=1e-12,
    max_step=0.001,
    dense_output=True,
)


# ----------------------------
# Build even solution on [-pi/2, pi/2]
# ----------------------------

theta_pos = np.linspace(0, L, 2000)
f_pos = sol_pos.sol(theta_pos)[0]
fp_pos = sol_pos.sol(theta_pos)[1]

theta_full = np.concatenate((-theta_pos[:0:-1], theta_pos))
f_full = np.concatenate((f_pos[:0:-1], f_pos))
fp_full = np.concatenate((-fp_pos[:0:-1], fp_pos))

r_full = np.exp(f_full)
rp_full = fp_full * r_full


# ----------------------------
# Check boundary condition
# ----------------------------

fp_at_L = sol_pos.sol(L)[1]

print(f"f'(pi/2) = {fp_at_L:.12f}")
print(f"target slope = {target_slope:.12f}")


# ----------------------------
# Compute polar curvature
# ----------------------------
#
# For r = exp(f):
#
#     r'  = f' r
#     r'' = (f'' + (f')^2) r

fpp_full = (1 + fp_full**2) * (1 - C * np.exp(2 * f_full))

rpp_full = (fpp_full + fp_full**2) * r_full

curvature_numerator = (
    r_full**2
    + 2 * rp_full**2
    - r_full * rpp_full
)

curvature_denominator = (
    r_full**2 + rp_full**2
) ** 1.5

kappa = curvature_numerator / curvature_denominator

kappa_formula = C * r_full**2 / np.sqrt(r_full**2 + rp_full**2)

print(f"min numerical curvature = {np.min(kappa):.12f}")
print(f"max numerical curvature = {np.max(kappa):.12f}")

print(f"min formula curvature = {np.min(kappa_formula):.12f}")
print(f"max formula curvature = {np.max(kappa_formula):.12f}")


# ----------------------------
# Rotated radial curve
# ----------------------------
#
# theta = 0 points along e_2:
#
#     x = r cos(theta + pi/2)
#     y = r sin(theta + pi/2)
#
# equivalently:
#
#     x = -r sin(theta)
#     y =  r cos(theta)

rotated_theta = theta_full + np.pi / 2

x_upper = r_full * np.cos(rotated_theta)
y_upper = r_full * np.sin(rotated_theta)


# ----------------------------
# Plot upper smooth curve only
# ----------------------------

plt.figure(figsize=(7, 7))
plt.plot(x_upper, y_upper, label="smooth homothetic curve")

plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.title(f"Smooth homothetic arc, angle {angle_name}")

plt.axis("equal")
plt.grid(True)


# Reference ray of length 1 starting at endpoint
corner_x = x_upper[-1]
corner_y = y_upper[-1]

dx = np.cos(angle)
dy = np.sin(angle)

x_line = np.array([corner_x, corner_x + dx])
y_line = np.array([corner_y, corner_y + dy])

plt.plot(
    x_line,
    y_line,
    "--",
    linewidth=2,
    label=rf"length-1 ray, slope $\tan({angle_name})$",
)

plt.scatter([corner_x], [corner_y], zorder=5, label="endpoint")

plt.legend()
plt.show()


# ----------------------------
# Plot mirrored closed shape
# ----------------------------

x_lower = x_upper
y_lower = -y_upper

x_closed = np.concatenate((x_upper, x_lower[::-1], [x_upper[0]]))
y_closed = np.concatenate((y_upper, y_lower[::-1], [y_upper[0]]))

plt.figure(figsize=(7, 7))
plt.plot(x_closed, y_closed, label="mirrored closed shape")

plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.title(f"Homothetic curve, {angle_name}")

plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()

