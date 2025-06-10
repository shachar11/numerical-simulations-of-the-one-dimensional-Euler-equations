import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton
import csv
import os
import time

################## Global functions and variables ##################

# Constants
gamma = 1.4  # Ratio of specific heats for air
N=100
x_min=0
x_max=1

# Define the initial conditions for the Riemann problem
def inital_conditions(case):
    if case==1:
        rho_L = 1.0  # Density on the left
        p_L = 1.0  # Pressure on the left
        u_L = 0.75  # Velocity on the left

        rho_R = 0.125  # Density on the right
        p_R = 0.1  # Pressure on the right
        u_R = 0.0  # Velocity on the right

        t_max=0.2
        x0=0.3 # Point of discontinuity

    else: #case==4
        rho_L = 5.99924  # Density on the left
        p_L = 460.894  # Pressure on the left
        u_L = 19.5975  # Velocity on the left

        rho_R = 5.99242  # Density on the right
        p_R = 46.0950  # Pressure on the right
        u_R = -6.19633 # Velocity on the right

        t_max=0.035
        x0=0.4 # Point of discontinuity

    return rho_L, p_L, u_L, rho_R, p_R, u_R, t_max, x0

# Compute conserved variables U = [rho, rho*u, E]
def primitive_to_conserved(rho, u, p):
    E = p / (gamma - 1) + 0.5 * rho * u**2
    return np.array([rho, rho * u, E])

# Compute primitive variables rho, u, p
def conserved_to_primitive(U):
    rho = U[0]
    u = U[1] / rho
    p = (gamma - 1) * (U[2] - 0.5 * rho * u**2)
    return rho, u, p

# Compute the sound speed a
def compute_a_sound_speed(U):
    rho, _, p = conserved_to_primitive(U)
    a= np.sqrt(gamma * p / rho)
    return a

# Estimate the maximum speed of the wave
def Max_Speed(U,i):
    a = compute_a_sound_speed(U)
    _, u, _=conserved_to_primitive(U)
    return np.abs(u) + a

# Compute the flux F(U)
def flux_from_U(U):
    rho, u, p = conserved_to_primitive(U)
    F = np.array([
        U[1],
        (rho * u**2) + p,
        u * (U[2] + p)
    ])
    return F

################## Functions for the exact solution ##################
# Compute a helpul parameter W
def compute_W(x_t, a_L, a_R, I, S, SH, ST,rho_L, p_L, u_L, rho_R, p_R, u_R, state):

    p_star=compute_p_star(a_L, a_R, p_L, u_L, rho_L, p_R, u_R, rho_R)
    u_star=compute_u_star(p_star, a_L, a_R, rho_L, p_L, u_L, rho_R, p_R, u_R)
    rho_star=compute_rho_star(p_star, rho_L, p_L, rho_R, p_R, I)

    if x_t < u_star:
        if state=="shock":
            if x_t < S:
                W= np.array([rho_L, u_L, p_L])

            else: # x_t > S
                W=np.array([rho_star, u_star, p_star])

        else: # state=="rarefaction"
            if x_t < SH:
                W= np.array([rho_L, u_L, p_L])
            elif SH < x_t < ST:
                W=compute_W_fan(x_t, a_L, rho_L, p_L, u_L, rho_R, p_R, u_R, "L")
            else: # ST < x_t < u_star
                W= np.array([rho_star, u_star, p_star])

    else: # x_t > u_star
        if state=="shock":
            if x_t > S:
                W= np.array([rho_R, u_R, p_R])

            else: # x_t < S
                W=np.array([rho_star, u_star, p_star])

        else: # state=="rarefaction"
            if x_t > SH:
                W= np.array([rho_R, u_R, p_R])
            elif ST < x_t < SH:
                W=compute_W_fan(x_t, a_R, rho_L, p_L, u_L, rho_R, p_R, u_R, "R")
            else: # u_star < x_t < ST
                W= np.array([rho_star, u_star, p_star])
                
                
    return W

# Compute W_fan
def compute_W_fan(x_t, a, rho_L, p_L, u_L, rho_R, p_R, u_R, I):
    if I=="L":
        a_L=a
        rho= rho_L*((2/(gamma+1)) + ((gamma-1)/((gamma+1)*a_L))*(u_L-x_t))**(2/(gamma-1))
        u= 2/(gamma+1) * (a_L + (gamma-1)/2 * u_L + x_t)
        p= p_L * (2/(gamma+1) + ((gamma-1)/((gamma+1)*a_L))*(u_L-x_t))**(2*gamma/(gamma-1))

    else: # I=="R"
        a_R=a
        rho= rho_R*((2/(gamma+1)) - ((gamma-1)/((gamma+1)*a_R))*(u_R-x_t))**(2/(gamma-1))
        u= 2/(gamma+1) * (-a_R + (gamma-1)/2 * u_R + x_t)
        p= p_R * (2/(gamma+1) - ((gamma-1)/((gamma+1)*a_R))*(u_R-x_t))**(2*gamma/(gamma-1))

    return np.array([rho, u, p])

# Is the wave a shock or a rarefaction wave?
def is_shock_or_rarefaction_wave(p, p_I, I):
    if I=="L":
        p_L=p_I
        if p>p_L:
            return "shock"
        else: # p<p_L
            return "rarefaction"
        
    else: # I=="R"
        p_R=p_I
        if  p>p_R:
            return "shock"
        else: # p<p_R
            return "rarefaction"

# Compute a helpul parameter A
def compute_A(rho_i):
    A=2/((gamma+1)*rho_i)
    return A

# Compute a helpul parameter B
def compute_B(p_i):
    B=((gamma-1)/(gamma+1))*p_i
    return B

# Compute the funcfion f whos root is the solution for p_star
def compute_f_and_df(p, a, rho_L, p_L, rho_R, p_R, I):
    if I=="L":
        a_L=a
        A_L=compute_A(rho_L)
        B_L=compute_B(p_L)
        state=is_shock_or_rarefaction_wave(p, p_L, "L")
        if state=="shock": # shock wave
            f=(p-p_L)*(A_L/(p+B_L))**0.5
            df=(A_L/(p+B_L))**0.5 * (1 - (p-p_L)/(2*(p+B_L))) 

        else: #  rarefaction wave
            f=(2*a_L/(gamma-1))*(((p/p_L)**((gamma-1)/2/gamma)) -1)
            df=1/(rho_L*a_L) * (p/p_L)**(-(gamma+1)/(2*gamma)) 
                
    else: # I=="R"
        a_R=a
        A_R=compute_A(rho_R)
        B_R=compute_B(p_R)
        state=is_shock_or_rarefaction_wave(p, p_R, "R")

        if state=="shock": # shock wave
            f=(p-p_R)*(A_R/(p+B_R))**0.5
            df=(A_R/(p+B_R))**0.5 * (1 - (p-p_R)/(2*(p+B_R)))

        else: # rarefaction wave
            f=(2*a_R/(gamma-1))*((p/p_R)**(((gamma-1)/2/gamma)) -1)
            df=1/(rho_R*a_R) * (p/p_R)**(-(gamma+1)/(2*gamma))

    return f, df

# Compute u_star - the velocity of the contact discontinuity
def compute_u_star(p_star, a_L, a_R, rho_L, p_L, u_L, rho_R, p_R, u_R):
    f_R,_=compute_f_and_df(p_star, a_R, rho_L, p_L, rho_R, p_R, "R")
    f_L,_=compute_f_and_df(p_star, a_L, rho_L, p_L, rho_R, p_R, "L")
    u_star=0.5*(u_L + u_R) + 0.5*(f_R - f_L)
    return u_star

# Compute p_star exact - the pressure at the contact discontinuity
def compute_p_star(a_L, a_R, p_L, u_L, rho_L, p_R, u_R, rho_R):
    func = lambda p: compute_f_and_df(p, a_L, rho_L, p_L, rho_R, p_R, "L")[0] + compute_f_and_df(p, a_R, rho_L, p_L, rho_R, p_R, "R")[0]  + u_R - u_L # Extracts f
    dfunc = lambda p: compute_f_and_df(p, a_L, rho_L, p_L, rho_R, p_R, "L")[1] + compute_f_and_df(p, a_R, rho_L, p_L, rho_R, p_R, "R")[1] # Extracts df
    p_guess = (p_L + p_R) / 2
    p_star = newton(func, x0=p_guess, fprime=dfunc, tol=1e-6, maxiter=1000)
    #print(f"p_star: {p_star:.6f}")
    return p_star

# Compute rho_star - the density at the contact discontinuity
def compute_rho_star(p_star,rho_L, p_L, rho_R, p_R, I):
    if I=="L":
        state= is_shock_or_rarefaction_wave(p_star, p_L, "L")
        if state=="shock":
            rho_star=rho_L*(((p_star/p_L) + ((gamma-1)/(gamma+1)))/((p_star/p_L)*((gamma-1)/(gamma+1)) + 1))
        else: # rarefaction wave
            rho_star=rho_L*((p_star/p_L)**(1/gamma))

    else: # I=="R" 
        state= is_shock_or_rarefaction_wave(p_star, p_R, "R")
        if state=="shock":
            rho_star=rho_R*(((p_star/p_R) + ((gamma-1)/(gamma+1)))/((p_star/p_R)*((gamma-1)/(gamma+1)) + 1))
        else: #rarefaction wave
            rho_star=rho_R*((p_star/p_R)**(1/gamma))

    return rho_star

# Compute speed of wave front S
def compute_S(p_star, a, u_star, p_L, u_L, p_R, u_R, I):
    if I=="L":
        a_L=a
        state= is_shock_or_rarefaction_wave(p_star, p_L, "L")
        if state=="shock":
            S=u_L - a_L*(((gamma+1)/(2*gamma))*(p_star/p_L) + ((gamma-1)/(2*gamma)))**0.5
            SH, ST= None, None

        else: #rarefaction wave
            a_star=a_L*((p_star/p_L)**((gamma-1)/(2*gamma)))
            SH=u_L - a_L
            ST=u_star - a_star
            S= None

    else: # I=="R"
        a_R=a
        state= is_shock_or_rarefaction_wave(p_star, p_R, "R")
        if state=="shock":
            S=u_R + a_R*(((gamma+1)/(2*gamma))*(p_star/p_R) + ((gamma-1)/(2*gamma)))**0.5
            SH, ST= None, None

        else: #rarefaction wave
            a_star=a_R*((p_star/p_L)**((gamma-1)/(2*gamma)))
            SH=u_R + a_R
            ST=u_star + a_star
            S= None

    return S, SH, ST, state

################## Functions for HLLC solver ##################

# Compute the pressure estimate at the contact discontinuity for HLLC
def p_star_estimate(rho_L, p_L, u_L, a_L, rho_R, p_R, u_R, a_R):
    rho_avg=(rho_L + rho_R)/2
    a_avg=(a_L + a_R)/2
    p_avg=(p_L + p_R)/2
    p_pvrs=p_avg - 0.5*(u_R - u_L)*rho_avg*a_avg
    p_star=max(0, p_pvrs)
    return p_star

# Compute q for HLLC
def compute_q(p_star, p):
    
    if p_star<=p:
        q=1
    else: # p_star>p
        q=(1 + (((gamma+1)/(2*gamma))*(p_star/p -1)))**0.5

    return q

# Compute the wave speeds estimates for HLLC
def wave_speed_estimates(p_L, u_L, a_L, q_L, p_R, u_R, a_R, q_R):
   
    S_L = u_L - a_L*q_L
    S_R = u_R + a_R*q_R
    S_star=(p_R-p_L+rho_L*u_L*(S_L-u_L)- rho_R*u_R*(S_R-u_R))/(rho_L*(S_L-u_L)-rho_R*(S_R-u_R))

    return S_L, S_star, S_R

# Compute the flux in star riegon for HLLC
def F_star(U, U_star, S):
    F_star=flux_from_U(U) + S*(U_star - U)
    return F_star

# Compute the flux for HLLC
def compute_flux_for_HLLC(U_L, U_star_L, U_R, U_star_R, S_L, S_star, S_R):

    if S_L >= 0:
        F_L=flux_from_U(U_L)
        F_HLLC=F_L
    elif S_star >= 0:
        F_L_star=F_star(U_L, U_star_L, S_L)
        F_HLLC=F_L_star
    elif S_R >= 0:
        F_R_star=F_star(U_R, U_star_R, S_R)
        F_HLLC=F_R_star
    else: # S_R < 0:
        F_R=flux_from_U(U_R)
        F_HLLC=F_R

    return F_HLLC

# Compute U in star riegon for HLLC
def compute_U_star(rho, p, u, S, S_star, U_I):
    E=U_I[2]
    scalar=rho*((S-u)/(S-S_star))
    U_star=scalar * np.array([
        1,
        S_star,
        E/rho + (S_star-u)*(S_star + (p/(rho*(S-u))))
    ])
    return U_star

################## Functions for Roe-pike solver ##################

# Compute enthalpy H for Roe-Pike
def compute_H(rho, p, u):
    U= primitive_to_conserved(rho, u, p)
    H = (U[2] + p) / rho
    return H

# Compute avrage values for Roe-Pike
def compute_avg_values(rho_L, p_L, u_L, rho_R, p_R, u_R):
    rho_avg = (rho_L*rho_R)**0.5
    u_avg = ((rho_L**0.5 * u_L) + (rho_R**0.5 * u_R)) / (rho_L**0.5 + rho_R**0.5)
    H_L = compute_H(rho_L, p_L, u_L)
    H_R = compute_H(rho_R, p_R, u_R)
    H_avg = ((rho_L**0.5 * H_L) + (rho_R**0.5 * H_R)) / (rho_L**0.5 + rho_R**0.5)
    a_avg = ((gamma-1)*(H_avg - 0.5*(u_avg**2)))**0.5
    return rho_avg, u_avg, H_avg, a_avg

# Compute the eigenvalues for Roe-Pike
def compute_lambda(u_avg, a_avg):
    lambda_vector = np.zeros(3)
    lambda_vector[0] = u_avg - a_avg
    lambda_vector[1] = u_avg
    lambda_vector[2] = u_avg + a_avg
    return lambda_vector

# Compute the right eigenvectors for Roe-Pike
def compute_K(u_avg, H_avg, a_avg):
    K_vector_1 = np.zeros(3)
    K_vector_1[0] = 1
    K_vector_1[1] = u_avg - a_avg
    K_vector_1[2] = H_avg - (u_avg * a_avg)

    K_vector_2 = np.zeros(3)
    K_vector_2[0] = 1
    K_vector_2[1] = u_avg
    K_vector_2[2] = 0.5 * (u_avg**2)

    K_vector_3 = np.zeros(3)
    K_vector_3[0] = 1
    K_vector_3[1] = u_avg + a_avg
    K_vector_3[2] = H_avg + (u_avg * a_avg)

    K_matrix = np.array([K_vector_1, K_vector_2, K_vector_3])
    
    return K_matrix

# Compute alpha for Roe-Pike
def compute_alpha(rho_L, p_L, u_L, rho_R, p_R, u_R, rho_avg, a_avg):
    alpha_1 = ((p_R - p_L) - (rho_avg * a_avg * (u_R - u_L))) / (2 * a_avg**2)
    alpha_2 = (rho_R - rho_L) - ((p_R - p_L) / a_avg**2)
    alpha_3 = ((p_R - p_L) + (rho_avg * a_avg * (u_R - u_L))) / (2 * a_avg**2)

    alpha_vector = np.array([alpha_1, alpha_2, alpha_3])

    return alpha_vector

################## Solvers for the Riemann problem ##################

# Exact solver for the Riemann problem
def solve_exact(rho_L, p_L, u_L, rho_R, p_R, u_R, t, x0):
    
    x = np.linspace(x_min, x_max, N)  # Spatial grid
    U_exact = np.zeros((N, 3))  # Initialize the array for conserved variables
    for i in range(N):
        x_i = x[i] - x0  # Centered at point of discontinuity
        if t==0:
            x_t = 0
        else:
            x_t = x_i / t_max  # Characteristic speed x/t
        U_exact[i] = Solve_Riemann_exact(rho_L, p_L, u_L, rho_R, p_R, u_R, x_t)

    return x, U_exact

# HLLC solver for the Riemann problem
def HLLC(rho_L, p_L, u_L, rho_R, p_R, u_R):
    # Compute L and R state parameters
    U_L = primitive_to_conserved(rho_L, u_L, p_L)
    U_R = primitive_to_conserved(rho_R, u_R, p_R)
    a_L = compute_a_sound_speed(U_L)
    a_R = compute_a_sound_speed(U_R)
    #print(f"U_L: {U_L} U_R: {U_R} F_L: {F_L} F_R: {F_R} a_L: {a_L} a_R: {a_R}")
    # Compute the pressure estimate at the contact discontinuity
    p_star = p_star_estimate(rho_L, p_L, u_L, a_L, rho_R, p_R, u_R, a_R)
    
    #print(f"p_star: {p_star}")
    # Compute paramter q
    q_L = compute_q(p_star, p_L)
    q_R = compute_q(p_star, p_R)
    #print(f"q_L: {q_L} q_R: {q_R}")
    # Compute the wave speeds estimates
    S_L, S_star, S_R = wave_speed_estimates(p_L, u_L, a_L, q_L, p_R, u_R, a_R, q_R)
    #print(f"S_L: {S_L} S_star: {S_star} S_R: {S_R}")
    # Compute U_star
    U_star_L = compute_U_star(rho_L, p_L, u_L, S_L, S_star, U_L)
    U_star_R = compute_U_star(rho_R, p_R, u_R, S_R, S_star, U_R)
    #print(f"U_star_L: {U_star_L} U_star_R: {U_star_R}")
    # Compute the fluxes
    F_HLLC = compute_flux_for_HLLC(U_L, U_star_L, U_R, U_star_R, S_L, S_star, S_R)
    #print(f"F_HLLC: {F_HLLC}")
    return F_HLLC

# Roe-Pike solver for the Riemann problem
def Roe_Pike(rho_L, p_L, u_L, rho_R, p_R, u_R):
    U_L= primitive_to_conserved(rho_L, u_L, p_L)
    U_R= primitive_to_conserved(rho_R, u_R, p_R)
    rho_avg, u_avg, H_avg, a_avg = compute_avg_values(rho_L, p_L, u_L, rho_R, p_R, u_R)
    lambda_vector = compute_lambda(u_avg, a_avg)
    K_matrix = compute_K(u_avg, H_avg, a_avg)
    alpha_vector = compute_alpha(rho_L, p_L, u_L, rho_R, p_R, u_R, rho_avg, a_avg)

    F_avg= 0.5*(flux_from_U(U_L) + flux_from_U(U_R))
    for i in range(3):
        F_avg -= 0.5 * alpha_vector[i] * abs(lambda_vector[i]) * K_matrix[i]

    return F_avg

################## Main simulation function ##################

# Exact solution for the Riemann problem
def Solve_Riemann_exact(rho_L, p_L, u_L, rho_R, p_R, u_R, x_t):

    U_L = primitive_to_conserved(rho_L, u_L, p_L)
    U_R = primitive_to_conserved(rho_R, u_R, p_R)
    a_L = compute_a_sound_speed(U_L)
    a_R = compute_a_sound_speed(U_R)

    # Solve for p_star, u_star, and rho_star
    p_star = compute_p_star(a_L, a_R, p_L, u_L, rho_L, p_R, u_R, rho_R)
    u_star = compute_u_star(p_star, a_L, a_R, rho_L, p_L, u_L, rho_R, p_R, u_R)
    
    S_L, SH_L, ST_L, state_L = compute_S(p_star, a_L, u_star, p_L, u_L, p_R, u_R, "L")
    S_R, SH_R, ST_R, state_R = compute_S(p_star, a_R, u_star, p_L, u_L, p_R, u_R, "R")
         
        
    if x_t < u_star:  # Left region
        W = compute_W(x_t, a_L, a_R, "L", S_L, SH_L, ST_L, rho_L, p_L, u_L, rho_R, p_R, u_R, state_L)
    else:  # Right region
        W = compute_W(x_t, a_L, a_R, "R", S_R, SH_R, ST_R, rho_L, p_L, u_L, rho_R, p_R, u_R, state_R)
        
    rho, u, p = W

    # Convert to conserved variables
    U = primitive_to_conserved(rho, u, p)

    return U

#Simulate using Godunov method
def Gondunov_method(case, solver):
    start_time = time.time()

    # Define starting time and counter
    t = 0
    counter = 0

    # Define grid step size
    dx=(x_max-x_min)/N
    x_half= np.linspace(x_min+dx/2, x_max-dx/2, N)
    x= np.linspace(x_min, x_max, N)
    #initalize conditions
    rho_L, p_L, u_L, rho_R, p_R, u_R, t_max, x0 = inital_conditions(case)

    # Initalize U_array
    U_L = primitive_to_conserved(rho_L, u_L, p_L)
    U_R = primitive_to_conserved(rho_R, u_R, p_R)
    U_array = np.zeros((N, 3))  # Initialize the array for conserved variables
    F_array = np.zeros((N+1, 3))
    for i in range(N):
        x_i = x[i]
        if x_i < x0:
            U_array[i] = U_L
        else:
            U_array[i] = U_R
    #print(f"U_array: {U_array}")

    # Main loop
    while t <= t_max:
        if counter <5:
            CFL=0.9*0.2
        else:
            CFL=0.9
        
        max_speed_i = np.max([Max_Speed(U_array[i],i) for i in range(N)])
        
        dt = CFL* dx / max_speed_i  # Time step size
        
        U_new= np.zeros_like(U_array) #create a new array to store the updated values

        for i in range(1, N):

            rho_L, u_L, p_L = conserved_to_primitive(U_array[i-1])
            rho_R, u_R, p_R = conserved_to_primitive(U_array[i])
            
            if solver=="Exact":
                U = Solve_Riemann_exact(rho_L, p_L, u_L, rho_R, p_R, u_R, 0) #x_t=0
                F_array[i] = flux_from_U(U)

            elif solver=="HLLC":
                F_array[i] = HLLC(rho_L, p_L, u_L, rho_R, p_R, u_R)

            elif solver=="Roe-Pike":
                F_array[i] = Roe_Pike(rho_L, p_L, u_L, rho_R, p_R, u_R)

        U_new = U_array - (dt/dx) * (F_array[1:] - F_array[:-1])
        U_new[0] = U_array[0]
        U_new[-1] = U_array[-1]

        # Update the solution
        U_array=U_new.copy()

        # Update time
        t += dt
        if t + dt > t_max:
            dt = t_max - t
        
        counter += 1

        # print(f"Time step {counter}: t = {t:.6f}")
        # print(f"F_array: {F_array}")
    end_time = time.time()
    print(f"Time taken for case {case} with {solver} solver: {end_time - start_time:.5f} seconds")
    return x_half, U_array

################## Visualization functions ##################

# Extract data from torro files
def Extract_torro_data(case, variables, solver):
    torro_data = {}
    
    if solver=="Exact Riemann solution":
        for var in variables:
            # Read exact data
            exact_file = os.path.join('torro_exact_solution',f'torro_case_{case}_{var}_exact.csv')
            x_values, y_values = [], []
            with open(exact_file, 'r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if row:
                        x_values.append(float(row[0]))
                        y_values.append(float(row[1]))
            torro_data[var] = (x_values, y_values)

    
    else:
        directory_path=f'torro_{solver}_solver'
        for var in variables:
            # Read numerical data
            num_file = os.path.join(directory_path,f'torro_case_{case}_{var}_numerical.csv')
            x_values, y_values = [], []
            with open(num_file, 'r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if row:
                        x_values.append(float(row[0]))
                        y_values.append(float(row[1]))

            torro_data[var] = (x_values, y_values)

    return torro_data

# Calculate internal energy and set up data dictionary
def calculate_e_and_set_up_data_dictionary(U_array):
    # Convert to primitives in vectorized fashion
    rho, u, p = np.array([conserved_to_primitive(U) for U in U_array]).T

    # Compute internal energy
    e = p / ((gamma - 1) * rho)

    # Package into dictionaries
    data = {
        "density": rho,
        "velocity": u,
        "pressure": p,
        "energy": e
    }

    return data

# Plot results for each case and solver
def plot_results_per_case_and_solver(x, x_half, torro_data_exact_solution, all_data_dic, case, variables, solver):
    
    plt.figure(figsize=(14, 12))

    torro_data=Extract_torro_data(case, variables, solver)
    plt.title(f"Case {case} - Riemann slover - {solver}")
    plt.axis("off")

    for i, var in enumerate(variables):
        plt.subplot(2, 2, i+1)
        plt.plot(x_half, all_data_dic[case][solver][var], label=f"{solver} solver", marker='o', markersize=2, linestyle='')
        plt.plot(x, all_data_dic[case]["Exact Riemann solution"][var], label="Exact Riemann solution")
        plt.plot(torro_data[var][0], torro_data[var][1], label="Torro numerical", marker='o', markersize=2, linestyle='')
        plt.plot(torro_data_exact_solution[var][0], torro_data_exact_solution[var][1], label="Torro exact", linestyle='--')
        plt.xlabel("x")
        plt.ylabel(f"{var}")
        plt.legend()

    #plt.show()
    plt.savefig(f"case_{case}_{solver}.png")

# Plot results for all solvers
def plot_results_per_case_all_solvers(x, all_data_dic, case, variables):
    plt.figure(figsize=(14, 12))
    plt.title(f"Case {case} - Numerical Solver Comparison")
    plt.axis("off")

    for i, var in enumerate(variables):
        plt.subplot(2, 2, i+1)
        
        # Plot all solvers
        for j, solver in enumerate(all_data_dic[case]):
            if solver != "Exact Riemann solution":
                plt.plot(x, all_data_dic[case][solver][var], label=f"{solver} solver", marker='o', markersize=3 - 0.5*j, linestyle='')
            else:
                plt.plot(x, all_data_dic[case][solver][var], label="Exact Riemann solution")
        
        plt.xlabel("x")
        plt.ylabel(f"{var}")
        plt.legend()

    #plt.show()
    plt.savefig(f"case_{case}_solver_comparison.png")
    
################## Run the simulation ##################

cases=[1,4]
variables = ["density", "velocity", "pressure", "energy"]
solvers=["Exact", "HLLC", "Roe-Pike"]
all_data_dic={}

for case in cases:
    all_data_dic[case] = {}
    rho_L, p_L, u_L, rho_R, p_R, u_R, t_max, x0 = inital_conditions(case)
    print(f"intialized case {case}")
    x, U_exact = solve_exact(rho_L, p_L, u_L, rho_R, p_R, u_R, t_max, x0)
    all_data_dic[case]["Exact Riemann solution"] = calculate_e_and_set_up_data_dictionary( U_exact)
    print("computed exact solution")
    torro_Exact_Riemann_solution=Extract_torro_data(case, variables,"Exact Riemann solution")
    print("extracted torro exact solution")

    for solver in solvers:
        x_half, U_numerical = Gondunov_method(case,solver) 
        print(f"computed numerical solution using {solver} solver")
        all_data_dic[case][solver] = calculate_e_and_set_up_data_dictionary(U_numerical)
        plot_results_per_case_and_solver(x, x_half, torro_Exact_Riemann_solution, all_data_dic, case, variables, solver)
        print(f"plotted case {case} with {solver} solver")

for case in cases:
    plot_results_per_case_all_solvers(x, all_data_dic, case, variables)
    print(f"plotted case {case} with all solvers")

