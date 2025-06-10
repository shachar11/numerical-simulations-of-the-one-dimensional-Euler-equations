# Riemann Problem CFD Solver

This repository contains a Python implementation of several numerical solvers for the 1D Riemann problem in computational fluid dynamics (CFD). The code was developed as a final project for a university course in CFD.

## Project Overview

The script solves the Riemann problem for gas dynamics using three different methods:
- **Exact Riemann Solver**
- **HLLC Solver**
- **Roe-Pike Solver**

It compares the results of these solvers with reference data (Torro data) and visualizes the results for different initial conditions (cases).

## Features

- Modular code structure with clear separation of physics, numerical methods, and plotting.
- Supports multiple initial conditions (cases).
- Compares numerical solutions to exact and reference (Torro) solutions.
- Generates plots for density, velocity, pressure, and internal energy.
- Saves plots for each solver and case.

## Usage

1. **Run the Code:**  
- Execute the script: python3 final_project.py
  The script will run all cases and solvers, print progress to the terminal, and save plots as PNG files.
  
2. **Reference Data:**
- The Toro reference CSV files are placed in the expected folders:
- torro_exact_solution/
- torro_HLLC_solver/
- torro_Roe-Pike_solver/

3. **File Structure:**
   final_project.py — Main script containing all solvers, data extraction, and plotting routines.
   torro_exact_solution/ — Folder for Torro reference CSV files for the exact solution.
   torro_HLLC_solver/, torro_Roe-Pike_solver/ — Folders for Torro reference CSV files for each solver.

Notes
Developed as a final project for a university CFD course.
Intended for educational and demonstration purposes.
Ensure all reference files are correctly named and placed.
For a detailed explanation, see the attached document:
Shachar_Charash_Final_Project

