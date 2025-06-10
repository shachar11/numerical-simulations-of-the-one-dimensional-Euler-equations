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
- Place the Torro reference CSV files in the appropriate folders (torro_exact_solution/ and torro_HLLC_solver/, etc.) as expected by the script.

3. **File Structure:**
final_project.py — Main script containing all solvers, data extraction, and plotting routines.
torro_exact_solution/ — Folder for Torro reference CSV files for the exact solution.
torro_HLLC_solver/, torro_Roe-Pike_solver/ — Folders for Torro reference CSV files for each solver.
Notes
This code was developed as a final project for a university CFD course.
The implementation is intended for educational and demonstration purposes.
For best results, ensure all reference data files are present and named as expected.
For a through explantion read the attached documant - Shachar_Charash_Final_Project

