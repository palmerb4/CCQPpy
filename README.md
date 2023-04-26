# <h1 align='center'> **CCQPpy**
CCQPpy is an open source Python library of algorithms to solve the convex constrained quadratic problem. It comes with a systematic comparison of various algorithms for solving convex constrained quadratic programming problems with convex feasible set. 

**It provides:**
- **Standardized implementations of modern algorithms**
    1. **PGD**
    2. **APGD**
    3. **BBPGD**
    4. **SPG**
    5. **MPRGP-BB**
- **A variety of convex sets**
    1. **Unconstrained**
    2. **Upper/lower bound**
    3. **Box**
    4. **Cone**
    5. **Disjoint Union**
- **Benchmarks**

## **Installiation:**
Can be installed with,
```bash
pip install ccqppy
```
or
```bash
git clone https://github.com/palmerb4/CCQPpy.git
```

## **Example Usage:**
```python
A = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
exact_x = np.array([1, 0, 1])
b = -A.dot(exact_x)
solution_space = BoxProjOp(3,np.array([-2,-2,-4]),np.array([2,2,5]))

desired_tol = 1e-10
max_mv_mults = 5000
solver = solvers.CCQPSolverSPG(desired_tol, max_mv_mults)
result = solver.solve(A, b, convex_proj_op=solution_space)

print("Solution:\t",result.solution)
print("Exact solution:\t",exact_x)
print("Is the solution correct?\t",np.all(np.isclose(
    result.solution, tpn.exact_solution)))
print("Converged?\t",result.solution_converged)
print("Solution time:\t",result.solution_time)
print("Residual:\t",result.solution_residual)
print("Number of matrix, vector multiplications:\t",result.solution_num_matrix_vector_multiplications)
```
```python output
solving SPG
Solution:	 [ 1.0000000e+00 -5.5187636e-11  1.0000000e+00]
Exact solution:	 [1 0 1]
Is the solution correct?	 True
Converged?	 True
Solution time:	 0.0059010982513427734
Residual:	 7.804688171132893e-11
Number of matrix, vector multiplications:	 86
```
