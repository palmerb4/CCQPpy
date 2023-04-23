# external
import numpy as np

# internal
from ccqppy import solution_spaces as ss

if __name__ == '__main__':
    dims = np.arange(3) + 1
    proj_ops_to_benchmark = []
    proj_ops_to_benchmark.append([ss.IdentityProjOp(dim)
                                 for dim in dims])
    proj_ops_to_benchmark.append([ss.LowerBoundProjOp(dim)
                                  for dim in dims])
    proj_ops_to_benchmark.append([ss.UpperBoundProjOp(dim)
                                  for dim in dims])
    proj_ops_to_benchmark.append([ss.BoxProjOp(dim)
                                  for dim in dims])
    proj_ops_to_benchmark.append([ss.SphereProjOp(dim)
                                 for dim in dims])
    proj_ops_to_benchmark.append([ss.ConeProjOp(dim)
                                  for dim in dims])
    for proj_ops in proj_ops_to_benchmark:
        for proj_op in proj_ops:
            dim = proj_op.embedded_dimension
            proj_op.plot(100, -2 * np.ones(dim), 2 * np.ones(dim))