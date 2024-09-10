
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_q_iso_q1.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include "util.h"

using namespace dealii;

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  // setup system
  const unsigned int dim            = 2;
  const unsigned int fe_degree      = 2;
  const unsigned int n_refinements  = 1;
  const unsigned int n_subdivisions = 1;

  MappingQ1<dim> mapping;
  // FE_Q<dim>      fe(fe_degree);
  FE_Q_iso_Q1<dim> fe(n_subdivisions);
  // QGauss<dim>    quadrature(fe_degree + 1);
  QIterated<dim> quadrature(QGauss<1>(2), n_subdivisions);

  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria);
  tria.refine_global(n_refinements);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  AffineConstraints<double> constraints;
  DoFTools::make_zero_boundary_constraints(dof_handler, constraints);
  constraints.close();

  const unsigned int n_dofs = dof_handler.n_dofs();

  // create matrix
  TrilinosWrappers::SparsityPattern sparsity_pattern(n_dofs, n_dofs);
  DoFTools::make_sparsity_pattern(dof_handler, sparsity_pattern, constraints);
  sparsity_pattern.compress();

  TrilinosWrappers::SparseMatrix sparse_matrix(sparsity_pattern);

  MatrixCreator::create_laplace_matrix<dim, dim>(
    mapping, dof_handler, quadrature, sparse_matrix, nullptr, constraints);

  // create preconditioner
  TrilinosWrappers::PreconditionILU ilu;
  ilu.initialize(sparse_matrix);


  // ndofs = Ndofs_fine
  const unsigned int Ndofs_coarse = 4;

  const unsigned int n_blocks        = Ndofs_coarse;
  const unsigned int n_blocks_stride = n_blocks;

  FullMatrix<double> rhs(n_dofs, Ndofs_coarse);
  FullMatrix<double> solution(n_dofs, Ndofs_coarse);
  FullMatrix<double> sol_via_inversion(n_dofs, Ndofs_coarse);

  for (unsigned int b = 0; b < Ndofs_coarse; ++b)
    rhs(b, b) = 1.0;

  for (unsigned int b = 0; b < Ndofs_coarse; ++b)
    {
      Vector<double> P_e_i(n_dofs);
      P_e_i[b] = 1;
      Vector<double>                         u_i(n_dofs);
      ReductionControl                       sc;
      dealii::TrilinosWrappers::SolverDirect sd(sc);
      sd.solve(sparse_matrix, u_i, P_e_i);
      for (unsigned int i = 0; i < n_dofs; ++i)
        sol_via_inversion(i, b) = u_i[i];
    }

  for (unsigned int b = 0; b < n_blocks; b += n_blocks_stride)
    {
      const unsigned int bend = std::min(n_blocks, b + n_blocks_stride);

      std::vector<double> rhs_temp(n_dofs * (bend - b));
      std::vector<double> solution_temp(n_dofs * (bend - b));

      for (unsigned int i = 0; i < (bend - b); ++i)
        for (unsigned int j = 0; j < Ndofs_coarse; ++j)
          {
            rhs_temp[i * n_dofs + j] = rhs(i + b, j); // rhs[i + b][j];
            solution_temp[i * n_dofs + j] =
              0.0; // solution(i+b,j); //solution[i + b][j];
          }

      std::vector<double *> rhs_ptrs(bend - b);
      std::vector<double *> sultion_ptrs(bend - b);

      for (unsigned int i = 0; i < (bend - b); ++i)
        {
          rhs_ptrs[i]     = &rhs_temp[i * n_dofs];      //&rhs[i + b][0];
          sultion_ptrs[i] = &solution_temp[i * n_dofs]; //&solution[i + b][0];
        }

      const Epetra_CrsMatrix &mat  = sparse_matrix.trilinos_matrix();
      const Epetra_Operator & prec = ilu.trilinos_operator();

      Epetra_MultiVector trilinos_dst(View,
                                      mat.OperatorRangeMap(),
                                      sultion_ptrs.data(),
                                      sultion_ptrs.size());
      Epetra_MultiVector trilinos_src(View,
                                      mat.OperatorDomainMap(),
                                      rhs_ptrs.data(),
                                      rhs_ptrs.size());

      ReductionControl solver_control;

      if (false)
        {
          TrilinosWrappers::SolverCG solver(solver_control);
          solver.solve(mat, trilinos_dst, trilinos_src, prec);
        }
      else
        {
          TrilinosWrappers::MySolverDirect solver(solver_control);
          solver.initialize(sparse_matrix);
          solver.solve(mat, trilinos_dst, trilinos_src);
        }

      for (unsigned int i = 0; i < (bend - b); ++i)
        for (unsigned int j = 0; j < Ndofs_coarse; ++j)
          {
            solution(i + b, j) = solution_temp[i * n_dofs + j];
          }
    }

  double error = 0;
  for (unsigned int i = 0; i < n_dofs; ++i)
    for (unsigned int j = 0; j < Ndofs_coarse; ++j)
      {
        error =
          std::max(error, std::abs(solution(i, j) - sol_via_inversion(i, j)));
      }
  std::cout << std::max(error, 1e-2) << std::endl;
}
