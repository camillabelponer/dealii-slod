#ifndef dealii_lod_tools_h
#define dealii_lod_tools_h

// #include <deal.II/base/timer.h>
//
// #include <deal.II/lac/dynamic_sparsity_pattern.h>
// #include <deal.II/lac/sparsity_tools.h>
// #include <deal.II/lac/trilinos_sparse_matrix.h>
//
// #include <deal.II/matrix_free/fe_evaluation.h>
// #include <deal.II/matrix_free/matrix_free.h>
// #include <deal.II/matrix_free/operators.h>
// #include <deal.II/matrix_free/tools.h>
//
// #include <deal.II/multigrid/mg_tools.h>
//
// #include <deal.II/grid/grid_generator.h>

using namespace dealii;


template <int dim>
void
projection_P0_P1(FullMatrix<double> &projection_matrix)
{
  unsigned int n_fine_dofs = projection_matrix.n();
  unsigned int p           = (int)sqrt(n_fine_dofs);
  Assert(p * p == n_fine_dofs, ExcNotImplemented());
  Assert(projection_matrix.n() != 0, ExcNotImplemented());
  Assert(dim == 2, ExcNotImplemented());

  unsigned int col_index = 0;
  while (col_index < 2 * dim)
    {
      projection_matrix(0, col_index) = 1.0;
      col_index++;
    }
  while (col_index < (2 * dim * (p - 2) + 2 * dim))
    {
      projection_matrix(0, col_index) = 2.0;
      col_index++;
    }
  while (col_index < projection_matrix.n())
    {
      projection_matrix(0, col_index) = 4.0;
      col_index++;
    }
}


template <int dim>
const Table<2, bool>
create_bool_dof_mask(const FiniteElement<dim> &fe,
                     const Quadrature<dim> &   quadrature)
{
  const auto compute_scalar_bool_dof_mask = [&quadrature](const auto &fe) {
    Table<2, bool> bool_dof_mask(fe.dofs_per_cell, fe.dofs_per_cell);
    MappingQ1<dim> mapping;
    FEValues<dim>  fe_values(mapping, fe, quadrature, update_values);

    Triangulation<dim> tria;
    GridGenerator::hyper_cube(tria);

    fe_values.reinit(tria.begin());
    for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
      for (unsigned int j = 0; j < fe.dofs_per_cell; ++j)
        {
          double sum = 0;
          for (unsigned int q = 0; q < quadrature.size(); ++q)
            sum += fe_values.shape_value(i, q) * fe_values.shape_value(j, q);
          if (sum != 0)
            bool_dof_mask(i, j) = true;
        }

    return bool_dof_mask;
  };

  Table<2, bool> bool_dof_mask(fe.dofs_per_cell, fe.dofs_per_cell);

  if (fe.n_components() == 1)
    {
      bool_dof_mask = compute_scalar_bool_dof_mask(fe);
    }
  else
    {
      const auto scalar_bool_dof_mask =
        compute_scalar_bool_dof_mask(fe.base_element(0));

      for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
        for (unsigned int j = 0; j < fe.n_dofs_per_cell(); ++j)
          if (scalar_bool_dof_mask[fe.system_to_component_index(i).second]
                                  [fe.system_to_component_index(j).second])
            bool_dof_mask[i][j] = true;
    }


  return bool_dof_mask;
};

template <int dim>
const Table<2, bool>
create_bool_dof_mask_Q_iso_Q1(const FiniteElement<dim> &fe,
                              const Quadrature<dim> &   quadrature,
                              unsigned int              n_subdivisions)
{
  const auto compute_scalar_bool_dof_mask =
    [&quadrature](const auto &fe, const auto n_subdivisions) {
      Table<2, bool> bool_dof_mask(fe.dofs_per_cell, fe.dofs_per_cell);
      MappingQ1<dim> mapping;
      FEValues<dim>  fe_values(mapping, fe, quadrature, update_values | update_gradients);

      Triangulation<dim> tria;
      GridGenerator::hyper_cube(tria);

      fe_values.reinit(tria.begin());

      const auto lexicographic_to_hierarchic_numbering =
        FETools::lexicographic_to_hierarchic_numbering<dim>(n_subdivisions);

      for (unsigned int c_1 = 0; c_1 < n_subdivisions; ++c_1)
        for (unsigned int c_0 = 0; c_0 < n_subdivisions; ++c_0)

          for (unsigned int i_1 = 0; i_1 < 2; ++i_1)
            for (unsigned int i_0 = 0; i_0 < 2; ++i_0)
              {
                const unsigned int i =
                  lexicographic_to_hierarchic_numbering[(c_0 + i_0) +
                                                        (c_1 + i_1) *
                                                          (n_subdivisions + 1)];

                for (unsigned int j_1 = 0; j_1 < 2; ++j_1)
                  for (unsigned int j_0 = 0; j_0 < 2; ++j_0)
                    {
                      const unsigned int j =
                        lexicographic_to_hierarchic_numbering
                          [(c_0 + j_0) + (c_1 + j_1) * (n_subdivisions + 1)];

                      double sum = 0;

                      for (unsigned int q_1 = 0; q_1 < 2; ++q_1)
                        for (unsigned int q_0 = 0; q_0 < 2; ++q_0)
                          {
                            const unsigned int q_index =
                              (c_0 * 2 + q_0) +
                              (c_1 * 2 + q_1) * (2 * n_subdivisions);

                            sum += fe_values.shape_grad(i, q_index) *
                                   fe_values.shape_grad(j, q_index);
                          }
                      if (sum != 0)
                        bool_dof_mask(i, j) = true;
                    }
              }

      return bool_dof_mask;
    };

  Table<2, bool> bool_dof_mask(fe.dofs_per_cell, fe.dofs_per_cell);

  if (fe.n_components() == 1)
    {
      bool_dof_mask = compute_scalar_bool_dof_mask(fe, n_subdivisions);
    }
  else
    {
      const auto scalar_bool_dof_mask =
        compute_scalar_bool_dof_mask(fe.base_element(0), n_subdivisions);

      for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
        for (unsigned int j = 0; j < fe.n_dofs_per_cell(); ++j)
          if (scalar_bool_dof_mask[fe.system_to_component_index(i).second]
                                  [fe.system_to_component_index(j).second])
            bool_dof_mask[i][j] = true;
    }


  return bool_dof_mask;
};

template <int dim>
std::vector<std::vector<unsigned int>>
create_quadrature_dofs_map(const FiniteElement<dim> &fe,
                           const Quadrature<dim> &   quadrature)
{
  std::vector<std::vector<unsigned int>> map;

  MappingQ1<dim> mapping;
  FEValues<dim>  fe_values(mapping,
                          fe,
                          quadrature,
                          update_values | update_gradients);

  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria);

  fe_values.reinit(tria.begin());
  for (unsigned int q = 0; q < quadrature.size(); ++q)
    {
      std::vector<unsigned int> dofs_of_q;
      for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
        {
          if (fe_values.shape_grad(i, q).norm() != 0)
            dofs_of_q.push_back(i);
        }
      map.push_back(dofs_of_q);
    }

  return map;
};

template <int dim>
void
extend_vector_to_boundary_values(const Vector<double> & vector_in,
                                 const DoFHandler<dim> &dh,
                                 Vector<double> &       vector_out)
{
  Assert(dh.n_dofs() == vector_out.size(), ExcNotImplemented());

  IndexSet     boundary_dofs_set = DoFTools::extract_boundary_dofs(dh);
  unsigned int N_internal_dofs   = dh.n_dofs() - boundary_dofs_set.n_elements();

  AssertDimension(N_internal_dofs, vector_in.size());//, ExcNotImplemented());
  Assert(N_internal_dofs < dh.n_dofs(), ExcNotImplemented());

  unsigned int in_index = 0;
  for (unsigned int out_index = 0; out_index < vector_out.size(); ++out_index)
    {
      if (!boundary_dofs_set.is_element(out_index))
        {
          vector_out[out_index] = vector_in[in_index];
          in_index++;
        }
      else
        vector_out[out_index] = 0.0;
    }
}

template <int dim>
void
fill_dofs_indices_vector(const DoFHandler<dim> & dh,
                         std::vector<unsigned int> & all_dofs,
                         std::vector<unsigned int> & internal_dofs,
                         std::vector<unsigned int> & boundary_dofs)
{
  auto boundary_indices(dh.get_triangulation().get_boundary_ids());
  unsigned int N_boundary_indices = boundary_indices.size();
  Assert(N_boundary_indices < 3, ExcNotImplemented());


  IndexSet all(dh.n_dofs());
  all.add_range(0, dh.n_dofs());
  IndexSet internal(all);
  
  IndexSet boundary_of_domain_and_patch_set;
  IndexSet boundary_of_patch_not_of_domain_set;

  boundary_of_domain_and_patch_set = DoFTools::extract_boundary_dofs(dh, ComponentMask(), std::set<unsigned int>{0});

  boundary_of_patch_not_of_domain_set = DoFTools::extract_boundary_dofs(dh, ComponentMask(), std::set<unsigned int>{99});

  internal.subtract_set(boundary_of_patch_not_of_domain_set);
  internal.subtract_set(boundary_of_domain_and_patch_set);
  //boundary_of_patch_not_of_domain_set.subtract_set(boundary_of_domain_and_patch_set);
  
  //Assert((internal.n_elements() + boundary_of_patch_not_of_domain_set.n_elements() + boundary_of_domain_and_patch_set.n_elements()) == all.n_elements(), ExcNotImplemented());

  boundary_of_patch_not_of_domain_set.fill_index_vector(boundary_dofs);
  internal.fill_index_vector(internal_dofs);
  all.fill_index_vector(all_dofs);

  
}


namespace dealii::TrilinosWrappers
{
  class MySolverDirect : public SolverDirect
  {
  private:
    /**
     * Actually performs the operations for solving the linear system,
     * including the factorization and forward and backward substitution.
     */
    void
    do_solve()
    {
      // Fetch return value of Amesos Solver functions
      int ierr;

      // First set whether we want to print the solver information to screen or
      // not.
      // ConditionalOStream verbose_cout(std::cout,
      //                                additional_data.output_solver_details);

      // Next allocate the Amesos solver, this is done in two steps, first we
      // create a solver Factory and generate with that the concrete Amesos
      // solver, if possible.
      Amesos Factory;

      // AssertThrow(Factory.Query(additional_data.solver_type.c_str()),
      //             ExcMessage(
      //               std::string("You tried to select the solver type <") +
      //               additional_data.solver_type +
      //               "> but this solver is not supported by Trilinos either "
      //               "because it does not exist, or because Trilinos was not "
      //               "configured for its use."));

      solver.reset(
        Factory.Create(additional_data.solver_type.c_str(), *linear_problem));

      // verbose_
      // std::cout << "Starting symbolic factorization" << std::endl;
      ierr = solver->SymbolicFactorization();
      AssertThrow(ierr == 0, ExcTrilinosError(ierr));

      // verbose_
      // std::cout << "Starting numeric factorization" << std::endl;
      ierr = solver->NumericFactorization();
      AssertThrow(ierr == 0, ExcTrilinosError(ierr));

      // verbose_
      // std::cout << "Starting solve" << std::endl;
      ierr = solver->Solve();
      // std::cout << ierr << std::endl;
      AssertThrow(ierr == 0, ExcTrilinosError(ierr));
      // std::cout << ierr << std::endl;
      // Finally, let the deal.II SolverControl object know what has
      // happened. If the solve succeeded, the status of the solver control will
      // turn into SolverControl::success.
      solver_control.check(0, 0);

      if (solver_control.last_check() != SolverControl::success)
        AssertThrow(false,
                    SolverControl::NoConvergence(solver_control.last_step(),
                                                 solver_control.last_value()));
    }

    /**
     * Local dummy solver control object.
     */
    SolverControl solver_control_own;

    /**
     * Reference to the object that controls convergence of the iterative
     * solver. In fact, for these Trilinos wrappers, Trilinos does so itself,
     * but we copy the data from this object before starting the solution
     * process, and copy the data back into it afterwards.
     */
    SolverControl &solver_control;

    /**
     * A structure that collects the Trilinos sparse matrix, the right hand
     * side vector and the solution vector, which is passed down to the
     * Trilinos solver.
     */
    std::unique_ptr<Epetra_LinearProblem> linear_problem;

    /**
     * A structure that contains the Trilinos solver and preconditioner
     * objects.
     */
    std::unique_ptr<Amesos_BaseSolver> solver;

    /**
     * Store a copy of the flags for this particular solver.
     */
    AdditionalData additional_data;

  public:
    /**
     * Constructor. Creates the solver without solver control object.
     */
    explicit MySolverDirect(const AdditionalData &data = AdditionalData());

    /**
     * Constructor. Takes the solver control object and creates the solver.
     */
    MySolverDirect(SolverControl &       cn,
                   const AdditionalData &data = AdditionalData())
      : SolverDirect(cn, data)
      , solver_control(cn)
      , additional_data(data.output_solver_details, data.solver_type)
      // , SolverDirect(cn, data)
      {};

    /**
     * Destructor.
     */
    virtual ~MySolverDirect() = default;

    void
    solve(const Epetra_Operator &   A,
          Epetra_MultiVector &      x,
          const Epetra_MultiVector &b)
    {
      linear_problem = std::make_unique<Epetra_LinearProblem>(
        const_cast<Epetra_Operator *>(&A),
        &x,
        const_cast<Epetra_MultiVector *>(&b));
      do_solve();
    }
  };

}; // namespace dealii::TrilinosWrappers



void
Gauss_elimination(const FullMatrix<double> &rhs,
                  const // TrilinosWrappers::SparseMatrix
                  SparseMatrix<double> 
                  &sparse_matrix_origin
                  ,
                  FullMatrix<double> &  solution,
                  double                reduce    = 1.e-2,
                  double                tolerance = 1.e-10,
                  double                iter      = 100)
{
  TrilinosWrappers::SparseMatrix sparse_matrix;
  sparse_matrix.reinit(sparse_matrix_origin);

  // create preconditioner
  TrilinosWrappers::PreconditionILU ilu;
  ilu.initialize(sparse_matrix);

  Assert(sparse_matrix.m() == sparse_matrix.n(), ExcInternalError());
  Assert(rhs.m() == sparse_matrix.m(), ExcInternalError());
  Assert(rhs.m() == solution.m(), ExcInternalError());
  Assert(rhs.n() == solution.n(), ExcInternalError());

  solution = 0.0;

  const unsigned int n_dofs       = rhs.m();
  const unsigned int Ndofs_coarse = rhs.n();

  const unsigned int n_blocks        = Ndofs_coarse;
  const unsigned int n_blocks_stride = n_blocks;


  for (unsigned int b = 0; b < n_blocks; b += n_blocks_stride)
    {
      const unsigned int bend = std::min(n_blocks, b + n_blocks_stride);

      std::vector<double> rhs_temp(n_dofs * (bend - b));
      std::vector<double> solution_temp(n_dofs * (bend - b));

      for (unsigned int i = 0; i < (bend - b); ++i)
        for (unsigned int j = 0; j < n_dofs; ++j)
          {
            rhs_temp[i * n_dofs + j] = rhs(j, i + b); // rhs[i + b][j];
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

      ReductionControl solver_control(iter, tolerance, reduce, false, false);

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
        for (unsigned int j = 0; j < n_dofs; ++j)
          {
            solution(j, i + b) = solution_temp[i * n_dofs + j];
          }
    }
}

void
Gauss_elimination_vector_vector(
  const FullMatrix<double> &            matrix_rhs,
  const TrilinosWrappers::SparseMatrix &sparse_matrix,
  FullMatrix<double> &                  matrix_solution,
  double                                reduce    = 1.e-2,
  double                                tolerance = 1.e-10,
  double                                iter      = 100)
{
  // create preconditioner
  TrilinosWrappers::PreconditionILU ilu;
  ilu.initialize(sparse_matrix);

  Assert(sparse_matrix.m() == sparse_matrix.n(), ExcInternalError());
  Assert(matrix_rhs.m() == sparse_matrix.m(), ExcInternalError());
  Assert(matrix_rhs.m() == matrix_solution.m(), ExcInternalError());
  Assert(matrix_rhs.n() == matrix_solution.n(), ExcInternalError());

  const unsigned int n_dofs       = matrix_rhs.m();
  const unsigned int Ndofs_coarse = matrix_rhs.n();

  std::vector<Vector<double>> rhs(Ndofs_coarse);
  std::vector<Vector<double>> solution(Ndofs_coarse);

  for (unsigned int b = 0; b < Ndofs_coarse; ++b)
    {
      rhs[b].reinit(n_dofs);
      solution[b].reinit(n_dofs);
    }

  for (unsigned int i = 0; i < n_dofs; ++i)
    for (unsigned int j = 0; j < Ndofs_coarse; ++j)
      rhs[j][i] = matrix_rhs(i, j);


  const unsigned int n_blocks        = Ndofs_coarse;
  const unsigned int n_blocks_stride = n_blocks;



  for (unsigned int b = 0; b < n_blocks; b += n_blocks_stride)
    {
      const unsigned int bend = std::min(n_blocks, b + n_blocks_stride);

      std::vector<double> rhs_temp(n_dofs * (bend - b));
      std::vector<double> solution_temp(n_dofs * (bend - b));

      for (unsigned int i = 0; i < (bend - b); ++i)
        for (unsigned int j = 0; j < n_dofs; ++j)
          {
            rhs_temp[i * n_dofs + j]      = rhs[i + b][j];
            solution_temp[i * n_dofs + j] = solution[i + b][j];
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

      ReductionControl solver_control(iter, tolerance, reduce, false, false);

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
        for (unsigned int j = 0; j < n_dofs; ++j)
          {
            solution[i + b][j] = solution_temp[i * n_dofs + j];
          }
    }

  for (unsigned int i = 0; i < n_dofs; ++i)
    for (unsigned int j = 0; j < Ndofs_coarse; ++j)
      matrix_solution(i, j) = solution[j][i];
}


#endif