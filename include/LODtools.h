#ifndef dealii_lod_tools_h
#  define dealii_lod_tools_h

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
    //ConditionalOStream verbose_cout(std::cout,
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
    std::cout << "Starting symbolic factorization" << std::endl;
    ierr = solver->SymbolicFactorization();
    AssertThrow(ierr == 0, ExcTrilinosError(ierr));

    // verbose_
    std::cout << "Starting numeric factorization" << std::endl;
    ierr = solver->NumericFactorization();
    AssertThrow(ierr == 0, ExcTrilinosError(ierr));

    // verbose_
    std::cout << "Starting solve" << std::endl;
    ierr = solver->Solve();
    std::cout << ierr << std::endl;
    AssertThrow(ierr == 0, ExcTrilinosError(ierr));
std::cout << ierr << std::endl;
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
    MySolverDirect(SolverControl        &cn,
                 const AdditionalData &data = AdditionalData()
                 )
    : additional_data(data.output_solver_details, data.solver_type)
    , solver_control(cn)
    , SolverDirect(cn, data)
    {};

    /**
     * Destructor.
     */
    virtual ~MySolverDirect() = default;

    void
    solve(const Epetra_Operator &A, Epetra_MultiVector &x, const Epetra_MultiVector &b)
    {
      linear_problem = std::make_unique<Epetra_LinearProblem>(const_cast<Epetra_Operator *>(&A), &x, const_cast<Epetra_MultiVector *>(&b));
      do_solve();
    }


  };
};

#endif