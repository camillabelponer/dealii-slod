#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parsed_convergence_table.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/types.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_q_iso_q1.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/intergrid_map.h>

#include <deal.II/lac/arpack_solver.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/petsc_full_matrix.h>
#include <deal.II/lac/petsc_matrix_free.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/slepc_solver.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_vector.h>

#include <deal.II/multigrid/mg_transfer_global_coarsening.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <memory>


namespace LA
{
  using namespace dealii::LinearAlgebraTrilinos;
} // namespace LA



using namespace dealii;




template <int dim, int spacedim>
void
projection_P1_P0(FullMatrix<double> &projection_matrix)
{
  Assert(dim == 2,
         ExcNotImplemented(
           "Projection P0 to P1 only implemented for 2D problems"));

  unsigned int n_fine_dofs = projection_matrix.m();
  unsigned int p           = (int)sqrt(n_fine_dofs / spacedim);
  Assert(p * p * spacedim == n_fine_dofs,
         ExcNotImplemented("casting error")); // check the root to avoid casting
  Assert(projection_matrix.m() != 0, ExcNotImplemented("empty matrix"));
  Assert(projection_matrix.n() == spacedim,
         ExcNotImplemented(
           "only projection to P0 allowed")); // otherwise it's not P0

  if constexpr (spacedim == 1)
    {
      unsigned int row_index = 0;
      while (row_index < 2 * dim)
        {
          projection_matrix(row_index, 0) = 1.0;
          row_index++;
        }
      while (row_index < (2 * dim * (p - 2) + 2 * dim))
        {
          projection_matrix(row_index, 0) = 2.0;
          row_index++;
        }
      while (row_index < projection_matrix.m())
        {
          projection_matrix(row_index, 0) = 4.0;
          row_index++;
        }
    }
  else if constexpr (spacedim == 2)
    {
      unsigned int row_index = 0;
      while (row_index < 2 * dim * spacedim)
        {
          projection_matrix(row_index, 0) = 1.0;
          row_index++;
          projection_matrix(row_index, 1) = 1.0;
          row_index++;
        }
      while (row_index < (2 * dim * (p - 2) * spacedim + 2 * dim * spacedim))
        {
          projection_matrix(row_index, 0) = 2.0;
          row_index++;
          projection_matrix(row_index, 1) = 2.0;
          row_index++;
        }
      while (row_index < projection_matrix.m())
        {
          projection_matrix(row_index, 0) = 4.0;
          row_index++;
          projection_matrix(row_index, 1) = 4.0;
          row_index++;
        }
    }
  else
    AssertThrow(
      false,
      ExcNotImplemented(
        "projection matrix P0 to P1 not implemented for spacedim > 2"));
}


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
      FEValues<dim>  fe_values(mapping,
                              fe,
                              quadrature,
                              update_values | update_gradients);

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
void
extend_vector_to_boundary_values(Vector<double> &       vector_in,
                                 const DoFHandler<dim> &dh,
                                 Vector<double> &       vector_out)
{
  Assert(dh.n_dofs() == vector_out.size(),
         ExcNotImplemented("incoherent vector size"));

  IndexSet     boundary_dofs_set = DoFTools::extract_boundary_dofs(dh);
  unsigned int N_internal_dofs   = dh.n_dofs() - boundary_dofs_set.n_elements();

  AssertDimension(N_internal_dofs, vector_in.size()); //, ExcNotImplemented());
  Assert(N_internal_dofs < dh.n_dofs(),
         ExcNotImplemented("incoherent vector size"));

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
fill_dofs_indices_vector(const DoFHandler<dim> &    dh,
                         std::vector<unsigned int> &all_dofs,
                         std::vector<unsigned int> &internal_dofs,
                         std::vector<unsigned int> &boundary_dofs,
                         std::vector<unsigned int> &domain_boundary_dofs)
{
  auto         boundary_indices(dh.get_triangulation().get_boundary_ids());
  unsigned int N_boundary_indices = boundary_indices.size();
  Assert(N_boundary_indices < 3,
         ExcNotImplemented("too many doundary ids specified"));


  IndexSet all(dh.n_dofs());
  all.add_range(0, dh.n_dofs());
  IndexSet internal(all);

  IndexSet boundary_of_domain_and_patch_set;
  IndexSet boundary_of_patch_not_of_domain_set;

  boundary_of_domain_and_patch_set =
    DoFTools::extract_boundary_dofs(dh,
                                    ComponentMask(),
                                    std::set<unsigned int>{0});

  boundary_of_patch_not_of_domain_set =
    DoFTools::extract_boundary_dofs(dh,
                                    ComponentMask(),
                                    std::set<unsigned int>{99});

  internal.subtract_set(boundary_of_patch_not_of_domain_set);
  internal.subtract_set(boundary_of_domain_and_patch_set);
  // we DO NOT subtract boundary_of_domain_and_patch_set from
  // boundary_of_patch_not_of_domain_set
  // boundary_of_patch_not_of_domain_set.subtract_set(boundary_of_domain_and_patch_set);

  boundary_of_patch_not_of_domain_set.fill_index_vector(boundary_dofs);
  boundary_of_domain_and_patch_set.fill_index_vector(domain_boundary_dofs);
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
Gauss_elimination(const FullMatrix<double> &            rhs,
                  const TrilinosWrappers::SparseMatrix &sparse_matrix,
                  FullMatrix<double> &                  solution,
                  double                                reduce    = 1.e-16,
                  double                                tolerance = 1.e-18,
                  double                                iter      = 100)
{
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


      if (false)
        {
          ReductionControl solver_control(
            iter, tolerance, reduce, false, false);
          TrilinosWrappers::SolverCG solver(solver_control);
          solver.solve(mat, trilinos_dst, trilinos_src, prec);
        }
      else
        {
          SolverControl solver_control(iter, tolerance, false, false);
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


///////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <int dim>
class Patch
{
public:
  // TODO
  // make everything private and write getter and setter functions

  // coarse cells that make up the patch
  std::vector<typename DoFHandler<dim>::active_cell_iterator> cells;
  Triangulation<dim>                                          sub_tria;

  std::vector<Vector<double>> basis_function;
  std::vector<Vector<double>> basis_function_premultiplied;
  unsigned int                contained_patches = 0;
};

template <int dim>
class piecewiseconstantFunction : public Function<dim, double>
{
public:
  piecewiseconstantFunction(){};

  double
  value(const Point<dim> &p, const unsigned int) const override
  {
    return 1.0;
  }
};


template <int dim, int spacedim>
class LOD
{
public:
  LOD();

  unsigned int oversampling         = 1;
  unsigned int n_subdivisions       = 2;
  unsigned int n_global_refinements = 2;
  bool constant_coefficients = true;

  // rhs used to read from input file can also be defined with the class above
  piecewiseconstantFunction<dim> forcing;
  

  virtual void
  run();

protected:
  void
  make_fe();
  void
  make_grid();
  void
  create_patches();
  void
  compute_basis_function_candidates();
  void
  assemble_global_matrix();
  void
  solve();
  void
  assemble_and_solve_fem_problem();
  void
  compare_lod_with_fem();
  void
  initialize_patches();

  MPI_Comm                            mpi_communicator;
  ConditionalOStream                  pcout;

  void
  create_mesh_for_patch(Patch<dim> &current_patch);
  virtual void
  assemble_stiffness(LA::MPI::SparseMatrix &,
                     LA::MPI::Vector &,
                     const DoFHandler<dim> &,
                     AffineConstraints<double> &){
    // TODO: assert that lod is never called
  };

  parallel::shared::Triangulation<dim> tria;
  // check ghost layer, needs to be set to whole domain
  // shared not distributed bc we want all processors to get access to all cells
  DoFHandler<dim> dof_handler_coarse;
  DoFHandler<dim> dof_handler_fine;

  AffineConstraints<double> coarse_boundary_constraints;

  LA::MPI::SparseMatrix basis_matrix;
  LA::MPI::SparseMatrix basis_matrix_transposed;
  LA::MPI::SparseMatrix premultiplied_basis_matrix;
  LA::MPI::SparseMatrix global_stiffness_matrix;
  LA::MPI::Vector       solution;
  LA::MPI::Vector       solution_fine;
  LA::MPI::Vector       system_rhs;
  LA::MPI::Vector       fem_rhs;
  LA::MPI::Vector       fem_solution;
  LA::MPI::Vector       fem_coarse_solution_interpolated;
  LA::MPI::SparseMatrix presaved_patch_stiffness_matrix;
  LA::MPI::SparseMatrix presaved_constrained_patch_stiffness_matrix;

  std::unique_ptr<FiniteElement<dim>> fe_coarse;
  std::unique_ptr<FiniteElement<dim>> fe_fine;
  std::unique_ptr<Quadrature<dim>>    quadrature_fine;


  std::vector<Patch<dim>> patches;
  DynamicSparsityPattern  patches_pattern;
  DynamicSparsityPattern  patches_pattern_fine;

  IndexSet locally_owned_patches;
  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  Table<2, bool> bool_dof_mask;

  DataOut<dim> data_out;
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation;
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////


const unsigned int SPECIAL_NUMBER = 99;
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


template <int dim, int spacedim>
LOD<dim, spacedim>::LOD()
  : mpi_communicator(MPI_COMM_WORLD)
  , pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
  , tria(mpi_communicator)
  , dof_handler_coarse(tria)
  , dof_handler_fine(tria)
{
  if constexpr (spacedim == 1)
    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);
  else
    data_component_interpretation =
      std::vector<DataComponentInterpretation::DataComponentInterpretation>(
        spacedim, DataComponentInterpretation::component_is_part_of_vector);
}

template <int dim, int spacedim>
void
LOD<dim, spacedim>::make_fe()
{
  fe_coarse = std::make_unique<FESystem<dim>>(FE_DGQ<dim>(0), spacedim);
  dof_handler_coarse.distribute_dofs(*fe_coarse);

  locally_owned_dofs = dof_handler_coarse.locally_owned_dofs();
  locally_relevant_dofs =
    DoFTools::extract_locally_relevant_dofs(dof_handler_coarse);

  // constraints on the boundary of the domain
  coarse_boundary_constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler_coarse,
                                          coarse_boundary_constraints);
  VectorTools::interpolate_boundary_values(dof_handler_coarse,
                                           0,
                                           Functions::ZeroFunction<dim>(),
                                           coarse_boundary_constraints);
  coarse_boundary_constraints.close();

  fe_fine =
    std::make_unique<FESystem<dim>>(FE_Q_iso_Q1<dim>(n_subdivisions),
                                    spacedim);
  dof_handler_fine.distribute_dofs(*fe_fine);
  quadrature_fine = std::make_unique<Quadrature<dim>>(
    QIterated<dim>(QGauss<1>(2), n_subdivisions));

  patches_pattern.reinit(dof_handler_coarse.n_dofs(),
                         dof_handler_coarse.n_dofs(),
                         locally_relevant_dofs);
  patches_pattern_fine.reinit(dof_handler_coarse.n_dofs(),
                              dof_handler_fine.n_dofs(),
                              locally_relevant_dofs);

  bool_dof_mask = create_bool_dof_mask_Q_iso_Q1(*fe_fine,
                                                *quadrature_fine,
                                                n_subdivisions);
  // MPI: instead of having every processor compute it we could just communicate
  // it
}

template <int dim, int spacedim>
void
LOD<dim, spacedim>::make_grid()
{
  GridGenerator::hyper_cube(tria);
  tria.refine_global(n_global_refinements);

  locally_owned_patches =
    Utilities::MPI::create_evenly_distributed_partitioning(
      mpi_communicator, tria.n_global_active_cells());
}


template <int dim, int spacedim>
void
LOD<dim, spacedim>::create_patches()
{
  std::vector<unsigned int> fine_dofs(fe_fine->n_dofs_per_cell());
  std::vector<unsigned int> coarse_dofs(fe_coarse->n_dofs_per_cell());

  size_t size_biggest_patch = 0;
  size_t size_tiniest_patch = tria.n_active_cells();

  double       H                = pow(0.5, n_global_refinements);
  unsigned int N_cells_per_line = (int)1 / H;
  std::vector<typename DoFHandler<dim>::active_cell_iterator> ordered_cells;
  ordered_cells.resize(tria.n_active_cells());
  std::vector<std::vector<unsigned int>> cells_in_patch;
  cells_in_patch.resize(tria.n_active_cells());

  for (const auto &cell : dof_handler_coarse.active_cell_iterators())
    {
      const double x = cell->barycenter()(0);
      const double y = cell->barycenter()(1);

      // const unsigned int x_i = (int)floor(x/H);
      // const unsigned int y_i = (int)floor(y/H);
      const unsigned int vector_cell_index =
        (int)floor(x / H) + N_cells_per_line * (int)floor(y / H);
      ordered_cells[vector_cell_index] = cell;

      std::vector<unsigned int> connected_indices;
      connected_indices.push_back(
        vector_cell_index); // we need the central cell to be the first one,
                            // after that order is not relevant

      for (int l_row = -oversampling;
           l_row <= static_cast<int>(oversampling);
           ++l_row)
        {
          double x_j = x + l_row * H;
          if (x_j > 0 && x_j < 1) // domain borders
            {
              for (int l_col = -oversampling;
                   l_col <= static_cast<int>(oversampling);
                   ++l_col)
                {
                  const double y_j = y + l_col * H;
                  if (y_j > 0 && y_j < 1)
                    {
                      const unsigned int vector_cell_index_j =
                        (int)floor(x_j / H) +
                        N_cells_per_line * (int)floor(y_j / H);
                      if (vector_cell_index != vector_cell_index_j)
                        connected_indices.push_back(vector_cell_index_j);
                    }
                }
            }
        }

      cells_in_patch[vector_cell_index] = connected_indices;
    }

  // now looping and creating the patches
  for (const auto &cell : dof_handler_coarse.active_cell_iterators())
    { // 1d should not add just index but coarse dofs
      cell->get_dof_indices(coarse_dofs);
      const auto vector_cell_index =
        (int)floor(cell->barycenter()(0) / H) +
        N_cells_per_line * (int)floor(cell->barycenter()(1) / H);
      // auto cell_index = cell->active_cell_index();
      {
        auto patch = &patches.emplace_back();


        for (auto neighbour_ordered_index : cells_in_patch[vector_cell_index])
          {
            auto &cell_to_add = ordered_cells[neighbour_ordered_index];
            auto  cell_to_add_coarse_dofs = coarse_dofs;
            cell_to_add->get_dof_indices(cell_to_add_coarse_dofs);

            patch->cells.push_back(cell_to_add);
            // patches_pattern.add(cell_index,
            // cell_to_add->active_cell_index());
            for (unsigned int d = 0; d < spacedim; ++d)
              patches_pattern.add_row_entries(coarse_dofs[d], coarse_dofs);
            auto cell_fine =
              cell_to_add->as_dof_handler_iterator(dof_handler_fine);
            cell_fine->get_dof_indices(fine_dofs);
            for (unsigned int d = 0; d < spacedim; ++d)
              {
                patches_pattern_fine.add_row_entries(coarse_dofs[d], fine_dofs);
                // patches_pattern_fine.add_row_entries(cell_index, fine_dofs);
              }
          }

        size_biggest_patch = std::max(size_biggest_patch, patch->cells.size());
        size_tiniest_patch = std::min(size_tiniest_patch, patch->cells.size());
      }
    }


  DynamicSparsityPattern global_sparsity_pattern;
  global_sparsity_pattern.compute_mmult_pattern(patches_pattern,
                                                patches_pattern);
  // global_stiffness_matrix.reinit(locally_owned_patches,
  //                                global_sparsity_pattern,
  //                                mpi_communicator);
  // TODO: fpr MPI FIX THIS
  global_stiffness_matrix.reinit(global_sparsity_pattern);

  solution.reinit(locally_owned_dofs, mpi_communicator);


  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0 ||
      Utilities::MPI::n_mpi_processes(mpi_communicator) == 1)
    {
      pcout << "Number of coarse cell = " << tria.n_active_cells()
            << ", number of patches = " << patches.size()
            << " (locally owned: " << locally_owned_patches.n_elements()
            << ") \n"
            << "Patches size in (" << size_tiniest_patch << ", "
            << size_biggest_patch << ")" << std::endl;
    }
}




template <int dim, int spacedim>
void
LOD<dim, spacedim>::compute_basis_function_candidates()
{
  DoFHandler<dim> dh_coarse_patch;
  DoFHandler<dim> dh_fine_patch;

  using VectorType = Vector<double>;

  // objects to reinit in the patches loop
  TrilinosWrappers::SparseMatrix patch_stiffness_matrix;
  AffineConstraints<double>
    domain_boundary_constraints; // keeps track of the nodes on the boundary of
                                 // the domain
  AffineConstraints<double>
    patch_boundary_constraints; // keeps track of the nodes on the boundary of
                                // the patch
  AffineConstraints<double>
    empty_boundary_constraints; // empty, only used to assemble
  // TODO: we might consider defining patch_stiffness_matrix as a
  // SparseMatrix<double>, in this way we can assemble without using an affine
  // constrain object and we do not need to create the intermediary CPSM. we
  // still need a TrilinosWrappers::SparseMatrix for the gaussian elimination,
  // so both object will still be created anyway.
  empty_boundary_constraints.close();

  // we are assuming mesh to be created as hyper_cube l 83
  double H = pow(0.5, n_global_refinements);
  double h = H / (n_subdivisions);

  // create projection matrix from fine to coarse cell (DG)
  FullMatrix<double> projection_matrixT(fe_fine->n_dofs_per_cell(),
                                        fe_coarse->n_dofs_per_cell());
      projection_P1_P0<dim, spacedim>(projection_matrixT);
      projection_matrixT *= (h * h / 4);

  for (auto current_patch_id : locally_owned_patches)
    {

      AssertIndexRange(current_patch_id, patches.size());
      auto current_patch = &patches[current_patch_id];

      bool use_presaved = false;

      unsigned int patch_size = current_patch->cells.size();
      if (constant_coefficients && // criteria to be defined to reuse patch
                                       // matrix, could be that's based on tha
                                       // paraeters
          patch_size == pow((2 * oversampling + 1), dim) &&
          presaved_patch_stiffness_matrix.local_size() > 0)
        use_presaved = true;

      // create_mesh_for_patch(*current_patch);
      dh_fine_patch.reinit(current_patch->sub_tria);
      dh_fine_patch.distribute_dofs(*fe_fine);

      dh_coarse_patch.reinit(current_patch->sub_tria);
      dh_coarse_patch.distribute_dofs(*fe_coarse);

      auto N_dofs_coarse = dh_coarse_patch.n_dofs();
      auto N_dofs_fine   = dh_fine_patch.n_dofs();

      std::vector<unsigned int>            internal_dofs_fine;
      std::vector<unsigned int>            all_dofs_fine;
      std::vector<unsigned int> /*patch_*/ boundary_dofs_fine;
      std::vector<unsigned int>            domain_boundary_dofs_fine;

      fill_dofs_indices_vector(dh_fine_patch,
                               all_dofs_fine,
                               internal_dofs_fine,
                               /*patch_*/ boundary_dofs_fine,
                               domain_boundary_dofs_fine);

      std::vector<unsigned int> all_dofs_coarse(all_dofs_fine.begin(),
                                                all_dofs_fine.begin() +
                                                  N_dofs_coarse);
      const unsigned int N_internal_dofs       = internal_dofs_fine.size();

      domain_boundary_constraints.clear();
      patch_boundary_constraints.clear();
      DoFTools::make_zero_boundary_constraints(dh_fine_patch,
                                               0,
                                               domain_boundary_constraints);
      DoFTools::make_zero_boundary_constraints(dh_fine_patch,
                                               SPECIAL_NUMBER,
                                               patch_boundary_constraints);
      domain_boundary_constraints.close();
      patch_boundary_constraints.close();

      SparsityPattern patch_sparsity_pattern;
      // if (!use_presaved)
      {
        DynamicSparsityPattern patch_dynamic_sparsity_pattern(N_dofs_fine);
        // does the same as
        // DoFTools::make_sparsity_pattern() but also

        std::vector<types::global_dof_index> dofs_on_this_cell(
          fe_fine->n_dofs_per_cell());

        for (const auto &cell : dh_fine_patch.active_cell_iterators())
          if (cell->is_locally_owned())
            {
              cell->get_dof_indices(dofs_on_this_cell);

              empty_boundary_constraints.add_entries_local_to_global(
                dofs_on_this_cell,
                patch_dynamic_sparsity_pattern,
                true,
                bool_dof_mask); // keep constrained entries must be true
            }

        patch_dynamic_sparsity_pattern.compress();

        patch_sparsity_pattern.copy_from(patch_dynamic_sparsity_pattern);
        patch_stiffness_matrix.clear();
        patch_stiffness_matrix.reinit(patch_sparsity_pattern);
      }

      if (use_presaved)
        {
          patch_stiffness_matrix.copy_from(presaved_patch_stiffness_matrix);
        }
      else
        {
          LA::MPI::Vector dummy;
          assemble_stiffness(patch_stiffness_matrix,
                             dummy,
                             dh_fine_patch,
                             empty_boundary_constraints);
          // using empty_boundary the stiffness is assembled unconstrained

          if (constant_coefficients &&
              patch_size == pow((2 * oversampling + 1), dim))
            {
              presaved_patch_stiffness_matrix.copy_from(patch_stiffness_matrix);
            }
        }

      // we now compute c_loc_i = S^-1 P^T (P_tilde S^-1 P^T)^-1 e_i
      // where e_i is the indicator function of the patch

      VectorType P_e_i(N_dofs_fine);
      VectorType e_i(N_dofs_coarse); // reused also as temporary vector
      VectorType triple_product_inv_e_i(N_dofs_coarse);
      VectorType Ac_i(N_internal_dofs);

      FullMatrix<double> PT(N_dofs_fine, N_dofs_coarse);
      FullMatrix<double> P_Ainv_PT(N_dofs_coarse);
      FullMatrix<double> Ainv_PT(N_dofs_fine, N_dofs_coarse);
      SparseMatrix<double> semi_constrained_patch_stiffness_matrix;

      // assign rhs
      {
        std::vector<types::global_dof_index> dofs_on_this_cell(
          fe_fine->n_dofs_per_cell());
        std::vector<unsigned int> coarse_dofs_on_this_cell(
          fe_coarse->n_dofs_per_cell()
          // it should be called coarse cells instead of coarse dofs
        );
        for (unsigned int d = 0; d < spacedim; d++)
          coarse_dofs_on_this_cell[d] = d;

        for (auto &cell : dh_fine_patch.active_cell_iterators())
          {
            cell->get_dof_indices(dofs_on_this_cell);

            empty_boundary_constraints.distribute_local_to_global(
              projection_matrixT,
              dofs_on_this_cell,
              coarse_dofs_on_this_cell,
              PT);
            // here we cannot use any other constraint than empty, or we would
            // lose the boundary values that are needed for PT_boundary

            for (unsigned int d = 0; d < spacedim; d++)
              coarse_dofs_on_this_cell[d] += spacedim;
          }
      }

      // we set the fod corresponding to boundary nods = 0 in PT because when
      // we apply the boundary conditions on unconstrained_stiffness we will
      // still have values on the diagonal of the constrained nodes setting
      // those values to zero would result in a gauss_elimination not converging
      for (unsigned int i = 0; i < N_dofs_coarse; ++i)
        {
          for (auto j : boundary_dofs_fine)
            PT(j, i) = 0.0;
          for (auto j : domain_boundary_dofs_fine)
            PT(j, i) = 0.0;
        }

      // if(!use_presaved)
      {
        for (auto j : domain_boundary_dofs_fine)
          patch_stiffness_matrix.clear_row(j, 1);
        semi_constrained_patch_stiffness_matrix.reinit(patch_sparsity_pattern);
        semi_constrained_patch_stiffness_matrix.copy_from(
          patch_stiffness_matrix);
        for (auto j : boundary_dofs_fine)
          patch_stiffness_matrix.clear_row(j, 1);
      }

      Gauss_elimination(PT, patch_stiffness_matrix, Ainv_PT);

      PT.Tmmult(P_Ainv_PT, Ainv_PT);

      // P_tilde is actually P/ H^dim
      P_Ainv_PT /= pow(H, dim);

      P_Ainv_PT.gauss_jordan();

      std::vector<VectorType> candidates;
      std::vector<VectorType> Palpha_i;
      VectorType              Pa_i(N_dofs_fine);
      Vector<double>          selected_basis_function(N_dofs_fine);
      Vector<double>          internal_selected_basis_function(N_internal_dofs);

{
          // if we are not stabilizing then we only take candidates related to
          // the central cell of the patch 0 is the index of the central cell
          // (this is also the central dof because we use P0 elements)
          for (unsigned int d = 0; d < spacedim; ++d)
            {
              e_i                     = 0.0;
              triple_product_inv_e_i  = 0.0;
              selected_basis_function = 0.0;

              e_i[d] = 1.0;
              P_Ainv_PT.vmult(triple_product_inv_e_i, e_i);

              if (false)
                {
                  // Ainv_PT_internal.vmult(internal_selected_basis_function,
                  //             triple_product_inv_e_i);
                  // // this works but Ainv_PT_internal is not defined yet
                }
              else
                {
                  Ainv_PT.vmult(selected_basis_function,
                                triple_product_inv_e_i);
                }

              selected_basis_function /= selected_basis_function.l2_norm();
              current_patch->basis_function.push_back(selected_basis_function);
            }
        }
      for (unsigned int d = 0; d < spacedim; ++d)
        {
          VectorType Ac_i_0(N_dofs_fine);

          semi_constrained_patch_stiffness_matrix.vmult(
            Ac_i_0, current_patch->basis_function[d]);
          current_patch->basis_function_premultiplied.push_back(Ac_i_0);
        }
      dh_fine_patch.clear();
    }
}

template <int dim, int spacedim>
void
LOD<dim, spacedim>::create_mesh_for_patch(Patch<dim> &current_patch)
{
  current_patch.sub_tria.clear();
  // copy manifolds
  // for (const auto i : tria.get_manifold_ids())
  //   if (i != numbers::flat_manifold_id)
  //     current_patch.sub_tria.set_manifold(i, tria.get_manifold(i));

  // re-enumerate vertices
  std::vector<unsigned int> new_vertex_indices(tria.n_vertices(), 0);

  for (const auto &cell : current_patch.cells)
    for (const unsigned int v : cell->vertex_indices())
      new_vertex_indices[cell->vertex_index(v)] = 1;

  for (unsigned int i = 0, c = 0; i < new_vertex_indices.size(); ++i)
    if (new_vertex_indices[i] == 0)
      new_vertex_indices[i] = numbers::invalid_unsigned_int;
    else
      new_vertex_indices[i] = c++;

  // collect points
  std::vector<Point<dim>> sub_points;
  for (unsigned int i = 0; i < new_vertex_indices.size(); ++i)
    if (new_vertex_indices[i] != numbers::invalid_unsigned_int)
      sub_points.emplace_back(tria.get_vertices()[i]);

  // create new cell and data
  std::vector<CellData<dim>> coarse_cells_of_patch;

  for (const auto &cell : current_patch.cells)
    {
      CellData<dim> new_cell(cell->n_vertices());

      for (const auto v : cell->vertex_indices())
        new_cell.vertices[v] = new_vertex_indices[cell->vertex_index(v)];

      // new_cell.material_id = cell->material_id();
      // new_cell.manifold_id = cell->manifold_id();

      coarse_cells_of_patch.emplace_back(new_cell);
    }

  // create coarse mesh on the patch
  current_patch.sub_tria.create_triangulation(sub_points,
                                              coarse_cells_of_patch,
                                              {});

  auto sub_cell = current_patch.sub_tria.begin(0);
  for (const auto &cell : current_patch.cells)
    {
      // TODO: Find better way to get patch id
      // global_to_local_cell_map[cell->active_cell_index()].push_back(
      //   std::pair<unsigned int,
      //             typename Triangulation<dim>::active_cell_iterator>(
      //     current_patch.cells[0]->active_cell_index(), sub_cell));
      // faces
      for (const auto f : cell->face_indices())
        {
          const auto face = cell->face(f);
          // if we are at boundary of patch AND domain -> keep boundary_id
          if (face->at_boundary())
            sub_cell->face(f)->set_boundary_id(face->boundary_id());
          // if the face is not at the boundary of the domain, is it at the
          // boundary of the patch?
          else if (sub_cell->face(f)->boundary_id() !=
                   numbers::internal_face_boundary_id)
            // it's not at te boundary of the patch -> then is our "internal
            // boundary"
            sub_cell->face(f)->set_boundary_id(SPECIAL_NUMBER);
        }


      // // lines // useless??
      // if constexpr (dim == 3)
      //   for (const auto l : cell->line_indices())
      //     {
      //       const auto line = cell->line(l);

      //       if (line->manifold_id() != numbers::flat_manifold_id)
      //         sub_cell->line(l)->set_manifold_id(line->manifold_id());
      //     }

      sub_cell++;
    }
}

template <int dim, int spacedim>
void
LOD<dim, spacedim>::assemble_global_matrix()
{

  DoFHandler<dim> dh_fine_current_patch;

  // auto     lod = dh_fine_current_patch.locally_owned_dofs();
  // TODO: for mpi should not allocate all cols and rows->create partitioning
  // like we do for global_stiffness_matrix.reinit(..)

  // basis_matrix.reinit(patches_pattern_fine.nonempty_rows(),
  //                     patches_pattern_fine.nonempty_cols(),
  //                     patches_pattern_fine,
  //                     mpi_communicator);

  // if we don't want to use the operator to compute the global_stiffness matrix
  // as a multiplication then we need the transpose of the patches_pattern_fine
  // and in this case the matrix premultiplied_basis_matrix will saved already
  // in the transposed form
  DynamicSparsityPattern identity(patches_pattern_fine.nonempty_rows());
  for (unsigned int i = 0; i < patches_pattern_fine.n_rows(); ++i)
    identity.add(i, i);
  DynamicSparsityPattern patches_pattern_fine_T;
  patches_pattern_fine_T.compute_Tmmult_pattern(patches_pattern_fine, identity);
  // premultiplied_basis_matrix.reinit(patches_pattern_fine_T.nonempty_rows(),
  //                                   patches_pattern_fine_T.nonempty_cols(),
  //                                   patches_pattern_fine_T,
  //                                   mpi_communicator);
  //
  // basis_matrix_transposed.reinit(patches_pattern_fine_T.nonempty_rows(),
  //                                patches_pattern_fine_T.nonempty_cols(),
  //                                patches_pattern_fine_T,
  //                                mpi_communicator);
  // FIX FOR MPI TODO !!!!
  premultiplied_basis_matrix.reinit(patches_pattern_fine_T);
  basis_matrix_transposed.reinit(patches_pattern_fine_T);

  /*
  premultiplied_basis_matrix.reinit(
      patches_pattern_fine.nonempty_rows(),
      patches_pattern_fine.nonempty_cols(),
      patches_pattern_fine,
      mpi_communicator);
  */
  // basis_matrix               = 0.0;
  premultiplied_basis_matrix = 0.0;
  basis_matrix_transposed    = 0.0;


  system_rhs.reinit(patches_pattern_fine_T.nonempty_cols(), mpi_communicator);

  Vector<double>            phi_loc(fe_fine->n_dofs_per_cell());
  std::vector<unsigned int> global_dofs(fe_fine->n_dofs_per_cell());

  for (auto current_patch_id : locally_owned_patches)
    {
      const auto current_patch = &patches[current_patch_id];
      dh_fine_current_patch.reinit(current_patch->sub_tria);
      dh_fine_current_patch.distribute_dofs(*fe_fine);

      for (auto iterator_to_cell_in_current_patch :
           dh_fine_current_patch.active_cell_iterators())
        {
          auto iterator_to_cell_global =
            current_patch
              ->cells[iterator_to_cell_in_current_patch->active_cell_index()]
              ->as_dof_handler_iterator(dof_handler_fine);
          iterator_to_cell_global->get_dof_indices(global_dofs);

          for (unsigned int d = 0; d < spacedim; ++d)
            {
              iterator_to_cell_in_current_patch->get_dof_values(
                current_patch->basis_function[d], phi_loc);
              AssertDimension(global_dofs.size(), phi_loc.size());
              // basis_matrix.set(current_patch_id,
              //                  phi_loc.size(),
              //                  global_dofs.data(),
              //                  phi_loc.data());
              for (unsigned int idx = 0; idx < phi_loc.size(); ++idx)
                {
                  basis_matrix_transposed.set(global_dofs.data()[idx],
                                              spacedim * current_patch_id + d,
                                              phi_loc.data()[idx]);
                }

              iterator_to_cell_in_current_patch->get_dof_values(
                current_patch->basis_function_premultiplied[d], phi_loc);
              AssertDimension(global_dofs.size(), phi_loc.size());
              // premultiplied_basis_matrix.set(current_patch_id,
              //                                phi_loc.size(),
              //                                global_dofs.data(),
              //                                phi_loc.data());
              // if the matrix is already transposed we need to loop to add the
              // elements
              for (unsigned int idx = 0; idx < phi_loc.size(); ++idx)
                {
                  premultiplied_basis_matrix.set(global_dofs.data()[idx],
                                                 spacedim * current_patch_id +
                                                   d,
                                                 phi_loc.data()[idx]);
                }
            }
        }
    }
  // basis_matrix.compress(VectorOperation::insert);
  premultiplied_basis_matrix.compress(VectorOperation::insert);
  basis_matrix_transposed.compress(VectorOperation::insert);

  basis_matrix_transposed.Tmmult(global_stiffness_matrix,
                                 premultiplied_basis_matrix);
  global_stiffness_matrix.compress(VectorOperation::add);
}


template <int dim, int spacedim>
void
LOD<dim, spacedim>::solve()
{

  basis_matrix_transposed.Tvmult(system_rhs, fem_rhs);
  pcout << "     rhs l2 norm = " << system_rhs.l2_norm() << std::endl;

    {
      SolverControl solver_control(1e2, 1e-2, false, false);
      TrilinosWrappers::SolverDirect solver(solver_control);
      solver.initialize(global_stiffness_matrix);
      solver.solve(global_stiffness_matrix, solution, system_rhs);
    }

  pcout << "   size of u " << solution.size() << std::endl;
  coarse_boundary_constraints.distribute(solution);
}

template <int dim, int spacedim>
void
LOD<dim, spacedim>::assemble_and_solve_fem_problem() //_and_compare() // const
{
  const auto &dh = dof_handler_fine;

  auto     locally_owned_dofs = dh.locally_owned_dofs();
  IndexSet locally_relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(dh, locally_relevant_dofs);

  // create sparsity pattern fr global fine matrix
  AffineConstraints<double> fem_constraints(locally_relevant_dofs);
  // DoFTools::make_hanging_node_constraints(dh, fem_constraints); // not needed
  // with global refinement
  VectorTools::interpolate_boundary_values(dh, 0, Functions::ZeroFunction<dim>(), fem_constraints);
  fem_constraints.close();

  LA::MPI::SparseMatrix fem_stiffness_matrix;

  {
    SparsityPattern        sparsity_pattern;
    DynamicSparsityPattern dsp(dh.n_dofs());

    std::vector<types::global_dof_index> dofs_on_this_cell;

    for (const auto &cell : dh.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          const unsigned int dofs_per_cell = cell->get_fe().n_dofs_per_cell();
          dofs_on_this_cell.resize(dofs_per_cell);
          cell->get_dof_indices(dofs_on_this_cell);

          fem_constraints.add_entries_local_to_global(
            dofs_on_this_cell,
            dsp,
            true,
            bool_dof_mask); // keep constrained entries must be true
        }

    dsp.compress();
    sparsity_pattern.copy_from(dsp);
    fem_stiffness_matrix.reinit(sparsity_pattern);
  }

  fem_rhs.reinit(locally_owned_dofs, mpi_communicator);
  fem_solution.reinit(locally_owned_dofs, mpi_communicator);

  LA::MPI::Vector locally_relevant_solution(locally_owned_dofs,
                                            locally_relevant_dofs,
                                            mpi_communicator);

  assemble_stiffness(fem_stiffness_matrix, fem_rhs, dh, fem_constraints);

  pcout << "     fem rhs l2 norm = " << fem_rhs.l2_norm() << std::endl;

    {
      SolverControl solver_control(1e2, 1e-2, false, false);
      TrilinosWrappers::SolverDirect solver(solver_control);
      solver.initialize(fem_stiffness_matrix);
      solver.solve(fem_stiffness_matrix, fem_solution, fem_rhs);
    }

  pcout << "   size of fem u " << fem_solution.size() << std::endl;
  fem_constraints.distribute(fem_solution);
}


template <int dim, int spacedim>
void
LOD<dim, spacedim>::compare_lod_with_fem()
{
  const auto &dh = dof_handler_fine;

  LA::MPI::Vector lod_solution(patches_pattern_fine.nonempty_cols(),
                               mpi_communicator);
  lod_solution = 0;

  basis_matrix_transposed.vmult(lod_solution, solution);

  // output fem solution
  std::vector<std::string> fem_names(spacedim, "fem_reference");
  std::vector<std::string> lod_names(spacedim, "lod_solution");

  data_out.attach_dof_handler(dh);

  data_out.add_data_vector(fem_solution,
                           fem_names,
                           DataOut<dim>::type_dof_data,
                           data_component_interpretation);
  data_out.add_data_vector(lod_solution,
                           lod_names,
                           DataOut<dim>::type_dof_data,
                           data_component_interpretation);

  data_out.build_patches();
  const std::string filename = "solution_fine.vtu";
  data_out.write_vtu_in_parallel(filename,
                                 mpi_communicator);

  // std::ofstream pvd_solutions(par.output_directory + "/" + par.output_name +
  //                             "_fine.pvd");
  data_out.clear();
}

template <int dim, int spacedim>
void
LOD<dim, spacedim>::initialize_patches()
{
  create_patches();
  // MPI Barrier

  for (auto current_patch_id : locally_owned_patches)
    {
      AssertIndexRange(current_patch_id, patches.size());
      auto current_patch = &patches[current_patch_id];
      create_mesh_for_patch(*current_patch);
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////


template <int dim, int spacedim>
void
LOD<dim, spacedim>::run()
{
  make_grid();
  make_fe();
  initialize_patches();

  compute_basis_function_candidates();
  assemble_global_matrix();
  assemble_and_solve_fem_problem();
  solve();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <int dim>
class problem_parameter : public Function<dim, double>
{
private:
  const double        min_val;
  const double        max_val;
  const unsigned int  refinement;
  std::vector<double> random_values;
  unsigned int        N_cells_per_line;
  double              eta;

public:
  problem_parameter(double min, double max, unsigned int r)
    : min_val(min)
    , max_val(max)
    , refinement(r)
  {
    N_cells_per_line     = pow(2, refinement);
    eta                  = (double)1 / N_cells_per_line;
    unsigned int N_cells = pow(N_cells_per_line, dim);
    // random_values.reinit(N_cells);
    if (max_val != min_val)
      {
        for (unsigned int i = 0; i < N_cells; ++i)
          {
            const double v =
              min_val + static_cast<float>(rand()) /
                          (static_cast<float>(RAND_MAX / (max_val - min_val)));
            random_values.push_back(v);
          }
      }
  };

  double
  value(const Point<dim> &p, const unsigned int) const override
  {
    if (max_val == min_val) // constant coefficients
      return min_val;
    else
      {
        const double x = p(0);
        const double y = p(1);
        unsigned int vector_cell_index =
          (int)floor(x / eta) + N_cells_per_line * (int)floor(y / eta);
        return random_values[vector_cell_index];
      }
  }
};


template <int dim, int spacedim>
class DiffusionProblem : public LOD<dim, spacedim>
{
public:
  DiffusionProblem()
    : LOD<dim, spacedim>()
    , Alpha(1, 100, 8){};

  typedef LOD<dim, spacedim> lod;


protected:
  problem_parameter<dim> Alpha;


  virtual void
  assemble_stiffness(LA::MPI::SparseMatrix &    stiffness_matrix,
                     LA::MPI::Vector &          rhs,
                     const DoFHandler<dim> &    dh,
                     AffineConstraints<double> &stiffness_constraints) override
  {
    stiffness_matrix = 0;
    if (rhs.size())
      rhs = 0;

    FEValues<dim> fe_values(*lod::fe_fine,
                            *lod::quadrature_fine,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = lod::fe_fine->n_dofs_per_cell();
    const unsigned int n_q_points    = lod::quadrature_fine->size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    std::vector<double>                  rhs_values(n_q_points);
    std::vector<double>                  alpha_values(n_q_points);
    // using n_q_points gives us a vectore much larger than what we need due to
    // the Q_iso_Q1 nature

    const auto lexicographic_to_hierarchic_numbering =
      FETools::lexicographic_to_hierarchic_numbering<dim>(
        lod::n_subdivisions);


    for (const auto &cell : dh.active_cell_iterators())
      {
        cell_matrix = 0;
        cell_rhs    = 0;
        fe_values.reinit(cell);

        if (rhs.size())
          {
            lod::forcing.value_list(fe_values.get_quadrature_points(),
                                    rhs_values);
          }
        Alpha.value_list(fe_values.get_quadrature_points(), alpha_values);

        for (unsigned int c_1 = 0; c_1 < lod::n_subdivisions; ++c_1)
          for (unsigned int c_0 = 0; c_0 < lod::n_subdivisions; ++c_0)
            for (unsigned int q_1 = 0; q_1 < 2; ++q_1)
              for (unsigned int q_0 = 0; q_0 < 2; ++q_0)
                {
                  const unsigned int q_index =
                    (c_0 * 2 + q_0) +
                    (c_1 * 2 + q_1) * (2 * lod::n_subdivisions);

                  for (unsigned int i_1 = 0; i_1 < 2; ++i_1)
                    for (unsigned int i_0 = 0; i_0 < 2; ++i_0)
                      {
                        const unsigned int i =
                          lexicographic_to_hierarchic_numbering
                            [(c_0 + i_0) +
                             (c_1 + i_1) * (lod::n_subdivisions + 1)];

                        for (unsigned int j_1 = 0; j_1 < 2; ++j_1)
                          for (unsigned int j_0 = 0; j_0 < 2; ++j_0)
                            {
                              const unsigned int j =
                                lexicographic_to_hierarchic_numbering
                                  [(c_0 + j_0) +
                                   (c_1 + j_1) * (lod::n_subdivisions + 1)];

                              cell_matrix(i, j) +=
                                // alpha[vector_cell_index] *
                                alpha_values[q_index] *
                                (fe_values.shape_grad(i, q_index) *
                                 fe_values.shape_grad(j, q_index) *
                                 fe_values.JxW(q_index));
                            }
                        if (rhs.size())
                          cell_rhs(i) += fe_values.shape_value(i, q_index) *
                                         rhs_values[q_index] *
                                         fe_values.JxW(q_index);
                      }
                }

        cell->get_dof_indices(local_dof_indices);

        if (rhs.size())
          stiffness_constraints.distribute_local_to_global(
            cell_matrix, cell_rhs, local_dof_indices, stiffness_matrix, rhs);
        else
          stiffness_constraints.distribute_local_to_global(cell_matrix,
                                                           local_dof_indices,
                                                           stiffness_matrix);
      }
    stiffness_matrix.compress(VectorOperation::add);
    rhs.compress(VectorOperation::add);
  };
};



int
main(int argc, char *argv[])
{
  using namespace dealii;
  deallog.depth_console(1);

  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      DiffusionProblem<2, 1> problem;
      problem.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  return 0;
}
