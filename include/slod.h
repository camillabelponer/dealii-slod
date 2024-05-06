#ifndef dealii_slod_h
#  define dealii_slod_h

#  include <deal.II/base/conditional_ostream.h>
#  include <deal.II/base/exceptions.h>
#  include <deal.II/base/parameter_acceptor.h>
#  include <deal.II/base/parsed_convergence_table.h>
#  include <deal.II/base/parsed_function.h>
#  include <deal.II/base/quadrature.h>
#  include <deal.II/base/timer.h>
#  include <deal.II/base/types.h>

#  include <deal.II/distributed/grid_refinement.h>
#  include <deal.II/distributed/solution_transfer.h>
#  include <deal.II/distributed/tria.h>

#  include <deal.II/dofs/dof_tools.h>

#  include <deal.II/fe/fe_dgq.h>
#  include <deal.II/fe/fe_nothing.h>
#  include <deal.II/fe/fe_q.h>
#  include <deal.II/fe/fe_q_iso_q1.h>
#  include <deal.II/fe/fe_system.h>
#  include <deal.II/fe/fe_values.h>
#  include <deal.II/fe/mapping_fe_field.h>
#  include <deal.II/fe/mapping_q.h>

#  include <deal.II/grid/grid_generator.h>
#  include <deal.II/grid/grid_tools.h>
#  include <deal.II/grid/intergrid_map.h>

#  include <deal.II/lac/arpack_solver.h>
#  include <deal.II/lac/dynamic_sparsity_pattern.h>
#  include <deal.II/lac/full_matrix.h>
#  include <deal.II/lac/generic_linear_algebra.h>
#  include <deal.II/lac/la_parallel_vector.h>
#  include <deal.II/lac/lapack_full_matrix.h>
#  include <deal.II/lac/linear_operator.h>
#  include <deal.II/lac/linear_operator_tools.h>
#  include <deal.II/lac/petsc_full_matrix.h>
#  include <deal.II/lac/petsc_matrix_free.h>
#  include <deal.II/lac/petsc_vector.h>
#  include <deal.II/lac/slepc_solver.h>
#  include <deal.II/lac/solver_control.h>
#  include <deal.II/lac/solver_minres.h>
#  include <deal.II/lac/sparsity_tools.h>
#  include <deal.II/lac/trilinos_vector.h>

#  include <deal.II/multigrid/mg_transfer_global_coarsening.h>

#  include <deal.II/numerics/data_out.h>
#  include <deal.II/numerics/vector_tools.h>

#  include <memory>


namespace LA
{
  using namespace dealii::LinearAlgebraTrilinos;
} // namespace LA



using namespace dealii;

template <int dim>
class Patch
{
public:
  // TODO
  // make everything private and write getter and setter functions

  // coarse cells that make up the patch
  std::vector<typename DoFHandler<dim>::active_cell_iterator> cells;
  // IndexSet                                                    cell_indices;
  Triangulation<dim> sub_tria;
  // std::unique_ptr<DoFHandler<dim>>                            dh_fine;
  // change!!! cannot be unique and cannot point at stuff that's destructed
  Vector<double> basis_function;
  Vector<double> basis_function_premultiplied;
  unsigned int   contained_patches = 0;
  unsigned int   patch_id;
};


template <int dim, int spacedim = dim>
class SLODParameters : public ParameterAcceptor
{
public:
  SLODParameters();

  std::string  output_directory     = ".";
  std::string  output_name          = "solution";
  unsigned int oversampling         = 1;
  unsigned int n_subdivisions       = 5;
  unsigned int n_global_refinements = 2;
  unsigned int num_basis_vectors    = 1;
  unsigned int p_order_on_patch     = 2;
  bool         solve_fine_problem   = false;

  mutable ParameterAcceptorProxy<Functions::ParsedFunction<dim>> rhs;
  mutable ParameterAcceptorProxy<Functions::ParsedFunction<dim>> exact_solution;
  mutable ParameterAcceptorProxy<Functions::ParsedFunction<dim>> bc;

  mutable ParameterAcceptorProxy<ReductionControl> fine_solver_control;
  mutable ParameterAcceptorProxy<ReductionControl> coarse_solver_control;

  mutable ParsedConvergenceTable convergence_table_SLOD;
  mutable ParsedConvergenceTable convergence_table_FEM;
  mutable ParsedConvergenceTable convergence_table_compare;
};



template <int dim, int spacedim>
SLODParameters<dim, spacedim>::SLODParameters()
  : ParameterAcceptor("/Problem")
  , rhs("/Problem/Right hand side")              //, dim-1)
  , exact_solution("/Problem/Exact solution")    //, dim-1)
  , bc("/Problem/Dirichlet boundary conditions") //, dim-1)
  , fine_solver_control("/Problem/Solver/Fine solver control")
  , coarse_solver_control("/Problem/Solver/Coarse solver control")
{
  add_parameter("Output directory", output_directory);
  add_parameter("Output name", output_name);
  add_parameter("oversampling", oversampling);
  add_parameter("Number of subdivisions", n_subdivisions);
  add_parameter("Number of global refinements", n_global_refinements);
  add_parameter("Number of basis vectors", num_basis_vectors);
  add_parameter("order polynomial on patch", p_order_on_patch);
  add_parameter("Compare with fine global solution", solve_fine_problem);
  this->prm.enter_subsection("Error");
  convergence_table_SLOD.add_parameters(this->prm);
  convergence_table_FEM.add_parameters(this->prm);
  convergence_table_compare.add_parameters(this->prm);
  this->prm.leave_subsection();
}

template <int dim>
class SLOD
{
public:
  SLOD(const SLODParameters<dim, dim> &par);

  void
  run();

private:
  void
  make_fe();
  void
  make_grid();
  void
  create_patches();
  void
  compute_basis_function_candidates();
  void
  compute_basis_function_candidates_using_SVD(){};
  void
  assemble_global_matrix();
  void
  solve();
  void
  solve_fine_problem_and_compare(); // const;
  void
  output_results();
  void
  print_parameters() const;
  void
  initialize_patches();

  const SLODParameters<dim, dim> &par;
  MPI_Comm                        mpi_communicator;
  ConditionalOStream              pcout;
  mutable TimerOutput             computing_timer;
  // std::unique_ptr<Quadrature<dim>>             quadrature_coarse;

  void
  create_mesh_for_patch(Patch<dim> &current_patch);
  void
  check_nested_patches(); // AFTER PATCHES ARE CREATED
  void
  assemble_stiffness_for_patch( // Patch<dim> &           current_patch,
    LA::MPI::SparseMatrix &    stiffness_matrix,
    const DoFHandler<dim> &    dh,
    AffineConstraints<double> &local_stiffnes_constraints);

  parallel::distributed::Triangulation<dim> tria;
  DoFHandler<dim>                           dof_handler_coarse;
  DoFHandler<dim>                           dof_handler_fine;

  // AffineConstraints<double> constraints;

  LA::MPI::SparseMatrix basis_matrix;
  LA::MPI::SparseMatrix premultiplied_basis_matrix;
  LA::MPI::SparseMatrix global_stiffness_matrix;
  LA::MPI::Vector       solution;
  LA::MPI::Vector       system_rhs;

  std::unique_ptr<FE_DGQ<dim>>      fe_coarse;
  std::unique_ptr<FE_Q_iso_Q1<dim>> fe_fine;
  // std::unique_ptr<FiniteElement<dim>> fe_coarse;
  // std::unique_ptr<FiniteElement<dim>> fe_fine;
  std::unique_ptr<Quadrature<dim>> quadrature_fine;

  std::vector<Patch<dim>> patches;
  DynamicSparsityPattern  patches_pattern;
  DynamicSparsityPattern  patches_pattern_fine;
  // TrilinosWrappers::MPI::Vector patches;

  // List of lists of pairs (patch ID, cell in patch mesh)
  // The index in the outer list corresponds to the index of the cell in the
  // global (coarse) mesh and the inner lists store the information, which
  // patches contain this global cell.
  // std::vector<std::vector<
  //   std::pair<unsigned int, typename
  //   Triangulation<dim>::active_cell_iterator>>> global_to_local_cell_map;

  IndexSet locally_owned_patches;
  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;
};

// template <int dim>
// class TransferWrapper {
// public:
//   TransferWrapper(MGTwoLevelTransfer<dim,
//   LinearAlgebra::distributed::Vector<double>> &transfer, unsigned int
//   n_coarse, unsigned int n_fine); void
//   vmult(LinearAlgebra::distributed::Vector<double> &out, const
//   LinearAlgebra::distributed::Vector<double> &in) const; void
//   Tvmult(LinearAlgebra::distributed::Vector<double> &out, const
//   LinearAlgebra::distributed::Vector<double> &in) const; unsigned int m()
//   const; unsigned int n() const;
// private:
//   MGTwoLevelTransfer<dim, LinearAlgebra::distributed::Vector<double>>
//   &transfer; unsigned int n_coarse; unsigned int n_fine;
// };

#endif

// NOTE
// ONLY WORK WITH DIRICHLET = 0 EVERYWHERE because of LOD formulation


// 3.4.24
// TODO
// compute sp-1 blablabla
// asseble c
// asseble ct S c
// check Solve
// fix internal boundary
// check scaling inside "transfer"

// 10.4.24
// TODO
// testing LOD
// stabilization
// elasticity
// parallelization
// optimize C^T S C

// 30 04 24 : stabilization
/*
 get vector of cand
 define conormal op.
 apply op (B) to cand(0)
 apply b to all other candidates
 solve somehow d = B-1 B0 cannot do that then
 svd (Parpack)
 truncate by throwing away as many patches as this patches containes ( the
smallest) etc
*/
// cosntruct B
// take first col of B as rhs
// the rest is B_r
// SVD of B_r ^T B_r and then truncate