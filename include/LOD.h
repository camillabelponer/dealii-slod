#ifndef dealii_lod_h
#define dealii_lod_h

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parsed_convergence_table.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/timer.h>
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


template <int dim, int spacedim>
class LODParameters : public ParameterAcceptor
{
public:
  LODParameters();

  std::string  output_directory     = ".";
  std::string  output_name          = "solution";
  unsigned int oversampling         = 1;
  unsigned int n_subdivisions       = 5;
  unsigned int n_global_refinements = 2;
  bool         solve_fine_problem   = false;
  bool         LOD_stabilization    = false;

  mutable ParameterAcceptorProxy<Functions::ParsedFunction<dim>> rhs;
  mutable ParameterAcceptorProxy<Functions::ParsedFunction<dim>> exact_solution;
  mutable ParameterAcceptorProxy<Functions::ParsedFunction<dim>> bc;

  mutable ParameterAcceptorProxy<ReductionControl> fine_solver_control;
  mutable ParameterAcceptorProxy<ReductionControl> coarse_solver_control;
  mutable ParameterAcceptorProxy<ReductionControl> patch_solver_control;

  mutable ParsedConvergenceTable convergence_table_LOD;
  mutable ParsedConvergenceTable convergence_table_FEM;
  mutable ParsedConvergenceTable convergence_table_compare;
};



template <int dim, int spacedim>
LODParameters<dim, spacedim>::LODParameters()
  : ParameterAcceptor("/Problem")
  , rhs("/Problem/Right hand side", spacedim)
  , exact_solution("/Problem/Exact solution", spacedim)
  , bc("/Problem/Dirichlet boundary conditions", spacedim)
  , fine_solver_control("/Problem/Solver/Fine solver control")
  , coarse_solver_control("/Problem/Solver/Coarse solver control")
  , patch_solver_control("/Problem/Solver/Patch solver control")
  , convergence_table_LOD(std::vector<std::string>(spacedim, "u"))
  , convergence_table_FEM(std::vector<std::string>(spacedim, "u"))
  , convergence_table_compare(std::vector<std::string>(spacedim, "u"))
{
  add_parameter("Output directory", output_directory);
  add_parameter("Output name", output_name);
  add_parameter("oversampling", oversampling);
  add_parameter("Number of subdivisions", n_subdivisions);
  add_parameter("Number of global refinements", n_global_refinements);
  add_parameter("Compare with fine global solution", solve_fine_problem);
  add_parameter("Stabilize phi_LOD candidates", LOD_stabilization);
  this->prm.enter_subsection("Error");
  convergence_table_LOD.add_parameters(this->prm);
  convergence_table_FEM.add_parameters(this->prm);
  convergence_table_compare.add_parameters(this->prm);
  this->prm.leave_subsection();
}

template <int dim, int spacedim>
class LOD
{
public:
  LOD(const LODParameters<dim, spacedim> &par);

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
  // virtual
  void
  output_coarse_results();
  // virtual
  void
  output_fine_results(){};
  void
  print_parameters() const;
  void
  initialize_patches();
  void
  create_random_coefficients();

  const LODParameters<dim, spacedim> &par;
  MPI_Comm                            mpi_communicator;
  ConditionalOStream                  pcout;
  mutable TimerOutput                 computing_timer;

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
  // chek ghost layer, needs to be set to whole domain
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

  Vector<double> random_coefficients;

  DataOut<dim> data_out;
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation;
};


#endif