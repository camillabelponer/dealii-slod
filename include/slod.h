#ifndef dealii_slod_h
#define dealii_slod_h

#include <deal.II/base/exceptions.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/types.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q_iso_q1.h>
#include <deal.II/fe/fe_tools.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/petsc_full_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_matrix_free.h>
#include <deal.II/lac/slepc_solver.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/arpack_solver.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/multigrid/mg_transfer_global_coarsening.h>
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
  std::vector<typename DoFHandler<dim>::active_cell_iterator>    cells;
  IndexSet    cell_indices;
  Triangulation<dim>               sub_tria;
  std::unique_ptr<DoFHandler<dim>>                  dh_fine;
  std::vector<LinearAlgebra::distributed::Vector<double>> basis_function_candidates;
};

template <int dim>
class SLOD
{
public:
  SLOD(MPI_Comm mpi_communicator);

  void
  make_fe();
  void
  make_grid();
  void
  create_patches();
  void
  compute_basis_function_candidates();
  void
  stabilize();
  void
  assemble_global_matrix();

private:
  MPI_Comm mpi_communicator;
  void
  create_mesh_for_patch(Patch<dim> &current_patch);
  void
  assemble_stiffness_for_patch(Patch<dim> &        current_patch,
                               LA::MPI::SparseMatrix &stiffness_matrix);

  unsigned int oversampling         = 1;
  unsigned int n_subdivisions       = 5;
  unsigned int n_global_refinements = 2;
  unsigned int num_basis_vectors    = 10;

  Triangulation<dim> tria;
  DoFHandler<dim>    dof_handler;

  LA::MPI::SparseMatrix global_matrix;
  // TODO: Add rhs

  std::unique_ptr<FE_DGQ<dim>> fe_coarse;
  std::unique_ptr<FE_Q_iso_Q1<dim>> fe_fine;
  std::unique_ptr<Quadrature<dim>>    quadrature_fine;

  // TODO: This should be an MPI vector
  std::vector<Patch<dim>> patches;
};

// template <int dim>
// class TransferWrapper {
// public:
//   TransferWrapper(MGTwoLevelTransfer<dim, LinearAlgebra::distributed::Vector<double>> &transfer, unsigned int n_coarse, unsigned int n_fine);
//   void vmult(LinearAlgebra::distributed::Vector<double> &out, const LinearAlgebra::distributed::Vector<double> &in) const;
//   void Tvmult(LinearAlgebra::distributed::Vector<double> &out, const LinearAlgebra::distributed::Vector<double> &in) const;
//   unsigned int m() const;
//   unsigned int n() const;
// private:
//   MGTwoLevelTransfer<dim, LinearAlgebra::distributed::Vector<double>> &transfer;
//   unsigned int n_coarse;
//   unsigned int n_fine;
// };

#endif
