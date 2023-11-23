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

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/petsc_full_matrix.h>
#include <deal.II/lac/slepc_solver.h>
#include <deal.II/lac/solver_control.h>

#include <deal.II/numerics/vector_tools.h>


#define FORCE_USE_OF_TRILINOS
namespace LA
{
#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) && \
  !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
  using namespace dealii::LinearAlgebraPETSc;
#  define USE_PETSC_LA
#elif defined(DEAL_II_WITH_TRILINOS)
  using namespace dealii::LinearAlgebraTrilinos;
#else
#  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
} // namespace LA



using namespace dealii;

struct Patch {
  IndexSet cells;
  Triangulation<dim> sub_tria;
  DoFHandler<dim> dh_fine;
  std::vector<std::vector<double>> basis_function_candidates;
};

template<int dim>
class SLOD {
  public:
    SLOD(Triangulation<dim> tria);

    void make_fe();
    void create_patches();
    void compute_basis_function_candidates();
    void stabilize();
    void assemble_global_matrix();
  private:
    void create_mesh_for_patch(unsigned int patch_id);
    void assemble_stiffness_for_patch(unsigned int patch_id, FullMatrix<double> &stiffness);
    void assemble_rhs_fine_from_coarse(unsigned int patch_id, std::vector<double> &coarse_vec, Vector<double> &fine_vec);

    unsigned int oversampling           = 1;
    unsigned int n_subdivisions         = 5;
    unsigned int n_global_refinements   = 2;
    unsigned int num_basis_vectors = 10;

    Triangulation<dim> tria;
    DoFHandler<dim> dof_handler;

    LA::MPI::SparseMatrix global_matrix;
    // TODO: Add rhs

    std::unique_ptr<FiniteElement<dim>> fe_coarse;
    std::unique_ptr<FiniteElement<dim>> fe_fine;
    std::unique_ptr<Quadrature<dim>>    quadrature_fine;

    // TODO: This should be an MPI vector
    std::vector<Patch>                  patches;
};

#endif
