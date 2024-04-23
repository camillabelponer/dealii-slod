#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_q_iso_q1.h>
#include <deal.II/fe/fe_tools.h>

#include <deal.II/grid/tria.h>

#include <deal.II/multigrid/mg_transfer_global_coarsening.h>

using namespace dealii;

void
test()
{
  const unsigned int dim            = 2;
  const unsigned int n_subdivisions = 5;
  using Number                      = double;
  using VectorType                  = Vector<Number>;

  // Create patch mesh
  Triangulation<dim> tria;

  // TODO: create mesh

  // Create coarse system
  FE_DGQ<dim>       fe_coarse(0);
  DoFHandler<dim> dof_handler_coarse(tria);
  dof_handler_coarse.distribute_dofs(fe_coarse);

  // create fine system
  FE_Q_iso_Q1<dim> fe_fine(n_subdivisions);
  DoFHandler<dim>  dof_handler_fine(tria);
  dof_handler_fine.distribute_dofs(fe_fine);

  // create projection matrix from fine to coarse cell (DG)
  FullMatrix<Number> projection_matrix;
  FETools::get_projection_matrix(fe_fine, fe_coarse, projection_matrix);

  // avereging
  VectorType valence_coarse(dof_handler_coarse.n_dofs());
  VectorType local_identity_coarse(fe_coarse.n_dofs_per_cell());
  local_identity_coarse = 1.0;

  for (const auto &cell : dof_handler_coarse.active_cell_iterators())
    cell->distribute_local_to_global(local_identity_coarse, valence_coarse);

  for (auto &i : valence_coarse)
    i = 1.0 / i;

  // define interapolation function and its transposed
  const auto interpolate = [&](auto &dst, const auto &src) {
    VectorType vec_local_coarse(fe_coarse.n_dofs_per_cell());
    VectorType vec_local_fine(fe_fine.n_dofs_per_cell());
    VectorType weights(fe_coarse.n_dofs_per_cell());

    for (const auto &cell : tria.active_cell_iterators())
      {
        const auto cell_coarse =
          cell->as_dof_handler_iterator(dof_handler_coarse);
        const auto cell_fine = cell->as_dof_handler_iterator(dof_handler_fine);

        cell_fine->get_dof_values(src, vec_local_fine);

        projection_matrix.vmult(vec_local_coarse, vec_local_fine);

        cell_fine->get_dof_values(valence_coarse, weights);
        vec_local_coarse.scale(weights);

        cell_coarse->distribute_local_to_global(vec_local_coarse, dst);
      }
  };

  const auto interpolateT = [&](auto &dst, const auto &src) {
    VectorType vec_local_coarse(fe_coarse.n_dofs_per_cell());
    VectorType vec_local_fine(fe_fine.n_dofs_per_cell());
    VectorType weights(fe_coarse.n_dofs_per_cell());

    for (const auto &cell : tria.active_cell_iterators())
      {
        const auto cell_coarse =
          cell->as_dof_handler_iterator(dof_handler_coarse);
        const auto cell_fine = cell->as_dof_handler_iterator(dof_handler_fine);

        cell_coarse->get_dof_values(src, vec_local_coarse);

        cell_fine->get_dof_values(valence_coarse, weights);
        vec_local_coarse.scale(weights);

        projection_matrix.Tvmult(vec_local_fine, vec_local_coarse);

        cell_fine->distribute_local_to_global(vec_local_fine, dst);
      }
  };

  // test interpoalation and its transposed
  VectorType vec_coarse(dof_handler_coarse.n_dofs());
  VectorType vec_fine(dof_handler_fine.n_dofs());

  interpolate(vec_coarse, vec_fine);
  interpolateT(vec_fine, vec_coarse);
}

void
test_mg()
{
  const unsigned int dim            = 2;
  const unsigned int n_subdivisions = 5;
  using Number                      = double;
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  // Create patch mesh
  Triangulation<dim> tria;

  // Create coarse system
  FE_DGQ<dim>       fe_coarse(0);
  DoFHandler<dim> dof_handler_coarse(tria);
  dof_handler_coarse.distribute_dofs(fe_coarse);

  // create fine system
  FE_Q_iso_Q1<dim> fe_fine(n_subdivisions);
  DoFHandler<dim>  dof_handler_fine(tria);
  dof_handler_fine.distribute_dofs(fe_fine);

  MGTwoLevelTransfer<dim, VectorType> transfer;
  transfer.reinit_polynomial_transfer(dof_handler_coarse, dof_handler_fine);

  VectorType vec_coarse(dof_handler_coarse.n_dofs());
  VectorType vec_fine(dof_handler_fine.n_dofs());

  // interpolate
  transfer.prolongate_and_add(vec_coarse, vec_fine);

  // interpolateT
  transfer.restrict_and_add(vec_fine, vec_coarse);
}


int
main()
{
  return 0; // only test if program compiles

  test(); // manually

  test_mg(); // with MG infrastructure
}