#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_q_iso_q1.h>
#include <deal.II/fe/fe_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/multigrid/mg_transfer_global_coarsening.h>

using namespace dealii;

void
test()
{
  const unsigned int n_global_ref   = 2;
  const unsigned int dim            = 2;
  const unsigned int n_subdivisions = 5;
  using Number                      = double;
  using VectorType                  = Vector<Number>;

  // Create patch mesh
  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria);
  tria.refine_global(n_global_ref);

  // Create coarse system
  FE_DGQ<dim>     fe_coarse(0);
  DoFHandler<dim> dof_handler_coarse(tria);
  dof_handler_coarse.distribute_dofs(fe_coarse);

  // create fine system
  FE_Q_iso_Q1<dim> fe_fine(n_subdivisions);
  DoFHandler<dim>  dof_handler_fine(tria);
  dof_handler_fine.distribute_dofs(fe_fine);

  // create projection matrix from fine to coarse cell (DG)
  FullMatrix<Number> projection_matrix(fe_coarse.n_dofs_per_cell(),
                                       fe_fine.n_dofs_per_cell());
  FETools::get_projection_matrix(fe_fine, fe_coarse, projection_matrix);

  // averaging (inverse of P0 mass matrix)
  VectorType valence_coarse(dof_handler_coarse.n_dofs());
  VectorType local_identity_coarse(fe_coarse.n_dofs_per_cell());
  local_identity_coarse = 1.0;

  for (const auto &cell : dof_handler_coarse.active_cell_iterators())
    cell->distribute_local_to_global(local_identity_coarse, valence_coarse);

  for (auto &elem : valence_coarse)
    elem = 1.0 / elem;

  // define interapolation function and its transposed
  const auto projectT = [&](auto &dst, const auto &src) {
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

        cell_coarse->get_dof_values(valence_coarse, weights);
        vec_local_coarse.scale(weights);

        cell_coarse->distribute_local_to_global(vec_local_coarse, dst);
      }
  };

  const auto project = [&](auto &dst, const auto &src) {
    VectorType vec_local_coarse(fe_coarse.n_dofs_per_cell());
    VectorType vec_local_fine(fe_fine.n_dofs_per_cell());
    VectorType weights(fe_coarse.n_dofs_per_cell());

    for (const auto &cell : tria.active_cell_iterators())
      {
        const auto cell_coarse =
          cell->as_dof_handler_iterator(dof_handler_coarse);
        const auto cell_fine = cell->as_dof_handler_iterator(dof_handler_fine);

        cell_coarse->get_dof_values(src, vec_local_coarse);

        cell_coarse->get_dof_values(valence_coarse, weights);
        vec_local_coarse.scale(weights);

        projection_matrix.Tvmult(vec_local_fine, vec_local_coarse);

        cell_fine->distribute_local_to_global(vec_local_fine, dst);
      }
  };

  // test interpoalation and its transposed
  VectorType vec_coarse(dof_handler_coarse.n_dofs());
  VectorType vec_fine(dof_handler_fine.n_dofs());

  projectT(vec_coarse, vec_fine);
  project(vec_fine, vec_coarse);
}

void
test_mg()
{
  const unsigned int n_global_ref   = 2;
  const unsigned int dim            = 2;
  const unsigned int n_subdivisions = 5;
  using Number                      = double;
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  // Create patch mesh
  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria);
  tria.refine_global(n_global_ref);

  // Create coarse system
  FE_DGQ<dim>     fe_coarse(0);
  DoFHandler<dim> dof_handler_coarse(tria);
  dof_handler_coarse.distribute_dofs(fe_coarse);

  // create fine system
  FE_Q_iso_Q1<dim> fe_fine(n_subdivisions);
  DoFHandler<dim>  dof_handler_fine(tria);
  dof_handler_fine.distribute_dofs(fe_fine);

  MGTwoLevelTransfer<dim, VectorType> transfer;
  transfer.reinit_polynomial_transfer(dof_handler_fine, dof_handler_coarse);

  VectorType vec_coarse(dof_handler_coarse.n_dofs());
  VectorType vec_fine(dof_handler_fine.n_dofs());

  // interpolate
  transfer.restrict_and_add(vec_coarse, vec_fine);

  // interpolateT
  transfer.prolongate_and_add(vec_fine, vec_coarse);
}


int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  test(); // manually

  // THIS DOES NOT DO WHAT IT SHOULD
  // (the projection goes the wrong way)
  test_mg(); // with MG infrastructure
}