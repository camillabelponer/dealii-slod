#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_q_iso_q1.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>

using namespace dealii;

void
test_normal()
{
  const unsigned int dim                  = 2;
  const unsigned int n_global_refinements = 2;
  using Number                            = double;

  FE_Q<dim>   fe(1);
  QGauss<dim> quad(2);

  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria);
  tria.refine_global(n_global_refinements);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  // create sparsity pattern

  SparsityPattern sparsity_pattern(dof_handler.n_dofs(), dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, sparsity_pattern);
  sparsity_pattern.compress();

  // loop over all cells
  SparseMatrix<Number> A;
  A.reinit(sparsity_pattern);

  FEValues<dim>      fe_values(fe, quad, update_gradients | update_JxW_values);
  FullMatrix<Number> A_K(fe.n_dofs_per_cell(), fe.n_dofs_per_cell());
  AffineConstraints<Number> affine_constraints;

  std::vector<unsigned int> cell_indices(fe.n_dofs_per_cell());
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);

      cell->get_dof_indices(cell_indices);

      A_K = 0.0;

      for (const unsigned int q_index : fe_values.quadrature_point_indices())
        for (const unsigned int i : fe_values.dof_indices())
          for (const unsigned int j : fe_values.dof_indices())
            A_K(i, j) += (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                          fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                          fe_values.JxW(q_index));           // dx

      affine_constraints.distribute_local_to_global(A_K, cell_indices, A);
    }

  std::cout << A.frobenius_norm() << std::endl;
}

void
test_two_level()
{
  const unsigned int oversampling         = 1;
  const unsigned int dim                  = 2;
  const unsigned int n_subdivisions       = 5;
  const unsigned int n_global_refinements = 2;
  using Number                            = double;

  FE_DGQ<dim>      fe_coarse(0);
  FE_Q_iso_Q1<dim> fe_fine(n_subdivisions);
  QIterated<dim>   quad_fine(QGauss<1>(2), n_subdivisions);

  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria);
  tria.refine_global(n_global_refinements);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe_coarse);

  // variables for coarse cells/patches
  // cell_dof_indices[j] contains dofs of cell with id j
  // TODO: should this really be a vector of vectors?
  // Maybe a vector of index sets
  std::map<unsigned int, std::vector<types::global_dof_index>> cell_dof_indices;
  // patch_cell_indices[i] conatins cell ids that belong to patch centered at
  // cell i Should have (2 * oversampling + 1)^d elements max
  // TODO:
  // Maybe a map of sets
  std::map<unsigned int, IndexSet>             patch_cell_indices;
  std::vector<std::vector<FullMatrix<Number>>> patch_cell_constraints;

  // TODO: we fill the data structures cell_dof_indices,
  // patch_cell_indices, and patch_cell_constraints for a simple
  // two-level context; for LOD these have to be adjusted
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      std::vector<types::global_dof_index> dof_indices(
        fe_coarse.n_dofs_per_cell());
      cell->get_dof_indices(dof_indices);
      cell_dof_indices[cell->active_cell_index()] = dof_indices;

      // TODO: Handle special cases at the boundaries
      IndexSet cell_indices;
      for (unsigned int l = 1; l <= oversampling; l++)
        {
          auto neighbours =
            GridTools::get_patch_around_cell<DoFHandler<dim, dim>>(cell);
          for (const auto &neighbour : neighbours)
            {
              cell_indices.add_index(neighbour->active_cell_index());
            }
        }
      patch_cell_indices[cell->active_cell_index()] = cell_indices;

      // TODO
      FullMatrix<Number> matrix(fe_fine.n_dofs_per_cell(),
                                fe_coarse.n_dofs_per_cell());
      FETools::get_projection_matrix(fe_coarse, fe_fine, matrix);
      std::vector<FullMatrix<Number>> constraints;
      constraints.push_back(matrix);
      patch_cell_constraints.push_back(constraints);
    }

  // Solve local problems
  FEValues<dim>      fe_values(fe_fine,
                          quad_fine,
                          update_gradients | update_JxW_values);
  FullMatrix<Number> A_K_fem(fe_fine.n_dofs_per_cell(),
                             fe_fine.n_dofs_per_cell());
  for (const auto &cell : tria.active_cell_iterators())
    {
      const unsigned int id = cell->active_cell_index();

      // 1) assemble element stiffness matrix
      fe_values.reinit(cell);

      A_K_fem = 0.0;

      for (const unsigned int q_index : fe_values.quadrature_point_indices())
        for (const unsigned int i : fe_values.dof_indices())
          for (const unsigned int j : fe_values.dof_indices())
            A_K_fem(i, j) +=
              (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
               fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
               fe_values.JxW(q_index));           // dx
    }

  // TODO: This should be done later, first solve local problems
  // create sparsity pattern
  DynamicSparsityPattern dsp(dof_handler.n_dofs());

  for (auto const &pair : patch_cell_indices)
    {
      std::vector<types::global_dof_index> indices;
      for (const auto j : pair.second)
        for (const auto k : cell_dof_indices[j])
          indices.push_back(k);

      std::sort(patch_cell_indices.begin(), patch_cell_indices.end());
      patch_cell_indices.erase(std::unique(patch_cell_indices.begin(),
                                           patch_cell_indices.end()),
                               patch_cell_indices.end());

      for (const auto &i : indices)
        dsp.add_entries(i, indices.begin(), indices.end());
    }

  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);

  // loop over all cells
  SparseMatrix<Number> A_lod;
  A_lod.reinit(sparsity_pattern);

  FEValues<dim>      fe_values(fe_fine,
                          quad_fine,
                          update_gradients | update_JxW_values);
  FullMatrix<Number> A_K_fem(fe_fine.n_dofs_per_cell(),
                             fe_fine.n_dofs_per_cell());

  FullMatrix<Number>        A_K_lod(fe_coarse.n_dofs_per_cell(),
                             fe_coarse.n_dofs_per_cell());
  AffineConstraints<Number> affine_constraints;

  for (const auto &cell : tria.active_cell_iterators())
    {
      const unsigned int id = cell->active_cell_index();

      // 1) assemble element stiffness matrix
      fe_values.reinit(cell);

      A_K_fem = 0.0;

      for (const unsigned int q_index : fe_values.quadrature_point_indices())
        for (const unsigned int i : fe_values.dof_indices())
          for (const unsigned int j : fe_values.dof_indices())
            A_K_fem(i, j) +=
              (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
               fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
               fe_values.JxW(q_index));           // dx

      // 2) loop over patch pairs in cell
      for (unsigned int i = 0; i < patch_cell_indices[id].size(); ++i)
        for (unsigned int j = 0; j < patch_cell_indices[id].size(); ++j)
          {
            // a) perform matrix-matrix-matrix multiplication (basis change
            // fem->lod)
            A_K_lod = 0;
            A_K_lod.triple_product(A_K_fem,
                                   patch_cell_constraints[id][i],
                                   patch_cell_constraints[id][j],
                                   true,
                                   false);

            // b) write into global matrix
            affine_constraints.distribute_local_to_global(
              A_K_lod,
              cell_dof_indices[patch_cell_indices[id][i]],
              cell_dof_indices[patch_cell_indices[id][j]],
              A_lod);
          }
    }

  std::cout << A_lod.frobenius_norm() << std::endl;
}

int
main()
{
  test_normal();
  test_two_level();
}
