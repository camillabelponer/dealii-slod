#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/generic_linear_algebra.h>

#include <deal.II/multigrid/mg_transfer_global_coarsening.h>


namespace LA
{
  using namespace dealii::LinearAlgebraTrilinos;
} // namespace LA

using namespace dealii;

template <int dim>
class Patch
{
public:
  // coarse cells that make up the patch
  std::vector<typename DoFHandler<dim>::active_cell_iterator> cells;
  Triangulation<dim>                                          sub_tria;

  std::vector<Vector<double>> basis_function;
  std::vector<Vector<double>> basis_function_premultiplied;
};



template <int dim, int spacedim>
class LOD
{
public:
  LOD();

  unsigned int oversampling         = 1;
  unsigned int n_subdivisions       = 1;
  unsigned int n_global_refinements = 2;

  void
  run();

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
  initialize_patches();

  MPI_Comm mpi_communicator;

  void
  create_mesh_for_patch(Patch<dim> &current_patch);
  parallel::shared::Triangulation<dim> tria;
  // check ghost layer, needs to be set to whole domain
  // shared not distributed bc we want all processors to get access to all cells
  DoFHandler<dim> dof_handler_coarse;
  DoFHandler<dim> dof_handler_fine;

  LA::MPI::SparseMatrix basis_matrix_T;
  LA::MPI::SparseMatrix premultiplied_basis_matrix_T;
  LA::MPI::SparseMatrix global_stiffness_matrix;
  LA::MPI::Vector       solution;
  LA::MPI::Vector       system_rhs;

  std::unique_ptr<FiniteElement<dim>> fe_coarse;
  std::unique_ptr<FiniteElement<dim>> fe_fine;


  std::vector<Patch<dim>> patches;
  DynamicSparsityPattern  coarse_to_coarse_pattern;
  DynamicSparsityPattern  coarse_to_fine_pattern;
  DynamicSparsityPattern  fine_to_coarse_pattern;

  IndexSet locally_owned_patches;
  IndexSet locally_owned_coarse_dofs;
  IndexSet locally_relevant_coarse_dofs;
  IndexSet locally_owned_fine_dofs;
  IndexSet locally_relevant_fine_dofs;
};

template <int dim, int spacedim>
LOD<dim, spacedim>::LOD()
  : mpi_communicator(MPI_COMM_WORLD)
  , tria(mpi_communicator)
  , dof_handler_coarse(tria)
  , dof_handler_fine(tria)
{}

template <int dim, int spacedim>
void
LOD<dim, spacedim>::make_fe()
{
  fe_coarse = std::make_unique<FESystem<dim>>(FE_DGQ<dim>(0), spacedim);
  dof_handler_coarse.distribute_dofs(*fe_coarse);

  locally_owned_coarse_dofs = dof_handler_coarse.locally_owned_dofs();
  locally_relevant_coarse_dofs =
    DoFTools::extract_locally_relevant_dofs(dof_handler_coarse);

  fe_fine =
    std::make_unique<FESystem<dim>>(FE_Q_iso_Q1<dim>(n_subdivisions), spacedim);
  dof_handler_fine.distribute_dofs(*fe_fine);
  locally_owned_fine_dofs = dof_handler_fine.locally_owned_dofs();
  locally_relevant_fine_dofs =
    DoFTools::extract_locally_relevant_dofs(dof_handler_fine);

  coarse_to_coarse_pattern.reinit(dof_handler_coarse.n_dofs(),
                                  dof_handler_coarse.n_dofs(),
                                  locally_relevant_coarse_dofs);
  coarse_to_fine_pattern.reinit(dof_handler_coarse.n_dofs(),
                                dof_handler_fine.n_dofs(),
                                locally_relevant_coarse_dofs);
  fine_to_coarse_pattern.reinit(dof_handler_fine.n_dofs(),
                                dof_handler_fine.n_dofs(),
                                locally_relevant_fine_dofs);
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

      for (int l_row = -oversampling; l_row <= static_cast<int>(oversampling);
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
            for (unsigned int d = 0; d < spacedim; ++d)
              coarse_to_coarse_pattern.add_row_entries(coarse_dofs[d],
                                                       coarse_dofs);

            auto cell_fine =
              cell_to_add->as_dof_handler_iterator(dof_handler_fine);
            cell_fine->get_dof_indices(fine_dofs);
            for (unsigned int d = 0; d < spacedim; ++d)
              {
                coarse_to_fine_pattern.add_row_entries(coarse_dofs[d],
                                                       fine_dofs);
                for (auto i : fine_dofs)
                  fine_to_coarse_pattern.add(i, coarse_dofs[d]);
              }
          }

        size_biggest_patch = std::max(size_biggest_patch, patch->cells.size());
        size_tiniest_patch = std::min(size_tiniest_patch, patch->cells.size());
      }
    }
  coarse_to_coarse_pattern.symmetrize();
}



template <int dim, int spacedim>
void
LOD<dim, spacedim>::compute_basis_function_candidates()
{
  DoFHandler<dim> dh_fine_patch;
  for (auto current_patch_id : locally_owned_patches)
    {
      AssertIndexRange(current_patch_id, patches.size());
      auto current_patch = &patches[current_patch_id];
      dh_fine_patch.reinit(current_patch->sub_tria);
      dh_fine_patch.distribute_dofs(*fe_fine);
      auto           N_dofs_fine = dh_fine_patch.n_dofs();
      Vector<double> selected_basis_function(N_dofs_fine);
      selected_basis_function.add(1);
      for (unsigned int d = 0; d < spacedim; ++d)
        {
          current_patch->basis_function.push_back(selected_basis_function);
          current_patch->basis_function_premultiplied.push_back(
            selected_basis_function);
        }
      dh_fine_patch.clear();
    }
}

template <int dim, int spacedim>
void
LOD<dim, spacedim>::create_mesh_for_patch(Patch<dim> &current_patch)
{
  current_patch.sub_tria.clear();

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
      coarse_cells_of_patch.emplace_back(new_cell);
    }

  // create coarse mesh on the patch
  current_patch.sub_tria.create_triangulation(sub_points,
                                              coarse_cells_of_patch,
                                              {});

}

template <int dim, int spacedim>
void
LOD<dim, spacedim>::assemble_global_matrix()
{
  DoFHandler<dim> dh_fine_current_patch;


  // global_stiffness_matrix.reinit(locally_owned_patches,
  //  coarse_to_coarse_pattern,
  //  mpi_communicator);
  // TODO: fpr MPI FIX THIS
  global_stiffness_matrix.reinit(coarse_to_coarse_pattern);
  
  solution.reinit(locally_owned_coarse_dofs, mpi_communicator);
  
  LA::MPI::SparseMatrix basis_matrix;
  LA::MPI::SparseMatrix premultiplied_basis_matrix;

  premultiplied_basis_matrix.reinit(locally_owned_coarse_dofs,
                                    locally_relevant_fine_dofs,
                                    coarse_to_fine_pattern);
  basis_matrix.reinit(locally_owned_coarse_dofs,
                      locally_relevant_fine_dofs,
                      coarse_to_fine_pattern);

  premultiplied_basis_matrix = 0.0;
  basis_matrix               = 0.0;


  premultiplied_basis_matrix_T.reinit(locally_owned_fine_dofs,
                                      locally_relevant_fine_dofs,
                                      fine_to_coarse_pattern);
  basis_matrix_T.reinit(locally_owned_fine_dofs,
                        locally_relevant_fine_dofs,
                        fine_to_coarse_pattern);

  premultiplied_basis_matrix_T = 0.0;
  basis_matrix_T               = 0.0;

  system_rhs.reinit(fine_to_coarse_pattern.nonempty_cols(), mpi_communicator);

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

              for (unsigned int idx = 0; idx < phi_loc.size(); ++idx)
                {
                  basis_matrix.set(spacedim * current_patch_id + d,
                                   global_dofs.data()[idx],
                                   phi_loc.data()[idx]);
                  basis_matrix_T.set(global_dofs.data()[idx],
                                     spacedim * current_patch_id + d,
                                     phi_loc.data()[idx]);
                }

              iterator_to_cell_in_current_patch->get_dof_values(
                current_patch->basis_function_premultiplied[d], phi_loc);
              AssertDimension(global_dofs.size(), phi_loc.size());

              for (unsigned int idx = 0; idx < phi_loc.size(); ++idx)
                {
                  premultiplied_basis_matrix.set(spacedim * current_patch_id +
                                                   d,
                                                 global_dofs.data()[idx],
                                                 phi_loc.data()[idx]);
                  premultiplied_basis_matrix_T.set(global_dofs.data()[idx],
                                                   spacedim * current_patch_id +
                                                     d,
                                                   phi_loc.data()[idx]);
                }
            }
        }
    }
  // basis_matrix.compress(VectorOperation::insert);
  premultiplied_basis_matrix_T.compress(VectorOperation::insert);
  basis_matrix_T.compress(VectorOperation::insert);

  premultiplied_basis_matrix.compress(VectorOperation::insert);
  basis_matrix.compress(VectorOperation::insert);

  // opz 1
  // basis_matrix_T.Tmmult(global_stiffness_matrix,
  //                                premultiplied_basis_matrix_T);
  // opz 2
  basis_matrix.mmult(global_stiffness_matrix, premultiplied_basis_matrix_T);
  // opz 3: does not work
  // premultiplied_basis_matrix.transpose();
  // basis_matrix.mmult(global_stiffness_matrix,
  //                                 premultiplied_basis_matrix);

  global_stiffness_matrix.compress(VectorOperation::add);

  global_stiffness_matrix.print(std::cout);
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

template <int dim, int spacedim>
void
LOD<dim, spacedim>::run()
{
  make_grid();
  make_fe();
  initialize_patches();

  compute_basis_function_candidates();
  assemble_global_matrix();
}


int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  LOD<2, 2>                        problem;
  problem.run();
}
