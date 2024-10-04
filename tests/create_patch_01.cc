#include <deal.II/base/quadrature.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/types.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>



using namespace dealii;


const unsigned int dim      = 2;
const unsigned int spacedim = 1;
unsigned int                     oversampling         = 4;
unsigned int                     n_global_refinements = 5;

template <int dim>
class MyPatch
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
  unsigned int                patch_id;
};

unsigned int
coordinates_to_index(double x, double y)
{
  double H = pow(0.5, n_global_refinements);
  unsigned int N_cells_per_line = (int)1/H;


      const unsigned int x_i = (int)floor(x/H);
      const unsigned int y_i = (int)floor(y/H);

  return x_i + N_cells_per_line*y_i;

}


int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  MPI_Comm mpi_communicator(MPI_COMM_WORLD);


  std::unique_ptr<FiniteElement<dim>> fe_coarse;

  std::vector<MyPatch<dim>> patches;
  DynamicSparsityPattern    patches_pattern;

  IndexSet locally_owned_patches;
  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria);
  tria.refine_global(n_global_refinements);

  locally_owned_patches =
    Utilities::MPI::create_evenly_distributed_partitioning(
      mpi_communicator, tria.n_global_active_cells());

  fe_coarse = std::make_unique<FESystem<dim>>(FE_DGQ<dim>(0), spacedim);
  DoFHandler<dim> dof_handler_coarse(tria);
  dof_handler_coarse.distribute_dofs(*fe_coarse);

  locally_relevant_dofs =
    DoFTools::extract_locally_relevant_dofs(dof_handler_coarse);

  patches_pattern.reinit(dof_handler_coarse.n_dofs(),
                         dof_handler_coarse.n_dofs(),
                         locally_relevant_dofs);

  // create_patches();

  if (false) // 71.54s for ref 5 oversampling 4
  {
      // Queue for patches for which neighbours should be added

      std::vector<typename DoFHandler<dim>::active_cell_iterator> patch_iterators;
  for (const auto &cell : dof_handler_coarse.active_cell_iterators())
    {
      auto cell_index = cell->active_cell_index();
      {
        // for each cell we create its patch and add it to the global vector
        // of patches
        auto patch = &patches.emplace_back();
        patch_iterators.clear();
        patch_iterators.push_back(cell);

        // The iterators for level l are in the range [l_start, l_end) of
        // patch_iterators
        unsigned int l_start = 0;
        unsigned int l_end   = 1;
        patch->cells.push_back(cell);
        // patch->cell_indices.set_size(tria.n_active_cells());
        patches_pattern.add(cell_index, cell_index);
        for (unsigned int l = 1; l <= oversampling; l++)
          {
            for (unsigned int i = l_start; i < l_end; i++)
              {
                AssertIndexRange(i, patch_iterators.size());
                for (auto ver : patch_iterators[i]->vertex_indices())
                  {
                    auto vertex = patch_iterators[i]->vertex_index(ver);
                    for (const auto &neighbour :
                         GridTools::find_cells_adjacent_to_vertex(
                           dof_handler_coarse, vertex))
                      {
                        if (!patches_pattern.exists(
                              cell_index, neighbour->active_cell_index()))
                          {
                            patch_iterators.push_back(neighbour);
                            patches_pattern.add(cell_index,
                                                neighbour->active_cell_index());
                            patch->cells.push_back(neighbour);
                          }
                      }
                  }
              }
            l_start = l_end;
            l_end   = patch_iterators.size();
          }
      }
    }
  }
  else // 10.97s for ref 5 oversampling 4
  {
    // looping over all the cells once and storing them ordered

    double H = pow(0.5, n_global_refinements);
    unsigned int N_cells_per_line = (int)1/H;
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
      const unsigned int vector_cell_index = (int)floor(x/H) + N_cells_per_line*(int)floor(y/H);

      //const unsigned int vector_cell_index = coordinates_to_index(x, y);

      // std::cout << cell->barycenter() //<< " " << (int)floor(x/H) << " " << (int)floor(y/H) << " resulting index " << cell_index << std::endl;
      ordered_cells[vector_cell_index] = cell;

      std::vector<unsigned int> connected_indeces;
      // connected_indeces.push_back(vector_cell_index);

      for(int l_row = -oversampling; l_row <= static_cast<int>(oversampling); ++l_row)
      {
        double x_j = x + l_row * H;
        if (x_j > 0 && x_j < 1) // domain borders
        {
          for(int l_col = -oversampling; l_col <= static_cast<int>(oversampling); ++l_col)
          {
            const double y_j = y + l_col * H;
            if (y_j > 0 && y_j < 1)
            {
              const unsigned int vector_cell_index_j = (int)floor(x_j/H) + N_cells_per_line*(int)floor(y_j/H);
              connected_indeces.push_back(vector_cell_index_j);
            }

          }
        }
      }

      cells_in_patch[vector_cell_index] = connected_indeces;
    }

    // now looping and creating the patches
    for (const auto &cell : dof_handler_coarse.active_cell_iterators())
    {
      const auto vector_cell_index = coordinates_to_index(cell->barycenter()(0), cell->barycenter()(1));
      auto cell_index = cell->active_cell_index();
      {

        auto patch = &patches.emplace_back();

        patch->cells.push_back(cell);
        patches_pattern.add(cell_index, cell_index);
        for (auto neighbour_ordered_index : cells_in_patch[vector_cell_index])
          {
            patches_pattern.add(cell_index, ordered_cells[neighbour_ordered_index]->active_cell_index());
          }
      }
    }

  }

    {
      std::cout << "printing the sparsity pattern: [global_cell_id] = {cells}"
           << std::endl;
      // for (unsigned int cell = 0; cell < tria.n_active_cells(); ++cell)
      for (const auto &cell_it : tria.active_cell_iterators())
        {
          auto cell = cell_it->active_cell_index();
          std::cout << "- cell " << cell << " (baricenter " <<
          cell_it->barycenter()
               << ") is connected to patches/cells: {";
          for (unsigned int j = 0; j < patches_pattern.row_length(cell); j++)
            {
              std::cout << patches_pattern.column_number(cell, j) << " ";
            }
          std::cout << "}" << std::endl;
        }
    }


}
