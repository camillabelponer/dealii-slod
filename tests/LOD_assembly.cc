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

#include "util.h"


using namespace dealii;


const unsigned int dim      = 2;
const unsigned int spacedim = 1;

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


int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  unsigned int                     oversampling         = 1;
  unsigned int                     n_subdivisions       = 5;
  unsigned int                     n_global_refinements = 2;

  MPI_Comm mpi_communicator(MPI_COMM_WORLD);


  std::unique_ptr<FiniteElement<dim>> fe_coarse;
  std::unique_ptr<FiniteElement<dim>> fe_fine;
  std::unique_ptr<Quadrature<dim>>    quadrature_fine;

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


  fe_fine =
    std::make_unique<FESystem<dim>>(FE_Q_iso_Q1<dim>(n_subdivisions), spacedim);
  DoFHandler<dim> dof_handler_fine(tria);
  dof_handler_fine.distribute_dofs(*fe_fine);
  quadrature_fine = std::make_unique<Quadrature<dim>>(
    QIterated<dim>(QGauss<1>(2), n_subdivisions));


  patches_pattern.reinit(dof_handler_coarse.n_dofs(),
                         dof_handler_coarse.n_dofs(),
                         locally_relevant_dofs);

  // create_patches();

  std::vector<unsigned int> fine_dofs(fe_fine->n_dofs_per_cell());

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



  for (auto &current_patch : patches)
    {
      current_patch.sub_tria.clear();

      // copy manifolds
      for (const auto i : tria.get_manifold_ids())
        if (i != numbers::flat_manifold_id)
          current_patch.sub_tria.set_manifold(i, tria.get_manifold(i));

      // renumerate vertices
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

          new_cell.material_id = cell->material_id();
          new_cell.manifold_id = cell->manifold_id();

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
                sub_cell->face(f)->set_boundary_id(0);
            }


          // lines // useless??
          if constexpr (dim == 3)
            for (const auto l : cell->line_indices())
              {
                const auto line = cell->line(l);

                if (line->manifold_id() != numbers::flat_manifold_id)
                  sub_cell->line(l)->set_manifold_id(line->manifold_id());
              }

          sub_cell++;
        }
    }


  // solve fem
  const auto &dh = dof_handler_fine;

  auto     fem_locally_owned_dofs = dh.locally_owned_dofs();
  IndexSet fem_locally_relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(dh, fem_locally_relevant_dofs);
  FEValues<dim> fe_values(*fe_fine,
                          *quadrature_fine,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  // create sparsity pattern fr global fine matrix
  AffineConstraints<double> fem_constraints(fem_locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints(dh, fem_constraints);
  VectorTools::interpolate_boundary_values(
    dh, 0, Functions::ZeroFunction<dim, double>(spacedim), fem_constraints);
  fem_constraints.close();
  DynamicSparsityPattern fem_sparsity_pattern(fem_locally_relevant_dofs);
  DoFTools::make_sparsity_pattern(dh,
                                  fem_sparsity_pattern,
                                  fem_constraints,
                                  false);
  SparsityTools::distribute_sparsity_pattern(fem_sparsity_pattern,
                                             fem_locally_owned_dofs,
                                             mpi_communicator,
                                             fem_locally_relevant_dofs);

  LinearAlgebraTrilinos::MPI::SparseMatrix fem_stiffness_matrix;

  fem_stiffness_matrix.reinit(fem_locally_owned_dofs,
                              fem_locally_owned_dofs,
                              fem_sparsity_pattern,
                              mpi_communicator);

  MappingQ1<dim> mapping;

  MatrixCreator::create_laplace_matrix<dim, dim>(mapping,
                                                 dh,
                                                 *quadrature_fine,
                                                 fem_stiffness_matrix,
                                                 nullptr,
                                                 fem_constraints);


  //   compute_basis_function_candidates();

  DoFHandler<dim> dh_coarse_patch;
  DoFHandler<dim> dh_fine_patch;

  using VectorType = Vector<double>;

  // need reinit in loop
  //  LinearAlgebraTrilinos::MPI::SparseMatrix
  TrilinosWrappers::SparseMatrix patch_stiffness_matrix;
  AffineConstraints<double>      internal_boundary_constraints;
  // AffineConstraints<double> local_stiffnes_constraints;

  // TODO: use internal and local constraints to take care fo the boundary of
  // the patch that's not on the boundary of the domain now special number is
  // set to zero so they are treated as one together

  for (auto current_patch_id : locally_owned_patches)
    {
      AssertIndexRange(current_patch_id, patches.size());
      auto current_patch = &patches[current_patch_id];

      // create_mesh_for_patch(*current_patch);
      dh_fine_patch.reinit(current_patch->sub_tria);
      dh_fine_patch.distribute_dofs(*fe_fine);
      // DoFRenumbering::cell_wise(dh_fine_patch, cell_order);

      dh_coarse_patch.reinit(current_patch->sub_tria);
      dh_coarse_patch.distribute_dofs(*fe_coarse);
      // DoFRenumbering::cell_wise(dh_fine_patch, cell_order);

      auto Ndofs_fine = dh_fine_patch.n_dofs();


      internal_boundary_constraints.clear();
      DoFTools::make_zero_boundary_constraints(dh_fine_patch,
                                               0,
                                               internal_boundary_constraints);
      internal_boundary_constraints.close();

      TrilinosWrappers::SparsityPattern sparsity_pattern(Ndofs_fine,
                                                         Ndofs_fine);

      Table<2, bool> bool_dof_mask =
        create_bool_dof_mask(*fe_fine, *quadrature_fine);

      std::vector<types::global_dof_index> dofs_on_this_cell;

      for (const auto &cell : dh_fine_patch.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            const unsigned int dofs_per_cell = cell->get_fe().n_dofs_per_cell();
            dofs_on_this_cell.resize(dofs_per_cell);
            cell->get_dof_indices(dofs_on_this_cell);

            internal_boundary_constraints.add_entries_local_to_global(
              dofs_on_this_cell, sparsity_pattern, true, bool_dof_mask);
          }

      sparsity_pattern.compress();
      patch_stiffness_matrix.clear();
      patch_stiffness_matrix.reinit(sparsity_pattern);

      MappingQ1<dim> mapping;

      MatrixCreator::create_laplace_matrix<dim, dim>(
        mapping,
        dh_fine_patch,
        *quadrature_fine,
        patch_stiffness_matrix,
        nullptr,
        internal_boundary_constraints);


      IndexSet                  set(dof_handler_fine.n_dofs());
      std::vector<unsigned int> fine_dofs(fe_fine->n_dofs_per_cell());

      for (const auto &cell : current_patch->cells)
        {
          auto cell_fine = cell->as_dof_handler_iterator(dof_handler_fine);
          cell_fine->get_dof_indices(fine_dofs);
          set.add_indices(fine_dofs.begin(), fine_dofs.end());
        }
      auto row_index_set = set.get_index_vector();
      Assert(Ndofs_fine == row_index_set.size(), ExcNotImplemented());

      FullMatrix<double> A_fem_restricted(Ndofs_fine, Ndofs_fine);

      A_fem_restricted.extract_submatrix_from(fem_stiffness_matrix,
                                              row_index_set,
                                              row_index_set);

      TrilinosWrappers::SparseMatrix patch_stiffness_matrix_from_Afem(
        sparsity_pattern);

      A_fem_restricted.scatter_matrix_to(row_index_set,
                                         row_index_set,
                                         patch_stiffness_matrix_from_Afem);

      std::cout << patch_stiffness_matrix_from_Afem.diag_element(0)
                << std::endl;
      //     if (patch_stiffness_matrix_from_Afem.l1_norm() !=
      //     patch_stiffness_matrix.l1_norm()) std::cout << "no" << std::endl;
    }
}
