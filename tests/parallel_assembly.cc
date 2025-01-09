#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parsed_convergence_table.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/quadrature.h>
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

const unsigned int SPECIAL_NUMBER = 99;


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
  unsigned int                contained_patches = 0;
};



template <int dim, int spacedim>
class LOD
{
public:
  LOD();

  unsigned int oversampling         = 1;
  unsigned int n_subdivisions       = 2;
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

  MPI_Comm  mpi_communicator;

  void
  create_mesh_for_patch(Patch<dim> &current_patch);
  parallel::shared::Triangulation<dim> tria;
  // check ghost layer, needs to be set to whole domain
  // shared not distributed bc we want all processors to get access to all cells
  DoFHandler<dim> dof_handler_coarse;
  DoFHandler<dim> dof_handler_fine;

  LA::MPI::SparseMatrix basis_matrix_transposed;
  LA::MPI::SparseMatrix premultiplied_basis_matrix;
  LA::MPI::SparseMatrix global_stiffness_matrix;
  LA::MPI::Vector       solution;
  LA::MPI::Vector       system_rhs;
  
  std::unique_ptr<FiniteElement<dim>> fe_coarse;
  std::unique_ptr<FiniteElement<dim>> fe_fine;
  std::unique_ptr<Quadrature<dim>>    quadrature_fine;


  std::vector<Patch<dim>> patches;
  DynamicSparsityPattern  patches_pattern;
  DynamicSparsityPattern  patches_pattern_fine;

  IndexSet locally_owned_patches;
  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;
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

  locally_owned_dofs = dof_handler_coarse.locally_owned_dofs();
  locally_relevant_dofs =
    DoFTools::extract_locally_relevant_dofs(dof_handler_coarse);

  fe_fine =
    std::make_unique<FESystem<dim>>(FE_Q_iso_Q1<dim>(n_subdivisions),
                                    spacedim);
  dof_handler_fine.distribute_dofs(*fe_fine);
  quadrature_fine = std::make_unique<Quadrature<dim>>(
    QIterated<dim>(QGauss<1>(2), n_subdivisions));

  patches_pattern.reinit(dof_handler_coarse.n_dofs(),
                         dof_handler_coarse.n_dofs(),
                         locally_relevant_dofs);
  patches_pattern_fine.reinit(dof_handler_coarse.n_dofs(),
                              dof_handler_fine.n_dofs(),
                              locally_relevant_dofs);
  // MPI: instead of having every processor compute it we could just communicate
  // it
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

      for (int l_row = -oversampling;
           l_row <= static_cast<int>(oversampling);
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
            // patches_pattern.add(cell_index,
            // cell_to_add->active_cell_index());
            for (unsigned int d = 0; d < spacedim; ++d)
              patches_pattern.add_row_entries(coarse_dofs[d], coarse_dofs);
            auto cell_fine =
              cell_to_add->as_dof_handler_iterator(dof_handler_fine);
            cell_fine->get_dof_indices(fine_dofs);
            for (unsigned int d = 0; d < spacedim; ++d)
              {
                patches_pattern_fine.add_row_entries(coarse_dofs[d], fine_dofs);
                // patches_pattern_fine.add_row_entries(cell_index, fine_dofs);
              }
          }

        size_biggest_patch = std::max(size_biggest_patch, patch->cells.size());
        size_tiniest_patch = std::min(size_tiniest_patch, patch->cells.size());
      }
    }


  DynamicSparsityPattern global_sparsity_pattern;
  global_sparsity_pattern.compute_mmult_pattern(patches_pattern,
                                                patches_pattern);
  // global_stiffness_matrix.reinit(locally_owned_patches,
  //                                global_sparsity_pattern,
  //                                mpi_communicator);
  // TODO: fpr MPI FIX THIS
  global_stiffness_matrix.reinit(global_sparsity_pattern);

  solution.reinit(locally_owned_dofs, mpi_communicator);
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
    auto N_dofs_fine   = dh_fine_patch.n_dofs();
          Vector<double>          selected_basis_function(N_dofs_fine);
      for (unsigned int d = 0; d < spacedim; ++d)
        {
          current_patch->basis_function.push_back(selected_basis_function);
          current_patch->basis_function_premultiplied.push_back(selected_basis_function);
        }
      dh_fine_patch.clear();
    }
}

template <int dim, int spacedim>
void
LOD<dim, spacedim>::create_mesh_for_patch(Patch<dim> &current_patch)
{
  current_patch.sub_tria.clear();
  // copy manifolds
  // for (const auto i : tria.get_manifold_ids())
  //   if (i != numbers::flat_manifold_id)
  //     current_patch.sub_tria.set_manifold(i, tria.get_manifold(i));

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

      // new_cell.material_id = cell->material_id();
      // new_cell.manifold_id = cell->manifold_id();

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
            sub_cell->face(f)->set_boundary_id(SPECIAL_NUMBER);
        }


      // // lines // useless??
      // if constexpr (dim == 3)
      //   for (const auto l : cell->line_indices())
      //     {
      //       const auto line = cell->line(l);

      //       if (line->manifold_id() != numbers::flat_manifold_id)
      //         sub_cell->line(l)->set_manifold_id(line->manifold_id());
      //     }

      sub_cell++;
    }
}

template <int dim, int spacedim>
void
LOD<dim, spacedim>::assemble_global_matrix()
{
  DoFHandler<dim> dh_fine_current_patch;

  DynamicSparsityPattern identity(patches_pattern_fine.nonempty_rows());
  for (unsigned int i = 0; i < patches_pattern_fine.n_rows(); ++i)
    identity.add(i, i);
  DynamicSparsityPattern patches_pattern_fine_T;
  patches_pattern_fine_T.compute_Tmmult_pattern(patches_pattern_fine, identity);

  premultiplied_basis_matrix.reinit(patches_pattern_fine_T);
  basis_matrix_transposed.reinit(patches_pattern_fine_T);

  premultiplied_basis_matrix = 0.0;
  basis_matrix_transposed    = 0.0;

  system_rhs.reinit(patches_pattern_fine_T.nonempty_cols(), mpi_communicator);

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
              // basis_matrix.set(current_patch_id,
              //                  phi_loc.size(),
              //                  global_dofs.data(),
              //                  phi_loc.data());
              for (unsigned int idx = 0; idx < phi_loc.size(); ++idx)
                {
                  basis_matrix_transposed.set(global_dofs.data()[idx],
                                              spacedim * current_patch_id + d,
                                              phi_loc.data()[idx]);
                }

              iterator_to_cell_in_current_patch->get_dof_values(
                current_patch->basis_function_premultiplied[d], phi_loc);
              AssertDimension(global_dofs.size(), phi_loc.size());
              // premultiplied_basis_matrix.set(current_patch_id,
              //                                phi_loc.size(),
              //                                global_dofs.data(),
              //                                phi_loc.data());
              // if the matrix is already transposed we need to loop to add the
              // elements
              for (unsigned int idx = 0; idx < phi_loc.size(); ++idx)
                {
                  premultiplied_basis_matrix.set(global_dofs.data()[idx],
                                                 spacedim * current_patch_id +
                                                   d,
                                                 phi_loc.data()[idx]);
                }
            }
        }
    }
  // basis_matrix.compress(VectorOperation::insert);
  premultiplied_basis_matrix.compress(VectorOperation::insert);
  basis_matrix_transposed.compress(VectorOperation::insert);

  basis_matrix_transposed.Tmmult(global_stiffness_matrix,
                                 premultiplied_basis_matrix);
  global_stiffness_matrix.compress(VectorOperation::add);
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
  LOD<2,2> problem;
  problem.run();

}
