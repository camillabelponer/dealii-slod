#include <deal.II/grid/grid_tools.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/types.h>
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

#include <slod.h>

template<int dim>
SLOD<dim>::SLOD(Triangulation<dim> tria)
  : tria(tria)
  , dof_handler(tria)
{}

template<int dim>
void SLOD<dim>::make_fe() 
{
  fe_coarse = std::make_unique(FE_DGQ<dim>(0));
  fe_fine = std::make_unique(FE_Q_iso_Q1<dim>(n_subdivisions));
  dof_handler.distribute_dofs(fe_coarse);
  // TODO: 2?
  quadrature_fine = QIterated<dim>(QGauss<1>(2), n_subdivisions);
}

template<int dim>
void SLOD<dim>::create_patches() 
{
  // Queue for patches for which neighbours should be added
  std::vector<typename DoFHandler<dim>::active_cell_iterator> patch_iterators;
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      std::vector<types::global_dof_index> dof_indices(
        fe_coarse.n_dofs_per_cell());
      cell->get_dof_indices(dof_indices);
      cell_dof_indices[cell->active_cell_index()] = dof_indices;

      bool crosses_border = false;
      Patch patch;
      patch_iterators.clear();
      patch_iterators.push_back(cell);
      // The iterators for level l are in the range [l_start, l_end) of patch_iterators
      unsigned int l_start = 0;
      unsigned int l_end = 1;
      patch.cells.add_index(cell->active_cell_index());
      for (unsigned int l = 1; l <= oversampling; l++) {
        for (unsigned int i = l_start; i <= l_end; i++) {
          if (patch_iterators[i].at_boundary()) {
            crosses_border = true;
          }
          for (auto vertex : patch_iterators[i].vertex_iterator()) {
            for (const auto &neighbour : GridTools::find_cells_adjacent_to_vertex(tria, vertex)) {
              if (!patch.cells.is_element(neighbour->active_cell_index())) {
                patch_iterators.push_back(neighbour);
              }
              patch.cells.add_index(neighbour->active_cell_index());
            }
          }
        }
        l_start = l_end;
        l_end = patch_iterators.size();
      }
      if (crosses_border) {
        patch.num_basis_vectors = 0;
      } else {
        patch.num_basis_vectors = 1;
      }
      patches[cell->active_cell_index()] = patch;
    }

  // For patches at the border, find the neighbouring patch that is not at the border and which completely contains this patch
  for (auto &patch : patches) {
    if (patch.num_basis_vectors == 0) {
      bool success = false;
      for (auto i : patch.cells)
      {
        Patch neighbour = patches[i];
        if (neighbour.num_basis_vectors == 0) continue;
        // Check that the patch around i contains all of the cells
        // of the patch specified by pair
        bool contains_patch = true;
        for (auto j : patch.cells) {
          if (!neighbour.cells.is_element(j)) {
            contains_patch = false;
            break;
          }
        }
        if (contains_patch) {
          neighbour.num_basis_vectors++;
          success = true;
          break;
        }
      }
      Assert(success, ExcNotImplemented());
    }
  }
}

const unsigned int SPECIAL_NUMBER = 69;

template<int dim>
void SLOD<dim>::compute_coarse_basis()
{
  Triangulation<dim> sub_tria;
  DoFHandler<dim> dh_fine(sub_tria);
  IndexSet global_boundary_dofs;
  IndexSet internal_boundary_dofs;
  IndexSet internal_boundary_dofs_comp;
  std::vector<unsigned int> dofs;
  FullMatrix<double> patch_stiffness_matrix;
  FullMatrix<double> patch_stiffness_matrix_a0;
  FullMatrix<double> patch_stiffness_matrix_a1;
  for (unsigned int i = 0; i < patches.size(); i++) {
    global_boundary_dofs.clear();
    internal_boundary_dofs.clear();
    internal_boundary_dofs_comp.clear();
    create_mesh_for_patch(i, &sub_tria);
    dh_fine.reinit(sub_tria);
    dh_fine.distribute_dofs(*fe_fine);

    for (auto cell : dh_fine.active_cell_iterators()) {
      for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; f++) {
        unsigned int boundary_id = cell->face(f)->boundary_id();
        if (boundary_id != numbers::internal_face_boundary_id) {
          cell->face(f)->get_dof_indices(dofs);
          if (boundary_id == SPECIAL_NUMBER) {
            internal_boundary_dofs.add_indices(dofs.begin(), dofs.end());
          } else {
            global_boundary_dofs.add_indices(dofs.begin(), dofs.end());
          }
        }
      }
    }
    internal_boundary_dofs.subtract_set(global_boundary_dofs);
    internal_boundary_dofs_comp.add_range(0, dh_fine.n_dofs());
    internal_boundary_dofs_comp.subtract_set(internal_boundary_dofs);

    // TODO: Can fe_fine be used for different patches at the same time?
    assemble_stiffness(dh_fine, patch_stiffness_matrix);

    // TODO: This is f**in inefficient
    auto index_vector_0 = internal_boundary_dofs_comp.get_index_vector();
    auto index_vector_1 = internal_boundary_dofs.get_index_vector();
    patch_stiffness_matrix_a0.extract_submatrix_from(&patch_stiffness_matrix, index_vector_0, index_vector_0);
    patch_stiffness_matrix_a1.extract_submatrix_from(&patch_stiffness_matrix, index_vector_1, index_vector_0);

    // Invert A0
    patch_stiffness_matrix_a0.gauss_jordan();

    patch_stiffness_matrix_a1.mmult(patch_stiffness_matrix_a0, patch_stiffness_matrix_a1);
  }
}

template <int dim>
void
SLOD<dim>::create_mesh_for_patch(unsigned int patch_id, Triangulation<dim> &sub_tria)
{
  sub_tria.clear();

  // copy manifolds
  for (const auto i : tria.get_manifold_ids())
    if (i != numbers::flat_manifold_id)
      sub_tria.set_manifold(i, tria.get_manifold(i));

  // renumerate vertices
  std::vector<unsigned int> new_vertex_indices(tria.n_vertices(), numbers::invalid_unsigned_int);

  unsigned int c = 0;
  for (const auto &index : patches[patch_id].cells) {
    auto cell = tria.create_cell_iterator(index);
    if (cell != tria.end())
      for (const unsigned int v : cell->vertex_indices())
        new_vertex_indices[cell->vertex_index(v)] = c++;
  }

  // collect points
  std::vector<Point<dim>> sub_points;
  for (unsigned int i = 0; i < new_vertex_indices.size(); ++i)
    if (new_vertex_indices[i] != numbers::invalid_unsigned_int)
      sub_points.emplace_back(tria.get_vertices()[i]);

  // create new cell and subcell data
  std::vector<CellData<dim>> sub_cells;

  for (const auto &index : patches[patch_id].cells) {
    auto cell = tria.create_cell_iterator(index);
    if (cell != tria.end())
      {
        // cell
        CellData<dim> new_cell(cell->n_vertices());

        for (const auto v : cell->vertex_indices())
          new_cell.vertices[v] = new_vertex_indices[cell->vertex_index(v)];

        new_cell.material_id = cell->material_id();
        new_cell.manifold_id = cell->manifold_id();

        sub_cells.emplace_back(new_cell);
      }
  }

  // create mesh
  sub_tria.create_triangulation(sub_points, sub_cells, {});

  auto sub_cell = sub_tria.begin();

  for (const auto &index : patches[patch_id].cells) {
    auto cell = tria.create_cell_iterator(index);
    if (cell != tria.end())
      {
        // faces
        for (const auto f : cell->face_indices())
          {
            const auto face = cell->face(f);

            if (face->manifold_id() != numbers::flat_manifold_id)
              sub_cell->face(f)->set_manifold_id(face->manifold_id());

            if (face->boundary_id() != numbers::internal_face_boundary_id)
              sub_cell->face(f)->set_boundary_id(face->boundary_id());
            else if (sub_cell->face(f)->boundary_id() != numbers::internal_face_boundary_id)
              sub_cell->face(f)->set_boundary_id(SPECIAL_NUMBER);
          }

        // lines
        if (dim == 3)
          for (const auto l : cell->line_indices())
            {
              const auto line = cell->line(l);

              if (line->manifold_id() != numbers::flat_manifold_id)
                sub_cell->line(l)->set_manifold_id(line->manifold_id());
            }

        sub_cell++;
      }
  }
}

template <int dim>
void
SLOD<dim>::assemble_stiffness(DoFHandler<dim> &dh, FullMatrix<double> &stiffness_matrix)
{
  // TODO: avoid reallocations
  // TODO
  // TimerOutput::Scope t(computing_timer, "Assemble Stiffness and Neumann rhs");
  FEValues<dim> fe_values(*fe_fine,
                               *quadrature_fine,
                               update_values | update_gradients |
                                 update_quadrature_points | update_JxW_values);

  const unsigned int          dofs_per_cell = fe_fine->n_dofs_per_cell();
  const unsigned int          n_q_points    = quadrature_fine->size();
  FullMatrix<double>          cell_matrix(dofs_per_cell, dofs_per_cell);
  std::vector<Tensor<2, dim>>     grad_phi_u(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  AffineConstraints<double> constraints;

  // TODO: Parameter adaptable parameters
  // DoFTools::make_hanging_node_constraints(dh, constraints);
  VectorTools::interpolate_boundary_values(dh, 0, 0, constraints);

  for (const auto &cell : dh.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        cell_matrix = 0;
        fe_values.reinit(cell);
        // par.rhs.vector_value_list(fe_values.get_quadrature_points(),
        //                           rhs_values);
        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            for (unsigned int k = 0; k < dofs_per_cell; ++k)
              {
                grad_phi_u[k] =
                  fe_values.gradient(k, q);
              }
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  {
                    cell_matrix(i, j) +=
                         scalar_product(grad_phi_u[i], grad_phi_u[j]) *
                         fe_values.JxW(q);
                  }
              }
          }


        cell->get_dof_indices(local_dof_indices);
        // TODO: Handle inhomogeneous dirichlet bc
        constraints.distribute_local_to_global(cell_matrix,
                                               local_dof_indices,
                                               stiffness_matrix);
      }
}
