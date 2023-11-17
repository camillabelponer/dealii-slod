#include <deal.II/grid/grid_tools.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/types.h>

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
  std::vector<unsigned int> dofs;
  for (unsigned int i = 0; i < patches.size(); i++) {
    global_boundary_dofs.clear();
    internal_boundary_dofs.clear();
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
