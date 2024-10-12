#include <deal.II/base/quadrature.h>
#include <deal.II/base/types.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_q_iso_q1.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>



using namespace dealii;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// utilities
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <int dim>
class MyPatch
{
public:
  // coarse cells that make up the patch
  std::vector<typename DoFHandler<dim>::active_cell_iterator> cells;
  Triangulation<dim>                                          sub_tria;
};


template <int dim>
const Table<2, bool>
create_bool_dof_mask_Q_iso_Q1(const FiniteElement<dim> &fe,
                              const Quadrature<dim> &   quadrature,
                              unsigned int              n_subdivisions)
{
  const auto compute_scalar_bool_dof_mask =
    [&quadrature](const auto &fe, const auto n_subdivisions) {
      Table<2, bool> bool_dof_mask(fe.dofs_per_cell, fe.dofs_per_cell);
      MappingQ1<dim> mapping;
      FEValues<dim>  fe_values(mapping, fe, quadrature, update_values);

      Triangulation<dim> tria;
      GridGenerator::hyper_cube(tria);

      fe_values.reinit(tria.begin());

      const auto lexicographic_to_hierarchic_numbering =
        FETools::lexicographic_to_hierarchic_numbering<dim>(n_subdivisions);

      for (unsigned int c_1 = 0; c_1 < n_subdivisions; ++c_1)
        for (unsigned int c_0 = 0; c_0 < n_subdivisions; ++c_0)

          for (unsigned int i_1 = 0; i_1 < 2; ++i_1)
            for (unsigned int i_0 = 0; i_0 < 2; ++i_0)
              {
                const unsigned int i =
                  lexicographic_to_hierarchic_numbering[(c_0 + i_0) +
                                                        (c_1 + i_1) *
                                                          (n_subdivisions + 1)];

                for (unsigned int j_1 = 0; j_1 < 2; ++j_1)
                  for (unsigned int j_0 = 0; j_0 < 2; ++j_0)
                    {
                      const unsigned int j =
                        lexicographic_to_hierarchic_numbering
                          [(c_0 + j_0) + (c_1 + j_1) * (n_subdivisions + 1)];

                      double sum = 0;

                      for (unsigned int q_1 = 0; q_1 < 2; ++q_1)
                        for (unsigned int q_0 = 0; q_0 < 2; ++q_0)
                          {
                            const unsigned int q_index =
                              (c_0 * 2 + q_0) +
                              (c_1 * 2 + q_1) * (2 * n_subdivisions);

                            sum += fe_values.shape_value(i, q_index) *
                                   fe_values.shape_value(j, q_index);
                          }
                      if (sum != 0)
                        bool_dof_mask(i, j) = true;
                    }
              }

      return bool_dof_mask;
    };

  Table<2, bool> bool_dof_mask(fe.dofs_per_cell, fe.dofs_per_cell);

  if (fe.n_components() == 1)
    {
      bool_dof_mask = compute_scalar_bool_dof_mask(fe, n_subdivisions);
    }
  else
    {
      const auto scalar_bool_dof_mask =
        compute_scalar_bool_dof_mask(fe.base_element(0), n_subdivisions);

      for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
        for (unsigned int j = 0; j < fe.n_dofs_per_cell(); ++j)
          if (scalar_bool_dof_mask[fe.system_to_component_index(i).second]
                                  [fe.system_to_component_index(j).second])
            bool_dof_mask[i][j] = true;
    }


  return bool_dof_mask;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// code
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


const unsigned int dim                  = 2;
const unsigned int spacedim             = 1;
unsigned int       oversampling         = 2;
unsigned int       n_global_refinements = 3;
unsigned int       n_subdivisions       = 64;

int
main()
{
  std::unique_ptr<FiniteElement<dim>> fe_coarse;
  std::unique_ptr<FiniteElement<dim>> fe_fine;
  std::unique_ptr<Quadrature<dim>>    quadrature_fine;

  std::vector<MyPatch<dim>> patches;

  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria);
  tria.refine_global(n_global_refinements);

  fe_coarse = std::make_unique<FESystem<dim>>(FE_DGQ<dim>(0), spacedim);
  fe_fine =
    std::make_unique<FESystem<dim>>(FE_Q_iso_Q1<dim>(n_subdivisions), spacedim);
  DoFHandler<dim> dof_handler_coarse(tria);
  dof_handler_coarse.distribute_dofs(*fe_coarse);
  quadrature_fine = std::make_unique<Quadrature<dim>>(
    QIterated<dim>(QGauss<1>(2), n_subdivisions));

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

      const unsigned int vector_cell_index =
        (int)floor(x / H) + N_cells_per_line * (int)floor(y / H);
      ordered_cells[vector_cell_index] = cell;

      std::vector<unsigned int> connected_indeces;
      connected_indeces.push_back(vector_cell_index);
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
      const auto vector_cell_index =
        (int)floor(cell->barycenter()(0) / H) +
        N_cells_per_line * (int)floor(cell->barycenter()(1) / H);
      {
        auto patch = &patches.emplace_back();


        for (auto neighbour_ordered_index : cells_in_patch[vector_cell_index])
          {
            auto &cell_to_add = ordered_cells[neighbour_ordered_index];
            patch->cells.push_back(cell_to_add);
          }
      }
    }

  for (unsigned int current_patch_id = 0; current_patch_id < patches.size();
       ++current_patch_id)
    {
      auto current_patch = &patches[current_patch_id];

      current_patch->sub_tria.clear();

      // copy manifolds
      for (const auto i : tria.get_manifold_ids())
        if (i != numbers::flat_manifold_id)
          current_patch->sub_tria.set_manifold(i, tria.get_manifold(i));

      // renumerate vertices
      std::vector<unsigned int> new_vertex_indices(tria.n_vertices(), 0);

      for (const auto &cell : current_patch->cells)
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

      for (const auto &cell : current_patch->cells)
        {
          CellData<dim> new_cell(cell->n_vertices());

          for (const auto v : cell->vertex_indices())
            new_cell.vertices[v] = new_vertex_indices[cell->vertex_index(v)];

          new_cell.material_id = cell->material_id();
          new_cell.manifold_id = cell->manifold_id();

          coarse_cells_of_patch.emplace_back(new_cell);
        }

      // create coarse mesh on the patch
      current_patch->sub_tria.create_triangulation(sub_points,
                                                   coarse_cells_of_patch,
                                                   {});

      auto sub_cell = current_patch->sub_tria.begin(0);
      for (const auto &cell : current_patch->cells)
        {
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


          sub_cell++;
        }
    }

  Table<2, bool> bool_dof_mask =
    create_bool_dof_mask_Q_iso_Q1(*fe_fine, *quadrature_fine, n_subdivisions);

  DoFHandler<dim> dh_coarse_patch;
  DoFHandler<dim> dh_fine_patch;

  using VectorType = Vector<double>;

  AffineConstraints<double> internal_boundary_constraints;

  for (unsigned int current_patch_id = 0; current_patch_id < patches.size();
       ++current_patch_id)
    {
      auto current_patch = &patches[current_patch_id];

      dh_fine_patch.reinit(current_patch->sub_tria);
      dh_fine_patch.distribute_dofs(*fe_fine);

      internal_boundary_constraints.clear();
      // apply constraints: not needed now as we keep the dfos anyway
      internal_boundary_constraints.close();
      SparsityPattern patch_sparsity_pattern(dh_fine_patch.n_dofs(),
                                             dh_fine_patch.n_dofs());

      std::vector<types::global_dof_index> dofs_on_this_cell(
        fe_fine->n_dofs_per_cell());

      for (const auto &cell : dh_fine_patch.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(dofs_on_this_cell);

            internal_boundary_constraints.add_entries_local_to_global(
              dofs_on_this_cell,
              patch_sparsity_pattern,
              true,
              bool_dof_mask); // keep constrained entries must be true
          }

      patch_sparsity_pattern.compress();
    }
}