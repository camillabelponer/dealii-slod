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

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>



using namespace dealii;

const unsigned int SPECIAL_NUMBER = 0;


const unsigned int dim                  = 2;
const unsigned int spacedim             = 1;
unsigned int       oversampling         = 4;
unsigned int       n_global_refinements = 5;
unsigned int       n_subdivisions       = 10;

template <int dim>
class MyPatch
{
public:
  // coarse cells that make up the patch
  std::vector<typename DoFHandler<dim>::active_cell_iterator> cells;
  Triangulation<dim>                                          sub_tria;

  std::vector<Vector<double>> basis_function;
  std::vector<Vector<double>> basis_function_premultiplied;
  unsigned int                contained_patches = 0;
  unsigned int                patch_id;
};


/*
template <int dim>
void
test()
{
  Triangulation<dim> square;

  GridGenerator::hyper_cube(square);
  square.refine_global(n_global_refinements);

  FE_DGQ<dim>      fe1(0);
  FE_Q_iso_Q1<dim> fe2(n_subdivisions);

  DoFHandler<dim> dof_handler_1(square);
  DoFHandler<dim> dof_handler_2(square);

  dof_handler_1.distribute_dofs(fe1);
  dof_handler_2.distribute_dofs(fe2);

  std::cout << "number active cells: " << square.n_active_cells() << std::endl;

  std::cout << "P0 number dofs per cell: " << fe1.n_dofs_per_cell()
            << std::endl;
  std::cout << "Q_iso number dofs per cell: " << fe2.n_dofs_per_cell()
            << std::endl;

  FullMatrix<double> projection_matrix(fe1.n_dofs_per_cell(),
                                       fe2.n_dofs_per_cell());
  FETools::get_projection_matrix(fe2, fe1, projection_matrix);

  projection_matrix.print(std::cout);

  projection_matrix = 0.0;
  projection_P0_P1<dim>(projection_matrix);
  projection_matrix.print(std::cout);
}
*/

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  MPI_Comm mpi_communicator(MPI_COMM_WORLD);


  std::unique_ptr<FiniteElement<dim>> fe_coarse;
  std::unique_ptr<FiniteElement<dim>> fe_fine;

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
  fe_fine =
    std::make_unique<FESystem<dim>>(FE_Q_iso_Q1<dim>(n_subdivisions), spacedim);
  DoFHandler<dim> dof_handler_coarse(tria);
  dof_handler_coarse.distribute_dofs(*fe_coarse);

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
      auto cell_index = cell->active_cell_index();
      {
        auto patch = &patches.emplace_back();


        for (auto neighbour_ordered_index : cells_in_patch[vector_cell_index])
          {
            auto &cell_to_add = ordered_cells[neighbour_ordered_index];
            patch->cells.push_back(cell_to_add);
          }
      }
    }

  for (auto current_patch_id : locally_owned_patches)
    {
      AssertIndexRange(current_patch_id, patches.size());
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


          sub_cell++;
        }
    }

  DoFHandler<dim> dh_coarse_patch;
  DoFHandler<dim> dh_fine_patch;

  using VectorType = Vector<double>;

  AffineConstraints<double> internal_boundary_constraints;
  // AffineConstraints<double> local_stiffnes_constraints;

  // TODO: use internal and local constraints to take care fo the boundary of
  // the patch that's not on the boundary of the domain now special number is
  // set to zero so they are treated as one together


  // we are assuming mesh to be created as hyper_cube l 83
  double h = H / (n_subdivisions);

  // create projection matrix from fine to coarse cell (DG)
  FullMatrix<double> projection_matrix(fe_coarse->n_dofs_per_cell(),
                                       fe_fine->n_dofs_per_cell()
                                       //, fe_coarse->n_dofs_per_cell()
  );
  FullMatrix<double> projection_matrixT( // fe_coarse->n_dofs_per_cell(),
    fe_fine->n_dofs_per_cell(),
    fe_coarse->n_dofs_per_cell());
  FETools::get_projection_matrix(*fe_fine, *fe_coarse, projection_matrix);
  FETools::get_projection_matrix(*fe_coarse, *fe_fine, projection_matrixT);
  // projection_P0_P1<dim>(projection_matrix);
  // projection_matrix *= (h * h / 4);
  // this could be done via tensor product
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

      auto Ndofs_coarse = dh_coarse_patch.n_dofs();
      auto Ndofs_fine   = dh_fine_patch.n_dofs();

      internal_boundary_constraints.clear();
      // DoFTools::make_zero_boundary_constraints(dh_fine_patch,
      //                                          0,
      //                                          internal_boundary_constraints);
      internal_boundary_constraints.close();

      // averaging (inverse of P0 mass matrix)
      VectorType valence_coarse(Ndofs_coarse);
      VectorType local_identity_coarse(fe_coarse->n_dofs_per_cell());
      local_identity_coarse = 1.0;

      for (const auto &cell : dh_coarse_patch.active_cell_iterators())
        cell->distribute_local_to_global(local_identity_coarse, valence_coarse);
      for (auto &elem : valence_coarse)
        elem = 1.0 / elem;

      // TODO: const auto project_matrix = [&] (auto &dst, const &src) {}

      const auto project = [&](auto &dst, const auto &src) {
        VectorType vec_local_coarse(fe_coarse->n_dofs_per_cell());
        VectorType vec_local_fine(fe_fine->n_dofs_per_cell());
        VectorType weights(fe_coarse->n_dofs_per_cell());

        for (const auto &cell : current_patch->sub_tria.active_cell_iterators())
          // should be locally_owned ?
          {
            const auto cell_coarse =
              cell->as_dof_handler_iterator(dh_coarse_patch);
            const auto cell_fine = cell->as_dof_handler_iterator(dh_fine_patch);

            cell_coarse->get_dof_values(src, vec_local_coarse);

            cell_coarse->get_dof_values(valence_coarse, weights);
            vec_local_coarse.scale(weights);

            projection_matrix.Tvmult(vec_local_fine, vec_local_coarse);

            cell_fine->distribute_local_to_global(vec_local_fine, dst);
          }
      };

      VectorType P_e_i(Ndofs_fine);
      VectorType e_i(Ndofs_coarse);

      FullMatrix<double> PT(Ndofs_fine, Ndofs_coarse);
      FullMatrix<double> P(Ndofs_fine,
                           Ndofs_coarse); //(Ndofs_coarse, Ndofs_fine);


      // assign rhs
      // TODO: projection that works on matrices!
      // if (false) // project row by row
      {
        for (unsigned int i = 0; i < Ndofs_coarse; ++i)
          {
            e_i    = 0.0;
            P_e_i  = 0.0;
            e_i[i] = 1.0;

            project(P_e_i, e_i);

            for (unsigned int j = 0; j < Ndofs_fine; ++j)
              PT.set(j, i, P_e_i[j]);
          }
      }
      // else // project full amtrix using affine constrain
      {
        std::cout << P.m() << " " << P.n() << std::endl;
        std::cout << fe_fine->n_dofs_per_cell() << std::endl;
        std::cout << projection_matrixT.m() << " " << projection_matrixT.n()
                  << std::endl;
        for (const auto &cell : dh_fine_patch.active_cell_iterators())
          {
            std::vector<types::global_dof_index> local_dof_indices(
              fe_fine->n_dofs_per_cell());
            cell->get_dof_indices(local_dof_indices);

            internal_boundary_constraints.distribute_local_to_global(
              projection_matrix, local_dof_indices, P);
          }
      }

      for (unsigned int i = 0; i < Ndofs_fine; ++i)
        for (unsigned int j = 0; j < Ndofs_coarse; ++j)
          Assert(P(i, j) == PT(j, i), ExcNotImplemented());
    }
}