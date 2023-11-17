#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q_iso_q1.h>

using namespace dealii;

struct Patch {
  IndexSet cells;
  unsigned int num_basis_vectors;
};

template<int dim>
class SLOD {
  public:
    SLOD(Triangulation<dim> tria);

    void make_fe();
    void create_patches();
    void compute_coarse_basis();
  private:
    void create_mesh_for_patch(unsigned int patch_id, Triangulation<dim> &sub_tria);

    unsigned int oversampling           = 1;
    unsigned int n_subdivisions         = 5;
    unsigned int n_global_refinements   = 2;

    Triangulation<dim> tria;
    DoFHandler<dim> dof_handler;

    std::unique_ptr<FiniteElement<dim>> fe_coarse;
    std::unique_ptr<FiniteElement<dim>> fe_fine;

    // cell_dof_indices[j] contains dofs of cell with id j
    std::vector<std::vector<types::global_dof_index>> cell_dof_indices;
    // TODO: This should be an MPI vector
    std::vector<Patch>                  patches;
};

