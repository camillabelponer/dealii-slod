#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q_iso_q1.h>

using namespace dealii;

template<int dim>
class SLOD {
  public:
  private:
    unsigned int oversampling         = 1;
    unsigned int n_subdivisions       = 5;
    unsigned int n_global_refinements = 2;

    std::unique_ptr<FiniteElement<dim>> fe_coarse;
    std::unique_ptr<FiniteElement<dim>> fe_fine;
};

