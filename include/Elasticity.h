#ifndef dealii_elasticity_h
#define dealii_elasticity_h

#include <LOD.h>



template <int dim, int spacedim = dim >
class ElasticityProblem : public LOD<dim, spacedim>
{
public:
  ElasticityProblem(const LODParameters<dim, spacedim> &par)
    : LOD<dim, spacedim>(par)
    {};

  typedef LOD<dim, spacedim> lod;


protected:

  virtual void
  output_coarse_results()
  {
    lod::N_corrected_patches++;
  };
  virtual void
  output_fine_results() {};
  virtual void
  assemble_stiffness(LA::MPI::SparseMatrix /*<double>*/ &stiffness_matrix,
                     LA::MPI::Vector &                   rhs,
                     const DoFHandler<dim> &             dh,
                     AffineConstraints<double> &         stiffnes_constraints) {};

  virtual void
  assemble_stiffness_patch(SparseMatrix<double> & stiffness_matrix,
                           const DoFHandler<dim> &dh)  {};
};



#endif