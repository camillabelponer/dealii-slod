#include <deal.II/lac/lapack_full_matrix.h>

using namespace dealii;

int
main()
{
  // initialize vectors
  const unsigned int n_blocks = 3;
  const unsigned int N        = 10;

  std::vector<Vector<double>> PT(n_blocks);
  std::vector<Vector<double>> SinvPT(n_blocks);

  for (unsigned int i = 0; i < n_blocks; ++i)
    {
      PT[i].reinit(N);
      SinvPT[i].reinit(N);
    }

  // perform multivector dot product
  LAPACKFullMatrix<double> matrix(PT.size(), SinvPT.size());

  for (unsigned int i = 0; i < PT.size(); ++i)
    for (unsigned int j = 0; j < SinvPT.size(); ++j)
      matrix.set(i, j, PT[i] * SinvPT[j]);
}