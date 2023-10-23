#include "full.h"
#include "amen_solve.h"
#include "compression.h"
#include "dmrg_mv.h"
#include "amen_approx.h"


/// Functions from cpp to import in python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tt_full", &full, "TT to full");
  m.def("amen_solve", &amen_solve, "AMEn solve");
  m.def("amen_approx", &AMEn::amen_approx, "AMEn approximate");
  m.def("amen_mv", &AMEn::amen_mv, "AMEn matvec");
  m.def("amen_mv_multiple", &AMEn::amen_mv_multiple, "AMEn matvec with multiple vectors");
  m.def("round_this", &round_this, "Implace rounding");
  m.def("dmrg_mv", &dmrg_mv, "DMRG matrix vector product");
}







