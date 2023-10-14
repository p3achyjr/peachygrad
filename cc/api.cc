#include "api.h"

namespace peachygrad {
namespace py = ::pybind11;

Tensor tensor(py::list py_list, DType dtype) {
  Shape shape;
  py::list clist = py_list;
  while (true) {
    shape[shape.num_dims] = clist.size();
    shape.num_dims++;
    if (clist.size() == 0 || !py::isinstance<py::list>(clist[0])) {
      break;
    }

    clist = clist[0].cast<py::list>();
  }

  Tensor t(dtype, shape);
  auto set_index = [py_list, dtype](const Shape& index, void* it) {
    // find element in list.
    py::object obj = py_list;
    for (size_t i : index) {
      obj = obj.cast<py::list>()[i];
    }

    if (dtype == DType::kI32) {
      int val = obj.cast<int>();
      *((int*)it) = val;
    } else {
      float val = obj.cast<float>();
      *((float*)it) = val;
    }
  };

  GenericIterate(
      t, []() {}, set_index, []() {});
  return t;
}

}  // namespace peachygrad
